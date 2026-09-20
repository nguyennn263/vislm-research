"""Arm B/C/D - Byte Latent Transformer: dynamic patching + hierarchical local encoder /
latent transformer / local decoder (Pagnoni et al., 2024, facebookresearch/blt).
Scoped-down reference implementation for the small-scale ablation in plans/PLAN.md
question 1.3 - NOT the paper's full training codebase, but with two of its
simplifications now made optional (see phases/phase-2-small-train.md for why):

- `boundary_mode`: "entropy" (Arm B/C - a small frozen byte LM's next-byte entropy
  decides patch boundaries) or "bpe_guided" (Arm D - patch boundaries come from a real
  BPE tokenizer's token spans instead, borrowing its proven chunking without paying for
  its large vocab/embedding table - byte vocab stays 256 either way).
- `encoder_type`: "mean_pool" (original scoped-down version) or "cross_attention"
  (matches the real BLT paper's local encoder: patch summary as query, byte
  representations as key/value).

Still loops over the batch dim in Python (one sequence at a time - patch counts differ
per sequence); the local encoder and decoder are each batched across all patches within
a sequence (one transformer call per sequence, not one per patch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vislm.backbones.modules.entropy_model import BYTE_VOCAB_SIZE, ByteEntropyModel
from vislm.backbones.modules.patching import bpe_guided_patch_lengths, entropy_to_patch_lengths
from vislm.backbones.modules.transformer_block import (
    CrossAttentionBlock,
    TransformerBlock,
    count_params,
)


def build_causal_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """lengths: [n] valid length per row. Returns bool mask [n, 1, max_len, max_len],
    True = attend: causal AND key position is within that row's valid length."""
    device = lengths.device
    idx = torch.arange(max_len, device=device)
    valid_key = idx.unsqueeze(0) < lengths.unsqueeze(1)  # [n, max_len]
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # [max_len, max_len], True where key<=query
    mask = causal.unsqueeze(0) & valid_key.unsqueeze(1)  # [n, max_len, max_len]
    return mask.unsqueeze(1)  # [n, 1, max_len, max_len]


def build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """lengths: [n] valid length per row. Returns bool mask [n, max_len], True=valid -
    for the (non-causal) local encoder, which may look at every byte in its patch."""
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


class BLTLanguageModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        max_patch_len: int = 16,
        boundary_mode: str = "entropy",  # "entropy" | "bpe_guided"
        entropy_d_model: int = 128,
        entropy_n_layers: int = 4,
        entropy_n_heads: int = 4,
        entropy_threshold: float = 1.5,
        bpe_tokenizer_name: str = None,  # required if boundary_mode == "bpe_guided"
        encoder_type: str = "mean_pool",  # "mean_pool" | "cross_attention"
        n_encoder_layers: int = 2,  # used only if encoder_type == "cross_attention"
        n_decoder_layers: int = 2,
        seed_boundaries_fn=None,  # Arm C hook: (byte_ids_1d, lengths, max_patch_len) -> lengths
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_patch_len = max_patch_len
        self.boundary_mode = boundary_mode
        self.entropy_threshold = entropy_threshold
        self.encoder_type = encoder_type
        self.seed_boundaries_fn = seed_boundaries_fn

        if boundary_mode == "entropy":
            self.entropy_model = ByteEntropyModel(
                entropy_d_model, entropy_n_layers, entropy_n_heads, max_seq_len
            )
        elif boundary_mode == "bpe_guided":
            from transformers import AutoTokenizer

            assert bpe_tokenizer_name, "bpe_guided boundary_mode needs bpe_tokenizer_name"
            self.entropy_model = None
            self._bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_tokenizer_name)
        else:
            raise ValueError(f"unknown boundary_mode {boundary_mode!r}")

        self.byte_emb = nn.Embedding(BYTE_VOCAB_SIZE, d_model)

        if encoder_type == "mean_pool":
            self.local_encoder_proj = nn.Linear(d_model, d_model)
        elif encoder_type == "cross_attention":
            self.encoder_query_proj = nn.Linear(d_model, d_model)
            self.encoder_blocks = nn.ModuleList(
                [CrossAttentionBlock(d_model, n_heads) for _ in range(n_encoder_layers)]
            )
        else:
            raise ValueError(f"unknown encoder_type {encoder_type!r}")

        self.patch_pos_emb = nn.Embedding(max_seq_len, d_model)
        self.latent_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.latent_ln = nn.LayerNorm(d_model)

        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_decoder_layers)]
        )
        self.decoder_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, BYTE_VOCAB_SIZE, bias=False)
        self.head.weight = self.byte_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _patch_lengths_for(self, byte_ids_1d: torch.Tensor):
        if self.boundary_mode == "bpe_guided":
            lengths = bpe_guided_patch_lengths(byte_ids_1d, self._bpe_tokenizer, self.max_patch_len)
        else:
            with torch.no_grad():
                entropy = self.entropy_model.next_byte_entropy(byte_ids_1d.unsqueeze(0))[0]
            lengths = entropy_to_patch_lengths(entropy, self.entropy_threshold, self.max_patch_len)
        if self.seed_boundaries_fn is not None:
            lengths = self.seed_boundaries_fn(byte_ids_1d, lengths, self.max_patch_len)
        return lengths

    def _encode_patches(self, emb: torch.Tensor, patch_spans, device):
        """emb: [t, d_model] byte embeddings for one sequence. Returns [n_patches, d_model]."""
        d_model = emb.size(-1)
        n_patches = len(patch_spans)

        if self.encoder_type == "mean_pool":
            patch_vecs = torch.stack([emb[o : o + l].mean(dim=0) for o, l in patch_spans])
            return self.local_encoder_proj(patch_vecs)

        # cross_attention: batch all patches in this sequence into one padded call
        max_len = max(length for _, length in patch_spans)
        lengths_t = torch.tensor([length for _, length in patch_spans], device=device)
        enc_in = torch.zeros(n_patches, max_len, d_model, device=device)
        query = torch.zeros(n_patches, 1, d_model, device=device)
        for pi, (offset, length) in enumerate(patch_spans):
            span = emb[offset : offset + length]
            enc_in[pi, :length] = span
            query[pi, 0] = span.mean(dim=0)  # cheap, reasonable query init
        query = self.encoder_query_proj(query)
        padding_mask = build_padding_mask(lengths_t, max_len)
        for block in self.encoder_blocks:
            query = block(query, enc_in, key_padding_mask=padding_mask)
        return query.squeeze(1)

    def forward(self, byte_ids: torch.Tensor, targets: torch.Tensor = None):
        b, t = byte_ids.shape
        device = byte_ids.device
        all_logits = torch.zeros(b, t, BYTE_VOCAB_SIZE, device=device)

        for bi in range(b):
            ids_1d = byte_ids[bi]
            lengths = self._patch_lengths_for(ids_1d)
            emb = self.byte_emb(ids_1d)  # [t, d_model]

            patch_spans, offset = [], 0
            for length in lengths:
                patch_spans.append((offset, length))
                offset += length

            patch_vecs = self._encode_patches(emb, patch_spans, device)  # [n_patches, d_model]

            # latent transformer: causal over patches (the compute-matched component)
            n_patches = patch_vecs.size(0)
            pos = torch.arange(n_patches, device=device)
            x = (patch_vecs + self.patch_pos_emb(pos)).unsqueeze(0)
            for block in self.latent_blocks:
                x = block(x)
            latent_out = self.latent_ln(x)[0]  # [n_patches, d_model]

            # local decoder: patch i's bytes are predicted from patch (i-1)'s latent
            # output (causal across patches) + autoregressively within the patch.
            # Batched across all patches in this sequence via padding + a combined
            # causal/padding mask, instead of one transformer call per patch.
            d_model = emb.size(-1)
            max_len = max(length for _, length in patch_spans)
            lengths_t = torch.tensor([length for _, length in patch_spans], device=device)
            dec_in = torch.zeros(n_patches, max_len, d_model, device=device)
            for pi, (offset, length) in enumerate(patch_spans):
                prev_latent = latent_out[pi - 1] if pi > 0 else torch.zeros_like(latent_out[0])
                in_bytes = emb[offset : offset + length - 1]
                dec_in[pi, :length] = torch.cat([prev_latent.unsqueeze(0), in_bytes], dim=0)

            attn_mask = build_causal_padding_mask(lengths_t, max_len)
            seq = dec_in
            for block in self.decoder_blocks:
                seq = block(seq, attn_mask=attn_mask)
            seq = self.decoder_ln(seq)
            patch_logits = self.head(seq)  # [n_patches, max_len, vocab]

            for pi, (offset, length) in enumerate(patch_spans):
                all_logits[bi, offset : offset + length] = patch_logits[pi, :length]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(all_logits.reshape(-1, BYTE_VOCAB_SIZE), targets.reshape(-1))
        return all_logits, loss

    def num_params(self) -> int:
        return count_params(self)

    def num_latent_params(self) -> int:
        """Params outside the entropy model - what should be compute-matched to Arm A."""
        entropy_params = count_params(self.entropy_model) if self.entropy_model is not None else 0
        return self.num_params() - entropy_params
