"""Arm B - Byte Latent Transformer: entropy-based dynamic patching + hierarchical
local encoder / latent transformer / local decoder (Pagnoni et al., 2024,
facebookresearch/blt). Scoped-down reference implementation for the small-scale
ablation in plans/PLAN.md question 1.3 - NOT the paper's full training codebase:
- local encoder is mean-pooling instead of cross-attention over byte n-grams.
- still loops over the batch dim in Python (one sequence at a time - patch counts
  differ per sequence, so this isn't trivially vectorized); the local decoder,
  the main cost, IS batched across all patches within a sequence (see
  build_causal_padding_mask) so it's one transformer call per sequence, not one
  per patch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vislm.backbones.modules.entropy_model import BYTE_VOCAB_SIZE, ByteEntropyModel
from vislm.backbones.modules.patching import entropy_to_patch_lengths
from vislm.backbones.modules.transformer_block import TransformerBlock, count_params


def build_causal_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """lengths: [n] valid length per row. Returns bool mask [n, 1, max_len, max_len],
    True = attend: causal AND key position is within that row's valid length."""
    device = lengths.device
    idx = torch.arange(max_len, device=device)
    valid_key = idx.unsqueeze(0) < lengths.unsqueeze(1)  # [n, max_len]
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # [max_len, max_len], True where key<=query
    mask = causal.unsqueeze(0) & valid_key.unsqueeze(1)  # [n, max_len, max_len]
    return mask.unsqueeze(1)  # [n, 1, max_len, max_len]


class BLTLanguageModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        entropy_d_model: int = 128,
        entropy_n_layers: int = 4,
        entropy_n_heads: int = 4,
        entropy_threshold: float = 1.5,
        max_patch_len: int = 16,
        n_decoder_layers: int = 2,
        seed_boundaries_fn=None,  # Arm C hook: (byte_ids_1d, lengths, max_patch_len) -> lengths
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.entropy_threshold = entropy_threshold
        self.max_patch_len = max_patch_len
        self.seed_boundaries_fn = seed_boundaries_fn

        self.entropy_model = ByteEntropyModel(
            entropy_d_model, entropy_n_layers, entropy_n_heads, max_seq_len
        )

        self.byte_emb = nn.Embedding(BYTE_VOCAB_SIZE, d_model)
        self.local_encoder_proj = nn.Linear(d_model, d_model)

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
        with torch.no_grad():
            entropy = self.entropy_model.next_byte_entropy(byte_ids_1d.unsqueeze(0))[0]
        lengths = entropy_to_patch_lengths(entropy, self.entropy_threshold, self.max_patch_len)
        if self.seed_boundaries_fn is not None:
            lengths = self.seed_boundaries_fn(byte_ids_1d, lengths, self.max_patch_len)
        return lengths

    def forward(self, byte_ids: torch.Tensor, targets: torch.Tensor = None):
        b, t = byte_ids.shape
        device = byte_ids.device
        all_logits = torch.zeros(b, t, BYTE_VOCAB_SIZE, device=device)

        for bi in range(b):
            ids_1d = byte_ids[bi]
            lengths = self._patch_lengths_for(ids_1d)
            emb = self.byte_emb(ids_1d)  # [t, d_model]

            # local encoder: mean-pool byte embeddings within each patch
            patch_vecs, patch_spans = [], []
            offset = 0
            for length in lengths:
                span = emb[offset : offset + length]
                patch_vecs.append(span.mean(dim=0))
                patch_spans.append((offset, length))
                offset += length
            patch_vecs = self.local_encoder_proj(torch.stack(patch_vecs))  # [n_patches, d_model]

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
        return self.num_params() - count_params(self.entropy_model)
