"""BLT with a pretrained Qwen3.5 body as the latent (global) transformer.

Purpose: check whether BLT is worth it on Vietnamese by holding the global model fixed to a
strong pretrained LM (Qwen/Qwen3.5-0.8B-Base, Apache-2.0) and only swapping its BPE
embedding/lm_head for BLT's byte encoder/decoder. Layout follows facebookresearch/blt @ 9774ed4
(ByteLatentTransformer.forward with cross_attn_encoder/decoder, patching_mode "entropy",
nb_boe = 0) and the BLT-1B shape ratios from HF `BltConfig` defaults:

    BLT-1B (paper)                    this model
    local dim 1024, k=2  -> 2048      local dim 512, k=2 -> 1024 = Qwen3.5-0.8B hidden
    encoder 1 layer, decoder 9        same
    hash n-grams 3..8, 500002, 1 fn   same by default (see hash_vocab note below)
    global 25L x 2048 (from scratch)  Qwen3.5 text model, 24L x 1024 (pretrained)

What is taken from Qwen: its decoder layers and final norm (Qwen3_5TextModel). Dropped:
embed_tokens (248k x 1024), lm_head, the MTP head and the vision tower. Patch states go in
through `inputs_embeds`; positions are the patch index (Qwen builds text M-RoPE ids itself).

DEVIATIONS from the reference, all deliberate:
- the global model is causal over the whole row, not block-causal per document (Qwen's
  Gated DeltaNet layers take no block mask); pack one document per row, or accept it;
- Qwen's final RMSNorm is kept (the official GlobalTransformer has none) because the
  pretrained layers were trained to feed it;
- dense SDPA masks instead of xformers/flex (see blt_layers.local_block_causal_mask).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vislm.backbones.modules.blt_layers import (
    PAD_ID,
    LocalDecoder,
    LocalEncoder,
    patch_ids_from_lengths,
)

QWEN_BASE = "Qwen/Qwen3.5-0.8B-Base"


@dataclass
class BLTQwenConfig:
    qwen_name_or_path: str = QWEN_BASE
    dim_local: int = 512
    n_layers_encoder: int = 1
    n_layers_decoder: int = 9
    n_heads_local: int = 8
    cross_attn_nheads: int = 8
    cross_attn_k: int = 2
    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = True
    # None = a byte only meets its own patch (HF BLT-1B); debug.yaml uses 512.
    cross_attn_window_encoder: Optional[int] = None
    cross_attn_window_decoder: Optional[int] = None
    cross_attn_double_residual: bool = True  # official quirk, see blt_layers.CrossAttention
    local_attention_window: Optional[int] = 512
    hash_ngram_sizes: Tuple[int, ...] = field(default_factory=lambda: (3, 4, 5, 6, 7, 8))
    # 500002 is the BLT-1B value: 6 tables x 500002 x 512 = 1.54B params, more than the Qwen
    # body. With fp32 Adam that alone is ~25GB - lower it for a 24GB card and say so.
    hash_vocab: int = 500002
    hash_funcs: int = 1
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    multiple_of: int = 256


class QwenLatentTransformer(nn.Module):
    """Qwen3_5TextModel minus its token embedding, driven by inputs_embeds."""

    def __init__(self, text_model: nn.Module):
        super().__init__()
        text_model.embed_tokens = None
        self.model = text_model
        self.hidden_size = text_model.config.hidden_size

    @classmethod
    def from_pretrained(cls, name_or_path: str = QWEN_BASE, **kwargs) -> "QwenLatentTransformer":
        from transformers import Qwen3_5ForCausalLM

        # Loads the text weights of the multimodal checkpoint; mtp.* and model.visual.* are
        # skipped by the class's _keys_to_ignore_on_load_unexpected.
        lm = Qwen3_5ForCausalLM.from_pretrained(name_or_path, **kwargs)
        return cls(lm.model)

    @classmethod
    def from_config(cls, text_config) -> "QwenLatentTransformer":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        return cls(Qwen3_5TextModel(text_config))

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        out = self.model(inputs_embeds=patch_embeds.to(dtype), use_cache=False)
        return out.last_hidden_state.to(patch_embeds.dtype)


class BLTQwen(nn.Module):
    def __init__(self, cfg: BLTQwenConfig, global_model: QwenLatentTransformer):
        super().__init__()
        self.cfg = cfg
        d, k = cfg.dim_local, cfg.cross_attn_k
        self.local_encoder = LocalEncoder(
            dim=d, n_layers=cfg.n_layers_encoder, n_heads=cfg.n_heads_local, cross_attn_k=k,
            cross_attn_nheads=cfg.cross_attn_nheads, sliding_window=cfg.local_attention_window,
            hash_ngram_sizes=cfg.hash_ngram_sizes, hash_vocab=cfg.hash_vocab, hash_funcs=cfg.hash_funcs,
            cross_attn_all_layers=cfg.cross_attn_all_layers_encoder,
            cross_attn_window=cfg.cross_attn_window_encoder, rope_theta=cfg.rope_theta,
            multiple_of=cfg.multiple_of, norm_eps=cfg.norm_eps, double_residual=cfg.cross_attn_double_residual,
        )
        self.global_model = global_model
        g = global_model.hidden_size
        # GlobalTransformer.token_embedding_projection: only when dim_local*k != global dim.
        self.global_in_proj = nn.Linear(d * k, g, bias=False) if d * k != g else None
        if self.global_in_proj is not None:
            std = (d * k) ** -0.5
            nn.init.trunc_normal_(self.global_in_proj.weight, std=std, a=-3 * std, b=3 * std)
        self.local_decoder = LocalDecoder(
            dim=d, n_layers=cfg.n_layers_decoder, n_heads=cfg.n_heads_local, dim_global=g, cross_attn_k=k,
            cross_attn_nheads=cfg.cross_attn_nheads, sliding_window=cfg.local_attention_window,
            cross_attn_all_layers=cfg.cross_attn_all_layers_decoder,
            cross_attn_window=cfg.cross_attn_window_decoder, rope_theta=cfg.rope_theta,
            multiple_of=cfg.multiple_of, norm_eps=cfg.norm_eps, double_residual=cfg.cross_attn_double_residual,
        )

    @classmethod
    def from_qwen_pretrained(cls, cfg: Optional[BLTQwenConfig] = None, **qwen_kwargs) -> "BLTQwen":
        cfg = cfg or BLTQwenConfig()
        return cls(cfg, QwenLatentTransformer.from_pretrained(cfg.qwen_name_or_path, **qwen_kwargs))

    def forward(self, tokens: torch.Tensor, patch_lengths: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        """tokens: [bs, N] byte ids (OFFSET'd). patch_lengths: [bs, P], right-zero-padded,
        each row summing to N+1 with a 1-byte first patch (blt_patcher, include_next_token).

        Encoder patch j covers its bytes; latent j (global output j) has seen bytes up to the
        end of patch j. The decoder uses patch_lengths[:, 1:], which puts byte i in decoder
        patch j exactly when patch j ends at or before i - the lag fix: logits at i see byte
        i and nothing after it."""
        bs, n = tokens.shape
        num_patches = patch_lengths.shape[1]
        assert torch.all(patch_lengths[:, 0] == 1), "first patch must be exactly 1 byte"
        assert torch.all(patch_lengths.sum(-1) == n + 1), "patch lengths must sum to N+1"

        patch_ids = patch_ids_from_lengths(patch_lengths, n)
        h_enc, h_cross = self.local_encoder(tokens, patch_ids, num_patches)

        h = h_cross.reshape(bs, num_patches, -1)
        if self.global_in_proj is not None:
            h = self.global_in_proj(h)
        h = self.global_model(h)

        decoder_patch_ids = patch_ids_from_lengths(patch_lengths[:, 1:], n)
        logits = self.local_decoder(tokens, h_enc, h, decoder_patch_ids)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD_ID)
        return logits, loss

    @staticmethod
    def bits_per_byte(loss_nats: torch.Tensor) -> torch.Tensor:
        return loss_nats / math.log(2)

    def num_params(self, exclude_hash: bool = False) -> int:
        skip = {id(p) for p in self.local_encoder.hash_embeddings.parameters()} if exclude_hash else set()
        return sum(p.numel() for p in self.parameters() if id(p) not in skip)
