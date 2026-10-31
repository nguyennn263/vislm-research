"""BLT building blocks (Pagnoni et al., 2024, arXiv:2412.09871) for the BLT-on-Qwen testbed
(vislm/backbones/models/blt_qwen.py).

Re-implemented from scratch to match facebookresearch/blt @ 9774ed4 op-for-op
(bytelatent/model/{blt,local_models,latent_transformer,utils}.py, bytelatent/base_transformer.py).
The official code is CC BY-NC, so nothing is copied; every semantic choice follows it and the
reference is given next to each piece. Anything that differs on purpose is marked DEVIATION.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Byte tokenizer (bytelatent/tokenizers/{constants,blt_tokenizer}.py) -----------------
BOE_ID, BOS_ID, EOS_ID, BPE_ID = 0, 1, 2, 3
OFFSET = 4
PAD_ID = -1  # never fed to an embedding; used as ignore_index for targets
BYTE_VOCAB_SIZE = 256 + OFFSET


def encode_bytes(text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
    ids = [b + OFFSET for b in text.encode("utf-8")]
    return ([BOS_ID] if add_bos else []) + ids + ([EOS_ID] if add_eos else [])


def decode_bytes(ids: Sequence[int]) -> str:
    return bytes(i - OFFSET for i in ids if i >= OFFSET).decode("utf-8", errors="replace")


# --- Hash n-gram embeddings (blt.py: byte_group_hash_function, compute_hash_embeddings) -----
HASH_PRIMES = [
    1000000007, 5915587277, 1500450271, 3267000013, 5754853343, 4093082899,
    9576890767, 3628273133, 2860486313, 5463458053, 3367900313,
]


def byte_ngram_hash_ids(tokens: torch.Tensor, n: int, func_nb: int, num_buckets: int) -> torch.Tensor:
    """Hash of the n-gram ending at each position (left-padded with n-1 zeros = BOE).
    Rolling polynomial sum(t_i * prime^i) in int64 with the same silent wraparound as the
    official code, then `% num_buckets`. [bs, seq] -> [bs, seq]."""
    with torch.no_grad():
        pad = tokens.new_zeros(tokens.shape[0], n - 1, dtype=torch.int64)
        windows = torch.cat([pad, tokens.to(torch.int64)], dim=1).unfold(1, n, 1)
        prime = torch.tensor(HASH_PRIMES[func_nb], dtype=torch.int64, device=tokens.device)
        powers = torch.stack([prime**i for i in range(n)])
        return (windows * powers).sum(-1) % num_buckets


class HashNgramEmbedding(nn.Module):
    """One table per (hash function, n). Returns the plain SUM of the looked-up rows.
    The paper's eq. normalises by the number of n-gram sizes + 1; the official code does
    not, and we follow the code."""

    def __init__(self, dim: int, ngram_sizes: Sequence[int], num_buckets: int, num_funcs: int = 1):
        super().__init__()
        self.ngram_sizes = list(ngram_sizes)
        self.num_buckets = num_buckets
        self.num_funcs = num_funcs
        self.tables = nn.ModuleList(
            nn.Embedding(num_buckets, dim) for _ in range(num_funcs * len(self.ngram_sizes))
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out, i = 0, 0
        for func_nb in range(self.num_funcs):
            for n in self.ngram_sizes:
                out = out + self.tables[i](byte_ngram_hash_ids(tokens, n, func_nb, self.num_buckets))
                i += 1
        return out


# --- Patch bookkeeping (blt.py: patch_ids_from_lengths, create_patch_mask_from_ids, cross_attn_mask)
def patch_ids_from_lengths(patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """[bs, P] right-zero-padded lengths -> [bs, seq_len] patch index of every position."""
    zeros = patch_lengths.new_zeros(patch_lengths.shape[0], 1)
    cum = torch.cat([zeros, patch_lengths.cumsum(-1)], dim=-1)
    pos = torch.arange(seq_len, device=patch_lengths.device)
    return (cum.unsqueeze(-1) <= pos).sum(-2) - 1


def cross_attn_mask(
    patch_ids: torch.Tensor,
    num_patches: int,
    k: int,
    patches_as_queries: bool,
    window: Optional[int] = None,
) -> torch.Tensor:
    """Boolean mask (True = attend), shape [bs, 1, q, kv].
    patches_as_queries (encoder): q = num_patches*k patch slots, kv = bytes.
    otherwise (decoder): q = bytes, kv = num_patches*k latent slots.
    window None -> a byte only meets its own patch; window w -> patches q-w+1..q."""
    p = torch.arange(num_patches, device=patch_ids.device)
    if patches_as_queries:
        q_ids, kv_ids = p.view(1, -1, 1), patch_ids.unsqueeze(1)
    else:
        q_ids, kv_ids = patch_ids.unsqueeze(-1), p.view(1, 1, -1)
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask.repeat_interleave(k, dim=1 if patches_as_queries else 2).unsqueeze(1)


def local_block_causal_mask(tokens: torch.Tensor, window: Optional[int], eos_id: int = EOS_ID) -> torch.Tensor:
    """Sliding-window causal mask that also stops at document ends (an EOS closes its own
    document), i.e. xformers BlockDiagonalCausalMask.from_seqlens(...).make_local_attention(window)
    as built by utils.create_causal_mask(attn_bias_type="local_block_causal"). [bs, 1, q, kv].
    DEVIATION: a dense bool mask for SDPA instead of xformers/flex block masks - same values,
    O(seq^2) memory, fine at the testbed's byte lengths."""
    n = tokens.shape[1]
    pos = torch.arange(n, device=tokens.device)
    dist = pos.view(-1, 1) - pos.view(1, -1)
    mask = dist >= 0
    if window is not None:
        mask = mask & (dist < window)
    is_eos = (tokens == eos_id).long()
    doc = is_eos.cumsum(-1) - is_eos
    return (mask.unsqueeze(0) & (doc.unsqueeze(-1) == doc.unsqueeze(-2))).unsqueeze(1)


def max_pool_patches(h: torch.Tensor, patch_ids: torch.Tensor, num_patches: int) -> torch.Tensor:
    """utils.patch_reduce(..., "amax"): per-patch max over its bytes; empty patches stay 0."""
    bs, _, dim = h.shape
    out = h.new_zeros(bs, num_patches, dim)
    index = patch_ids.unsqueeze(-1).expand(-1, -1, dim)
    return out.scatter_reduce(1, index, h, reduce="amax", include_self=False)


# --- Transformer pieces (base_transformer.py) ---------------------------------------------
def _trunc_normal_(w: torch.Tensor, std: float) -> None:
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)


class RotaryEmbedding(nn.Module):
    """Llama-style RoPE on adjacent (even, odd) pairs, as base_transformer.apply_rotary_emb."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [bs, heads, seq, head_dim]."""
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        bs, n, _ = x.shape
        q = self.wq(x).view(bs, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bs, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bs, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(bs, n, -1))

    def reset_parameters(self, dim: int):
        for w in (self.wq, self.wk, self.wv, self.wo):
            _trunc_normal_(w.weight, dim**-0.5)


class FeedForward(nn.Module):
    """SwiGLU; hidden = int(2*4*dim/3) rounded up to multiple_of (FeedForward.__init__)."""

    def __init__(self, dim: int, multiple_of: int = 256):
        super().__init__()
        hidden = int(2 * 4 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, dim: int):
        _trunc_normal_(self.w1.weight, dim**-0.5)
        _trunc_normal_(self.w3.weight, dim**-0.5)
        _trunc_normal_(self.w2.weight, self.hidden_dim**-0.5)


class TransformerBlock(nn.Module):
    """Pre-norm RMSNorm block. Init = reset_parameters with init_std_factor "disabled"
    (the BaseTransformerArgs default)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 multiple_of: int = 256, norm_eps: float = 1e-5):
        super().__init__()
        self.attention = SelfAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, multiple_of)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attention.reset_parameters(dim)
        self.feed_forward.reset_parameters(dim)

    def forward(self, x, cos, sin, mask):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class CrossAttention(nn.Module):
    """latent_transformer.CrossAttention: RMSNorm on q and kv, no RoPE, n_kv_heads = n_heads.

    Faithful quirk: the official forward returns `x + out` and BOTH callers add the residual
    again (`patch_embeds + cross_attn(x=patch_embeds, ...)`, `h = h + cross_attn(x=h, ...)`),
    so the stream is 2*x + out per cross-attention layer. The HF port keeps it too. We keep
    it by default (double_residual=True) so results stay comparable to the reference.

    Rows with no allowed key (zero-length padding patches) output 0, matching flex
    attention; plain SDPA would give NaN there."""

    def __init__(self, dim: int, n_heads: int, norm_eps: float = 1e-5, double_residual: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.double_residual = double_residual
        self.cross_attn_norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = nn.RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        for w in (self.wq, self.wk, self.wv, self.wo):
            _trunc_normal_(w.weight, dim**-0.5)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bs, nq, _ = x.shape
        nkv = kv.shape[1]
        xn, kvn = self.cross_attn_norm_q(x), self.cross_attn_norm_kv(kv)
        q = self.wq(xn).view(bs, nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(kvn).view(bs, nkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kvn).view(bs, nkv, self.n_heads, self.head_dim).transpose(1, 2)
        has_key = mask.any(-1, keepdim=True)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask | ~has_key)
        out = (out * has_key).transpose(1, 2).reshape(bs, nq, -1)
        out = self.wo(out)
        return x + out if self.double_residual else out


# --- Local encoder / decoder (local_models.py) ---------------------------------------------
class LocalEncoder(nn.Module):
    """bytes -> (per-byte states h, k patch slots per patch).
    tok_embeddings + hash n-grams -> transformer layers (local_block_causal, sliding window)
    -> at the last layer (or every layer) cross-attention with patch slots as queries,
    initialised by max-pooling each patch and projecting dim -> dim*k."""

    def __init__(self, dim: int, n_layers: int, n_heads: int, cross_attn_k: int, cross_attn_nheads: int,
                 sliding_window: Optional[int], hash_ngram_sizes: Sequence[int], hash_vocab: int,
                 hash_funcs: int = 1, cross_attn_all_layers: bool = False, cross_attn_window: Optional[int] = None,
                 rope_theta: float = 500000.0, multiple_of: int = 256, norm_eps: float = 1e-5,
                 double_residual: bool = True):
        super().__init__()
        self.dim, self.k = dim, cross_attn_k
        self.sliding_window = sliding_window
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_window = cross_attn_window
        self.tok_embeddings = nn.Embedding(BYTE_VOCAB_SIZE, dim)
        self.hash_embeddings = HashNgramEmbedding(dim, hash_ngram_sizes, hash_vocab, hash_funcs)
        self.rope = RotaryEmbedding(dim // n_heads, rope_theta)
        self.layers = nn.ModuleList(
            TransformerBlock(dim, n_heads, multiple_of=multiple_of, norm_eps=norm_eps) for _ in range(n_layers)
        )
        self.patch_embedding_projection = nn.Linear(dim, dim * cross_attn_k, bias=False)
        self.cross_attn_layers = nn.ModuleList(
            CrossAttention(dim, cross_attn_nheads, norm_eps, double_residual)
            for _ in range(n_layers if cross_attn_all_layers else 1)
        )
        std = dim**-0.5  # LocalModelBase.init_weights + ByteLatentTransformer.init_weights
        _trunc_normal_(self.tok_embeddings.weight, std)
        for t in self.hash_embeddings.tables:
            _trunc_normal_(t.weight, std)
        _trunc_normal_(self.patch_embedding_projection.weight, std)

    def forward(self, tokens: torch.Tensor, patch_ids: torch.Tensor, num_patches: int):
        bs, n = tokens.shape
        h = self.tok_embeddings(tokens) + self.hash_embeddings(tokens)
        mask = local_block_causal_mask(tokens, self.sliding_window)
        cos, sin = self.rope(n, tokens.device)
        cross_mask = cross_attn_mask(patch_ids, num_patches, self.k, True, self.cross_attn_window)
        patch_embeds = None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, mask)
            if i == last or self.cross_attn_all_layers:
                if patch_embeds is None:
                    pooled = max_pool_patches(h, patch_ids, num_patches)
                    patch_embeds = self.patch_embedding_projection(pooled).reshape(bs, num_patches * self.k, self.dim)
                ca = self.cross_attn_layers[i if self.cross_attn_all_layers else 0]
                patch_embeds = patch_embeds + ca(patch_embeds, h, cross_mask)
        return h, patch_embeds


class LocalDecoder(nn.Module):
    """(per-byte encoder states, patch latents) -> byte logits.
    Latents projected dim_global -> dim*k; before each layer (all layers, or only the
    first) the bytes cross-attend to the k slots of their decoder patch; final RMSNorm and
    Linear(dim, 260), logits in fp32."""

    def __init__(self, dim: int, n_layers: int, n_heads: int, dim_global: int, cross_attn_k: int,
                 cross_attn_nheads: int, sliding_window: Optional[int], cross_attn_all_layers: bool = True,
                 cross_attn_window: Optional[int] = None, rope_theta: float = 500000.0,
                 multiple_of: int = 256, norm_eps: float = 1e-5, double_residual: bool = True):
        super().__init__()
        self.dim, self.k = dim, cross_attn_k
        self.sliding_window = sliding_window
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_window = cross_attn_window
        self.rope = RotaryEmbedding(dim // n_heads, rope_theta)
        self.patch_embedding_projection = nn.Linear(dim_global, dim * cross_attn_k, bias=False)
        self.layers = nn.ModuleList(
            TransformerBlock(dim, n_heads, multiple_of=multiple_of, norm_eps=norm_eps) for _ in range(n_layers)
        )
        self.cross_attn_layers = nn.ModuleList(
            CrossAttention(dim, cross_attn_nheads, norm_eps, double_residual)
            for _ in range(n_layers if cross_attn_all_layers else 1)
        )
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, BYTE_VOCAB_SIZE, bias=False)
        _trunc_normal_(self.patch_embedding_projection.weight, dim_global**-0.5)
        _trunc_normal_(self.output.weight, dim**-0.5)

    def forward(self, tokens: torch.Tensor, embeds: torch.Tensor, latents: torch.Tensor,
                decoder_patch_ids: torch.Tensor) -> torch.Tensor:
        bs, n = tokens.shape
        num_patches = latents.shape[1]
        kv = self.patch_embedding_projection(latents).reshape(bs, num_patches * self.k, self.dim)
        cross_mask = cross_attn_mask(decoder_patch_ids, num_patches, self.k, False, self.cross_attn_window)
        mask = local_block_causal_mask(tokens, self.sliding_window)
        cos, sin = self.rope(n, tokens.device)
        h = embeds
        for i, layer in enumerate(self.layers):
            if i == 0 or self.cross_attn_all_layers:
                h = h + self.cross_attn_layers[i](h, kv, cross_mask)
            h = layer(h, cos, sin, mask)
        return self.output(self.norm(h)).float()


class ByteEntropyLM(nn.Module):
    """Entropy model for patching: a plain byte LM over the 260-id vocab with sliding-window
    local_block_causal attention (configs/entropy_model.yaml: dim 768, 14 layers, 12 heads,
    window 512). Train once on Vietnamese bytes, then freeze."""

    def __init__(self, dim: int = 768, n_layers: int = 14, n_heads: int = 12, sliding_window: Optional[int] = 512,
                 rope_theta: float = 10000.0, multiple_of: int = 256, norm_eps: float = 1e-5):
        super().__init__()
        self.sliding_window = sliding_window
        self.tok_embeddings = nn.Embedding(BYTE_VOCAB_SIZE, dim)
        self.rope = RotaryEmbedding(dim // n_heads, rope_theta)
        self.layers = nn.ModuleList(
            TransformerBlock(dim, n_heads, multiple_of=multiple_of, norm_eps=norm_eps) for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, BYTE_VOCAB_SIZE, bias=False)
        _trunc_normal_(self.tok_embeddings.weight, dim**-0.5)
        _trunc_normal_(self.output.weight, dim**-0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        mask = local_block_causal_mask(tokens, self.sliding_window)
        cos, sin = self.rope(tokens.shape[1], tokens.device)
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        return self.output(self.norm(h)).float()
