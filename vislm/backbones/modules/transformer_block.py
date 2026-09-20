"""Standard pre-LN Transformer block (Arm A baseline)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """attn_mask: optional bool mask [b, 1, t, t] (True = attend), e.g. combined
        causal+padding for a ragged batch (see blt_lm.py's batched local decoder). When
        omitted, plain causal attention is used."""
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttention(nn.Module):
    """query attends over kv - used by the BLT local encoder (Arm B/C's cross_attention
    mode) to pool bytes in a patch into one patch vector, matching the real BLT paper
    (query=patch summary, key/value=byte representations) instead of plain mean-pooling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """query: [b, tq, c]. kv: [b, tk, c]. key_padding_mask: [b, tk] bool, True=valid."""
        b, tq, c = query.shape
        tk = kv.shape[1]
        q = self.q_proj(query).view(b, tq, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(kv).chunk(2, dim=-1)
        k = k.view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, tk, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].expand(b, 1, tq, tk)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, tq, c)
        return self.out_proj(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = CrossAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor = None):
        query = query + self.attn(self.ln_q(query), self.ln_kv(kv), key_padding_mask)
        query = query + self.mlp(self.ln2(query))
        return query


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
