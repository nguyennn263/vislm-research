"""Arm A - standard GPT-style causal LM, the compute-matched baseline for Pillar 1."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vislm.backbones.modules.transformer_block import TransformerBlock, count_params


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        b, t = input_ids.shape
        assert t <= self.max_seq_len, f"sequence length {t} > max_seq_len {self.max_seq_len}"
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return count_params(self)
