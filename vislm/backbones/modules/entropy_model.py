"""Small byte-level causal LM used only to score next-byte entropy for patching
(Arm B/C, plans/PLAN.md question 1.3). Trained briefly then frozen, matching the BLT
paper's use of a separately-trained, frozen entropy model (Pagnoni et al., 2024)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vislm.backbones.modules.transformer_block import TransformerBlock, count_params

BYTE_VOCAB_SIZE = 256


class ByteEntropyModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(BYTE_VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, BYTE_VOCAB_SIZE, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, byte_ids: torch.Tensor, targets: torch.Tensor = None):
        b, t = byte_ids.shape
        pos = torch.arange(t, device=byte_ids.device)
        x = self.tok_emb(byte_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def next_byte_entropy(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """P(byte_{t+1} | byte_{1..t}) entropy per position. Shape [b, t]."""
        logits, _ = self.forward(byte_ids)
        probs = torch.softmax(logits.float(), dim=-1)
        return -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)

    def num_params(self) -> int:
        return count_params(self)
