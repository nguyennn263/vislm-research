"""Patch boundaries for the BLT-on-Qwen testbed, matching facebookresearch/blt @ 9774ed4
bytelatent/data/patcher.py (find_entropy_patch_start_ids, find_space_patch_start_ids,
patch_lengths_from_start_ids, split_large_numbers). Own implementation, same outputs.

Conventions kept from the official code (they are what makes the lag fix work):
- position 0 and position 1 always start a patch, so the first patch is exactly 1 byte;
- entropies[i] is the entropy of the prediction for byte i+1, so "entropies[j] > t" makes
  byte j+1 start a patch (only entropies[1:] are used);
- include_next_token=True adds one slot for the (not yet seen) next byte, so the lengths
  of a length-N row sum to N+1. BLTQwen.forward expects exactly this.
All rules here are incremental: whether byte i starts a patch depends on bytes < i only.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from vislm.backbones.modules.blt_layers import OFFSET


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Natural-log entropy of softmax(logits) over the last dim (patcher.entropy)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def entropy_start_mask(
    entropies: torch.Tensor,
    threshold: float,
    threshold_add: Optional[float] = None,
    monotonicity: bool = False,
) -> torch.Tensor:
    """Rule on entropies[:, 1:] -> [bs, N-1] bool; element j means "byte j+2 starts a patch".
    Global (default), monotonic (e_t - e_{t-1} > threshold) or both (threshold_add)."""
    e = entropies[:, 1:]
    if e.shape[1] == 0:
        return e > threshold
    if monotonicity:
        mask = torch.zeros_like(e, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 1:] = (e[:, 1:] - e[:, :-1]) > threshold
        return mask
    if threshold_add is not None:
        mask = torch.zeros_like(e, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 1:] = ((e[:, 1:] - e[:, :-1]) > threshold_add) & (e[:, 1:] > threshold) & (~mask[:, :-1])
        return mask
    return e > threshold


def lengths_from_start_mask(start_mask: torch.Tensor, max_patch_length: Optional[int] = None) -> torch.Tensor:
    """[bs, L] bool (True = a patch starts here; column 0 must be True) -> [bs, P] lengths,
    right-padded with 0, each row summing to L. Long patches are split greedily into
    max_patch_length pieces (split_large_numbers)."""
    bs, L = start_mask.shape
    rows: List[List[int]] = []
    for r in range(bs):
        starts = torch.nonzero(start_mask[r], as_tuple=False).flatten().tolist()
        assert starts and starts[0] == 0, "position 0 must start a patch"
        ends = starts[1:] + [L]
        lengths = []
        for s, e in zip(starts, ends):
            n = e - s
            if max_patch_length is not None:
                while n > max_patch_length:
                    lengths.append(max_patch_length)
                    n -= max_patch_length
            lengths.append(n)
        rows.append(lengths)
    width = max(len(x) for x in rows)
    out = torch.zeros(bs, width, dtype=torch.long, device=start_mask.device)
    for r, lengths in enumerate(rows):
        out[r, : len(lengths)] = torch.tensor(lengths, dtype=torch.long)
    return out


def _with_first_two(rest: torch.Tensor, length: int) -> torch.Tensor:
    """Prepend the always-on starts at positions 0 and 1, truncate to `length` columns."""
    head = torch.ones(rest.shape[0], 2, dtype=torch.bool, device=rest.device)
    return torch.cat([head, rest], dim=1)[:, :length]


def entropy_patch_lengths(
    entropies: torch.Tensor,
    threshold: float,
    threshold_add: Optional[float] = None,
    monotonicity: bool = False,
    include_next_token: bool = True,
    max_patch_length: Optional[int] = None,
) -> torch.Tensor:
    """Patcher.patch(mode="entropy", threshold=...). entropies: [bs, N] from the entropy model
    run on the same N tokens. Returns [bs, P] lengths summing to N (+1 if include_next_token).
    (The no-threshold top-k variant is global, not incremental, and is not implemented.)"""
    n = entropies.shape[1]
    rest = entropy_start_mask(entropies, threshold, threshold_add, monotonicity)
    length = n + 1 if include_next_token else n
    return lengths_from_start_mask(_with_first_two(rest, length), max_patch_length)


def whitespace_patch_lengths(
    tokens: torch.Tensor,
    include_next_token: bool = True,
    max_patch_length: Optional[int] = None,
) -> torch.Tensor:
    """Patcher.patch(mode="space") with Vietnamese-correct byte classes.

    Official rule: a byte is "space-like" if it is not [0-9A-Za-z], counting every UTF-8 lead
    byte >= 0xC0 as space-like; runs of space-like bytes collapse to their first byte; special
    ids (< OFFSET) always end a patch; the byte after an end starts a patch. The >= 0xC0 rule
    cuts inside accented letters ("Nội" = N, 0xE1 0xBB 0x99, i), so here every non-ASCII byte
    counts as part of a word. That is the only change.
    """
    b = tokens - OFFSET
    ascii_alnum = (
        ((b >= ord("0")) & (b <= ord("9")))
        | ((b >= ord("A")) & (b <= ord("Z")))
        | ((b >= ord("a")) & (b <= ord("z")))
    )
    end = (b < 0x80) & ~ascii_alnum
    end[:, 1:] &= ~end[:, :-1]
    end |= tokens < OFFSET
    n = tokens.shape[1]
    length = n + 1 if include_next_token else n
    return lengths_from_start_mask(_with_first_two(end[:, 1:], length), max_patch_length)


class EntropyPatcher:
    """Frozen ByteEntropyLM + threshold rule. Call on the same tokens the BLT model sees."""

    def __init__(self, entropy_model, threshold: float, threshold_add: Optional[float] = None,
                 monotonicity: bool = False, max_patch_length: Optional[int] = None):
        self.entropy_model = entropy_model.eval()
        for p in self.entropy_model.parameters():
            p.requires_grad_(False)
        self.threshold = threshold
        self.threshold_add = threshold_add
        self.monotonicity = monotonicity
        self.max_patch_length = max_patch_length

    @torch.no_grad()
    def entropies(self, tokens: torch.Tensor) -> torch.Tensor:
        # DEVIATION: official calculate_entropies flattens the batch and re-chunks it into
        # max_length windows; we score each row on its own (same result when rows <= 8192).
        return entropy_from_logits(self.entropy_model(tokens))

    def __call__(self, tokens: torch.Tensor, include_next_token: bool = True) -> torch.Tensor:
        return entropy_patch_lengths(
            self.entropies(tokens), self.threshold, self.threshold_add, self.monotonicity,
            include_next_token, self.max_patch_length,
        )
