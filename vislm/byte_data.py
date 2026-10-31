"""Byte stream for the BLT-on-Qwen testbed: prepared shards (vislm/data.py) -> one flat
uint16 array of BLT byte ids (each document = BOS + utf-8 bytes + 4 + EOS), cached as a
memmap next to the shards so a 5GB run on the RTX box does not need 10GB of RAM.

Batches are random windows drawn from a generator seeded by (seed, step): resuming at
step s replays exactly the batches a non-interrupted run would have seen from s on.
"""
import os
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from vislm.backbones.modules.blt_layers import BOS_ID, EOS_ID, OFFSET
from vislm.data import iter_texts, shard_paths


def _cache_path(data_dir: str, cache_dir: Optional[str]) -> str:
    stamp = sum(os.path.getsize(p) for p in shard_paths(data_dir))
    return os.path.join(cache_dir or data_dir, f"blt_bytes_{stamp}.u16")


def load_byte_stream(data_dir: str, cache_dir: Optional[str] = None) -> np.ndarray:
    path = _cache_path(data_dir, cache_dir)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            for text in iter_texts(data_dir):
                b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + OFFSET
                np.concatenate([[BOS_ID], b, [EOS_ID]]).astype(np.uint16).tofile(f)
        os.replace(tmp, path)
    return np.memmap(path, dtype=np.uint16, mode="r")


def split_stream(stream: np.ndarray, val_fraction: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Held-out val = the last val_fraction of the stream (deterministic, as vislm/train.py)."""
    if not val_fraction:
        return stream, None
    cut = int(len(stream) * (1 - val_fraction))
    return stream[:cut], stream[cut:]


def _to_batch(windows, device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(np.stack(windows).astype(np.int64))
    return t[:, :-1].to(device, non_blocking=True), t[:, 1:].to(device, non_blocking=True)


def train_batch(stream: np.ndarray, seq_len: int, batch_size: int, seed: int, step: int, device):
    rng = np.random.default_rng([seed, step])
    starts = rng.integers(0, len(stream) - seq_len - 1, size=batch_size)
    return _to_batch([stream[s : s + seq_len + 1] for s in starts], device)


def val_batches(stream: np.ndarray, seq_len: int, batch_size: int, max_batches: int, device) -> Iterator:
    n_windows = (len(stream) - 1) // seq_len
    for b in range(min(max_batches, n_windows // batch_size)):
        idx = range(b * batch_size, (b + 1) * batch_size)
        yield _to_batch([stream[i * seq_len : i * seq_len + seq_len + 1] for i in idx], device)
