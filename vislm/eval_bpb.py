"""Document-level bits-per-byte, the one number comparable across tokenizers.

The same held-out documents (complete docs from the val tail of the byte stream, each cut
to --max-bytes at a UTF-8 boundary) are scored by
  - a BPE LM (Qwen3.5-0.8B-Base as is): [<|endoftext|>] + tokens(text), every text token predicted;
  - BLTQwen: [BOS] + bytes, every byte predicted (vislm/train_blt_qwen.py calls blt_doc_bpb).
bpb = total NLL in nats over the docs / total UTF-8 bytes / ln 2.

    python -m vislm.eval_bpb --dataset data/prepared/fw2_2gb --out runs/qwen_baseline.json
"""
import argparse
import json
import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from vislm.backbones.modules.blt_layers import BOS_ID, EOS_ID, OFFSET
from vislm.byte_data import load_byte_stream, split_stream


def val_documents(stream: np.ndarray, val_fraction: float, n_docs: int, max_bytes: int) -> List[bytes]:
    """First n_docs complete documents (BOS ... EOS) of the val tail, truncated to max_bytes
    without splitting a UTF-8 character."""
    _, val = split_stream(stream, val_fraction)
    val = np.asarray(val)
    bos = np.flatnonzero(val == BOS_ID)
    docs = []
    for start in bos:
        end = start + 1
        while end < len(val) and val[end] != EOS_ID:
            end += 1
        if end >= len(val):
            break
        b = bytes((val[start + 1 : end] - OFFSET).astype(np.uint8))
        b = b[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")
        if b:
            docs.append(b)
        if len(docs) == n_docs:
            break
    return docs


def summarize(nats: float, n_bytes: int, n_docs: int, **extra) -> dict:
    return {"bpb": nats / n_bytes / math.log(2), "nats": nats, "bytes": n_bytes, "docs": n_docs, **extra}


@torch.no_grad()
def bpe_doc_bpb(model_name: str, docs: List[bytes], device: str, dtype=torch.bfloat16) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    prefix = tok.convert_tokens_to_ids("<|endoftext|>")
    nats, n_bytes, n_tokens = 0.0, 0, 0
    for b in docs:
        ids = [prefix] + tok(b.decode("utf-8"), add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device=device)
        logits = model(input_ids=x).logits.float()
        nats += F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
        n_bytes += len(b)
        n_tokens += len(ids) - 1
    return summarize(nats, n_bytes, len(docs), model=model_name, tokens=n_tokens,
                     bytes_per_token=n_bytes / n_tokens)


@torch.no_grad()
def blt_doc_bpb(model, patch_fn, docs: List[bytes], device: str, autocast_ctx) -> dict:
    was_training = model.training
    model.eval()
    nats, n_bytes, n_patches = 0.0, 0, 0
    for b in docs:
        ids = [BOS_ID] + [c + OFFSET for c in b]
        t = torch.tensor([ids], device=device)
        x, y = t[:, :-1], t[:, 1:]
        with autocast_ctx():
            lengths = patch_fn(x)
            logits, _ = model(x, lengths)
        nats += F.cross_entropy(logits[0].float(), y[0], reduction="sum").item()
        n_bytes += len(b)
        n_patches += int((lengths > 0).sum())
    model.train(was_training)
    return summarize(nats, n_bytes, len(docs), bytes_per_patch=n_bytes / n_patches)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--n-docs", type=int, default=500)
    ap.add_argument("--max-bytes", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = val_documents(load_byte_stream(args.dataset), args.val_fraction, args.n_docs, args.max_bytes)
    res = bpe_doc_bpb(args.model, docs, device)
    res.update(dataset=args.dataset, val_fraction=args.val_fraction, max_bytes=args.max_bytes)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
