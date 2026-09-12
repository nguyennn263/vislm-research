"""Config-driven trainer (see vislm/args.py). Currently supports architecture:
transformer_standard (Arm A). Logs to <run_dir>/metrics.jsonl per the runs/ convention
in plans/PLAN.md.

    python -m vislm.train experiments/pillar1_patch_encoder/configs/1_3_arm_A_bpe.yaml
    python -m vislm.train ...same... train.max_steps=2000 dataset_target_gb=5
"""
import json
import os
import time

import torch
import yaml

from vislm.args import parse_args
from vislm.backbones.models.transformer_lm import TransformerLM
from vislm.data import iter_texts
from vislm.tokenizers import bpe_baseline


def build_model(cfg: dict, vocab_size: int):
    arch = cfg["architecture"]
    if arch == "transformer_standard":
        m = cfg["model"]
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=m["d_model"],
            n_layers=m["n_layers"],
            n_heads=m["n_heads"],
            max_seq_len=m["max_seq_len"],
        )
    raise NotImplementedError(f"architecture '{arch}' not implemented yet")


def batches(token_ids: torch.Tensor, seq_len: int, batch_size: int, device: str):
    n_chunks = (len(token_ids) - 1) // seq_len
    token_ids = token_ids[: n_chunks * seq_len + 1]
    for start in range(0, n_chunks, batch_size):
        idx = range(start, min(start + batch_size, n_chunks))
        xs = torch.stack([token_ids[i * seq_len : (i + 1) * seq_len] for i in idx])
        ys = torch.stack([token_ids[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in idx])
        yield xs.to(device), ys.to(device)


def pick_device() -> str:
    """cuda can report available but still fail at the first real op (e.g. a GPU
    whose compute capability the installed torch build doesn't support - seen with
    Kaggle's P100 + recent torch). Probe with a tiny op before committing to it."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")).cpu()
        return "cuda"
    except RuntimeError as e:
        print(f"cuda reported available but failed a smoke op ({e}); falling back to cpu")
        return "cpu"


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.get("seed", 42))
    device = pick_device()

    tok = bpe_baseline.load(cfg["tokenizer_name"])
    all_ids = []
    for text in iter_texts(cfg["dataset"]):
        all_ids.extend(tok.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_ids, dtype=torch.long)
    print(f"loaded {len(token_ids)} tokens from {cfg['dataset']}")

    model = build_model(cfg, vocab_size=tok.vocab_size).to(device)
    n_params = model.num_params()
    print(f"device={device} architecture={cfg['architecture']} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    run_dir = cfg.get("run_dir", "runs/debug_run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    max_steps = cfg["train"]["max_steps"]
    step = 0
    t0 = time.time()
    with open(os.path.join(run_dir, "metrics.jsonl"), "w") as mf:
        mf.write(json.dumps({"event": "start", "params": n_params, "device": device}) + "\n")
        while step < max_steps:
            for xs, ys in batches(
                token_ids, cfg["model"]["max_seq_len"], cfg["train"]["batch_size"], device
            ):
                _, loss = model(xs, ys)
                opt.zero_grad()
                loss.backward()
                opt.step()
                mf.write(
                    json.dumps(
                        {"step": step, "loss": loss.item(), "elapsed_s": time.time() - t0}
                    )
                    + "\n"
                )
                mf.flush()
                if step % 10 == 0:
                    print(f"step {step} loss {loss.item():.4f}")
                step += 1
                if step >= max_steps:
                    break

    print(f"done: {step} steps, final loss {loss.item():.4f}, params {n_params:,}")


if __name__ == "__main__":
    main()
