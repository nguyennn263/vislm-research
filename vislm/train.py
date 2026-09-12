"""Config-driven trainer (see vislm/args.py). Supports architecture: transformer_standard
(Arm A), blt_entropy_patching (Arm B), blt_entropy_patching_syllable_seeded (Arm C).
Logs to <run_dir>/metrics.jsonl per the runs/ convention in plans/PLAN.md.

    python -m vislm.train experiments/pillar1_patch_encoder/configs/1_3_arm_A_bpe.yaml
    python -m vislm.train ...same... train.max_steps=2000 dataset_target_gb=5

Resume an interrupted run (e.g. hit Kaggle's 12h session limit) with:
    python -m vislm.train <config> resume_from=<run_dir>/checkpoints/latest.pt
"""
import importlib
import json
import os
import time

import torch
import yaml

from vislm import checkpoint
from vislm.args import parse_args
from vislm.backbones.models.blt_lm import BLTLanguageModel
from vislm.backbones.models.transformer_lm import TransformerLM
from vislm.data import iter_texts
from vislm.lr_schedule import get_lr
from vislm.tokenizers import syllable_seed


def load_tokenizer(cfg: dict):
    module = importlib.import_module(cfg["tokenizer"])
    return module.load(cfg.get("tokenizer_name"))


def build_model(cfg: dict, vocab_size: int):
    arch = cfg["architecture"]
    m = cfg["model"]
    if arch == "transformer_standard":
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=m["d_model"],
            n_layers=m["n_layers"],
            n_heads=m["n_heads"],
            max_seq_len=m["max_seq_len"],
        )
    if arch in ("blt_entropy_patching", "blt_entropy_patching_syllable_seeded"):
        e = cfg.get("entropy_model", {})
        p = cfg.get("patching", {})
        return BLTLanguageModel(
            d_model=m["d_model"],
            n_layers=m["n_layers"],
            n_heads=m["n_heads"],
            max_seq_len=m["max_seq_len"],
            entropy_d_model=e.get("d_model", 128),
            entropy_n_layers=e.get("n_layers", 4),
            entropy_n_heads=e.get("n_heads", 4),
            entropy_threshold=p.get("entropy_threshold", 1.5),
            max_patch_len=p.get("max_patch_len", 16),
            seed_boundaries_fn=(
                syllable_seed.seed_boundaries
                if arch == "blt_entropy_patching_syllable_seeded"
                else None
            ),
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


def split_train_val(token_ids: torch.Tensor, val_fraction: float):
    """Held-out val split is the LAST val_fraction of the stream - simple and
    deterministic, no shuffling infra needed at this data scale."""
    if not val_fraction:
        return token_ids, None
    n_val = int(len(token_ids) * val_fraction)
    return token_ids[:-n_val], token_ids[-n_val:]


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


def pretrain_entropy_model(model: BLTLanguageModel, token_ids, cfg, device, mf):
    steps = cfg["train"].get("entropy_pretrain_steps", 50)
    print(f"pretraining entropy model for {steps} steps...")
    opt = torch.optim.AdamW(model.entropy_model.parameters(), lr=cfg["train"]["lr"])
    loss = None
    for step, (xs, ys) in enumerate(
        batches(token_ids, cfg["model"]["max_seq_len"], cfg["train"]["batch_size"], device)
    ):
        _, loss = model.entropy_model(xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mf.write(json.dumps({"entropy_pretrain_step": step, "loss": loss.item()}) + "\n")
        if step >= steps:
            break
    for p in model.entropy_model.parameters():
        p.requires_grad = False
    print(f"entropy model frozen, final pretrain loss {loss.item():.4f}")


@torch.no_grad()
def evaluate(model, val_ids, cfg, device, eval_steps):
    model.eval()
    losses = []
    for xs, ys in batches(val_ids, cfg["model"]["max_seq_len"], cfg["train"]["batch_size"], device):
        _, loss = model(xs, ys)
        losses.append(loss.item())
        if len(losses) >= eval_steps:
            break
    model.train()
    return sum(losses) / len(losses) if losses else None


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.get("seed", 42))
    device = pick_device()
    tcfg = cfg["train"]

    tok = load_tokenizer(cfg)
    all_ids = []
    for text in iter_texts(cfg["dataset"]):
        all_ids.extend(tok.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_ids, dtype=torch.long)
    train_ids, val_ids = split_train_val(token_ids, tcfg.get("val_fraction", 0.0))
    print(
        f"loaded {len(token_ids)} tokens from {cfg['dataset']} "
        f"({len(train_ids)} train, {len(val_ids) if val_ids is not None else 0} val)"
    )

    model = build_model(cfg, vocab_size=tok.vocab_size).to(device)
    n_params = model.num_params()
    n_latent_params = getattr(model, "num_latent_params", model.num_params)()
    print(
        f"device={device} architecture={cfg['architecture']} "
        f"params={n_params:,} latent_params={n_latent_params:,}"
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"]
    )

    run_dir = cfg.get("run_dir", "runs/debug_run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    max_steps = tcfg["max_steps"]
    start_step = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        state = checkpoint.load(resume_from, model, opt)
        start_step = state["step"]
        print(f"resumed from {resume_from} at step {start_step}")
        # NOTE: data iteration restarts from the beginning of train_ids (deterministic,
        # no shuffling), not from the exact batch the checkpoint stopped at - simplest
        # correct thing at this data scale, though it means the first batches after
        # resume repeat data already seen before the interruption. Step counter and LR
        # schedule DO continue correctly from where they left off.

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    with open(metrics_path, "a" if resume_from else "w") as mf:
        mf.write(
            json.dumps(
                {"event": "start", "params": n_params, "latent_params": n_latent_params, "device": device}
            )
            + "\n"
        )

        if isinstance(model, BLTLanguageModel) and not resume_from:
            pretrain_entropy_model(model, train_ids, cfg, device, mf)

        min_lr = tcfg.get("min_lr", tcfg["lr"])
        warmup_steps = tcfg.get("warmup_steps", 0)
        grad_clip = tcfg.get("grad_clip")
        save_every = tcfg.get("save_every")
        eval_every = tcfg.get("eval_every")
        eval_steps = tcfg.get("eval_steps", 20)

        step = start_step
        t0 = time.time()
        while step < max_steps:
            for xs, ys in batches(train_ids, cfg["model"]["max_seq_len"], tcfg["batch_size"], device):
                lr = get_lr(step, tcfg["lr"], min_lr, warmup_steps, max_steps)
                for g in opt.param_groups:
                    g["lr"] = lr

                _, loss = model(xs, ys)
                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                opt.step()

                mf.write(
                    json.dumps(
                        {"step": step, "loss": loss.item(), "lr": lr, "elapsed_s": time.time() - t0}
                    )
                    + "\n"
                )
                mf.flush()
                if step % 10 == 0:
                    print(f"step {step} loss {loss.item():.4f} lr {lr:.2e}")

                if val_ids is not None and eval_every and step > 0 and step % eval_every == 0:
                    val_loss = evaluate(model, val_ids, cfg, device, eval_steps)
                    mf.write(json.dumps({"eval_step": step, "val_loss": val_loss}) + "\n")
                    print(f"step {step} val_loss {val_loss:.4f}")

                if save_every and step > 0 and step % save_every == 0:
                    checkpoint.save(run_dir, step, model, opt)

                step += 1
                if step >= max_steps:
                    break

    checkpoint.save(run_dir, step, model, opt)
    print(f"done: {step} steps, final loss {loss.item():.4f}, params {n_params:,}")


if __name__ == "__main__":
    main()
