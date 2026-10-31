"""Trainer for the BLT-on-Qwen testbed. One script, one config, any machine:

    python -m vislm.train_blt_qwen <config.yaml> profile=<name> [key=value ...]

`profile` merges `profiles.<name>` from the YAML over the base config (e.g. kaggle_smoke,
kaggle_t4, rtx24gb); dot-list overrides on the command line win over both. Precision is
picked from the hardware unless train.precision is set: bf16 on Ampere+, fp16 + GradScaler
on older GPUs (Kaggle T4), fp32 on CPU.

Tasks (config `task`):
  entropy_lm  train the byte entropy model used for entropy patching (ByteEntropyLM)
  blt_qwen    train BLTQwen (local encoder/decoder around Qwen3.5-0.8B-Base)

train.max_hours stops cleanly (final save + eval) before a session limit; continue with
resume_from=<run_dir>/checkpoints/latest.pt.
Writes <run_dir>/config.yaml and <run_dir>/metrics.jsonl (runs/ convention, plans/PLAN.md).
"""
import copy
import json
import math
import os
import sys
import time

import torch
import yaml

from vislm.args import _cast, _set_nested
from vislm.backbones.models.blt_qwen import BLTQwen, BLTQwenConfig, QwenLatentTransformer
from vislm.backbones.modules.blt_layers import PAD_ID, ByteEntropyLM
from vislm.backbones.modules.blt_patcher import EntropyPatcher, whitespace_patch_lengths
from vislm.byte_data import load_byte_stream, split_stream, train_batch, val_batches
from vislm.eval_bpb import blt_doc_bpb, val_documents
from vislm.lr_schedule import get_lr

LN2 = math.log(2)


# --- config ---------------------------------------------------------------------------
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(argv=None) -> dict:
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        cfg = yaml.safe_load(f)
    overrides = dict(a.partition("=")[::2] for a in argv[1:])
    profile = overrides.pop("profile", cfg.get("profile"))
    if profile:
        cfg = _deep_merge(cfg, cfg.get("profiles", {})[profile])
    cfg.pop("profiles", None)
    cfg["profile"] = profile
    for k, v in overrides.items():
        _set_nested(cfg, k, _cast(v))
    return cfg


# --- hardware -------------------------------------------------------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        try:
            (torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")).cpu()
            return "cuda"
        except RuntimeError as e:
            print(f"cuda failed a smoke op ({e}); using cpu")
    return "cpu"


def pick_precision(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    if device != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"


def autocast(precision: str, device: str):
    if precision == "fp32":
        return torch.autocast(device_type=device, enabled=False)
    return torch.autocast(device_type=device, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def device_info(device: str) -> dict:
    if device != "cuda":
        return {"device": device}
    p = torch.cuda.get_device_properties(0)
    return {"device": device, "gpu": p.name, "gpu_mem_gb": round(p.total_memory / 1e9, 1),
            "n_gpus": torch.cuda.device_count(), "torch": torch.__version__}


# --- models ---------------------------------------------------------------------------
def build_entropy_lm(cfg: dict) -> ByteEntropyLM:
    return ByteEntropyLM(**cfg.get("entropy_model", {}))


def load_entropy_lm(path: str, device: str) -> ByteEntropyLM:
    state = torch.load(path, map_location="cpu")
    model = ByteEntropyLM(**state["entropy_model_cfg"])
    model.load_state_dict(state["model"])
    return model.to(device)


def build_blt_qwen(cfg: dict) -> BLTQwen:
    m = dict(cfg.get("model", {}))
    if "hash_ngram_sizes" in m:
        m["hash_ngram_sizes"] = tuple(m["hash_ngram_sizes"])
    bcfg = BLTQwenConfig(**m)
    debug_qwen = cfg.get("debug_qwen_text_config")
    if debug_qwen:  # random tiny Qwen3.5 body, for plumbing checks without the 1.7GB download
        from transformers import Qwen3_5TextConfig

        return BLTQwen(bcfg, QwenLatentTransformer.from_config(Qwen3_5TextConfig(**debug_qwen)))
    # fp32 master weights; autocast does the low-precision math
    return BLTQwen.from_qwen_pretrained(bcfg, dtype=torch.float32)


def make_patcher(cfg: dict, device: str, val, seq_len: int, precision: str, mf):
    """Returns tokens -> patch_lengths. For entropy mode, calibrates the threshold on val
    bytes to hit patching.target_mean_patch_bytes unless patching.threshold is given."""
    p = cfg.get("patching", {})
    max_len = p.get("max_patch_length")
    if p.get("mode", "whitespace") == "whitespace":
        return lambda x: whitespace_patch_lengths(x, max_patch_length=max_len)

    patcher = EntropyPatcher(load_entropy_lm(p["entropy_ckpt"], device), threshold=0.0,
                             monotonicity=p.get("monotonicity", False), max_patch_length=max_len)
    threshold = p.get("threshold")
    if threshold is None:
        ents = []
        with autocast(precision, device):
            for x, _ in val_batches(val, seq_len, 4, 4, device):
                ents.append(patcher.entropies(x)[:, 1:].flatten().float())
        e = torch.cat(ents)
        threshold = torch.quantile(e[: 2**24], 1 - 1 / p["target_mean_patch_bytes"]).item()
    patcher.threshold = threshold
    mf.write(json.dumps({"event": "patching", "mode": "entropy", "threshold": threshold}) + "\n")
    print(f"entropy patching threshold = {threshold:.4f}")

    def patch(x):
        with autocast(precision, device):
            return patcher(x)
    return patch


def param_groups(model, cfg: dict):
    """Two LRs: the new local modules learn fast, the pretrained Qwen body slowly (or not
    at all with train.freeze_global). Weight decay only on matrices, never on embeddings/norms."""
    t = cfg["train"]
    wd = t.get("weight_decay", 0.1)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_global = name.startswith("global_model.")
        decay = p.ndim >= 2 and "embeddings" not in name and "tok_embeddings" not in name
        key = ("global" if is_global else "local", decay)
        groups.setdefault(key, []).append(p)
    out = []
    for (part, decay), params in groups.items():
        lr = t.get("lr_global", t["lr"]) if part == "global" else t["lr"]
        out.append({"params": params, "weight_decay": wd if decay else 0.0, "max_lr": lr, "part": part})
    return out


def trainable_state(model) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers()}
    return {k: v for k, v in model.state_dict().items() if k in keep}


def save_ckpt(run_dir, step, model, opt, scaler, extra):
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    state = {"step": step, "model": trainable_state(model), "optimizer": opt.state_dict(),
             "scaler": scaler.state_dict() if scaler else None, **extra}
    path = os.path.join(run_dir, "checkpoints", "latest.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


# --- train ----------------------------------------------------------------------------
def main(argv=None):
    cfg = load_config(argv)
    t = cfg["train"]
    task = cfg["task"]
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device = pick_device()
    precision = pick_precision(t.get("precision", "auto"), device)
    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    stream = load_byte_stream(cfg["dataset"], cfg.get("cache_dir"))
    train, val = split_stream(stream, t.get("val_fraction", 0.01))
    seq_len, bs = t["seq_len"], t["batch_size"]
    accum = t.get("grad_accum", 1)
    print(f"bytes: {len(train):,} train / {len(val) if val is not None else 0:,} val; "
          f"device={device} precision={precision} profile={cfg.get('profile')}")

    resume = cfg.get("resume_from")
    mf = open(os.path.join(run_dir, "metrics.jsonl"), "a" if resume else "w")

    if task == "entropy_lm":
        model = build_entropy_lm(cfg).to(device)
        patch_fn = None
    elif task == "blt_qwen":
        model = build_blt_qwen(cfg).to(device)
        if t.get("freeze_global"):
            model.global_model.requires_grad_(False)
        if t.get("grad_checkpointing"):
            model.global_model.model.gradient_checkpointing_enable()
        patch_fn = make_patcher(cfg, device, val, seq_len, precision, mf)
    else:
        raise ValueError(f"unknown task {task}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = param_groups(model, cfg) if task == "blt_qwen" else [
        {"params": [p for p in model.parameters()], "weight_decay": t.get("weight_decay", 0.1),
         "max_lr": t["lr"], "part": "all"}]
    opt = torch.optim.AdamW(groups, lr=t["lr"], betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None

    start = 0
    if resume:
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        opt.load_state_dict(state["optimizer"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start = state["step"]
        print(f"resumed from {resume} at step {start}")

    mf.write(json.dumps({"event": "start", "task": task, "params": n_params, "trainable_params": n_train,
                         "precision": precision, "profile": cfg.get("profile"), **device_info(device)}) + "\n")
    print(f"params {n_params:,} (trainable {n_train:,})")

    def step_loss(x, y):
        with autocast(precision, device):
            if task == "entropy_lm":
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=PAD_ID)
                return loss, None
            lengths = patch_fn(x)
            _, loss = model(x, lengths, targets=y)
            return loss, lengths

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for x, y in val_batches(val, seq_len, bs, t.get("eval_batches", 20), device):
            loss, _ = step_loss(x, y)
            tot, n = tot + loss.item(), n + 1
        model.train()
        return tot / max(n, 1)

    # document-level bpb on the same docs vislm/eval_bpb.py scores Qwen-BPE on (comparable)
    d = cfg.get("doc_eval", {})
    docs = (val_documents(stream, t.get("val_fraction", 0.01), d["n_docs"], d.get("max_bytes", 4096))
            if task == "blt_qwen" and d.get("n_docs") else None)

    def doc_eval(step, final=False):
        if not docs:
            return
        res = blt_doc_bpb(model, patch_fn, docs, device, lambda: autocast(precision, device))
        mf.write(json.dumps({"event": "final_doc_eval" if final else "doc_eval", "step": step, **res}) + "\n")
        mf.flush()
        print(f"step {step} doc_bpb {res['bpb']:.4f} ({res['docs']} docs, {res['bytes_per_patch']:.2f} B/patch)")

    max_steps = t["max_steps"]
    max_lr = t["lr"]
    min_ratio = t.get("min_lr_ratio", 0.1)
    model.train()
    cuda = device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = start
    max_seconds = t.get("max_hours", 0) * 3600 or float("inf")
    end_step = max_steps
    for step in range(start, max_steps):
        if time.time() - t0 > max_seconds:  # stop cleanly before e.g. Kaggle's 12h kill
            end_step = step
            mf.write(json.dumps({"event": "time_limit", "step": step, "max_hours": t["max_hours"]}) + "\n")
            print(f"max_hours={t['max_hours']} reached at step {step}; saving and stopping")
            break
        scale = get_lr(step, 1.0, min_ratio, t.get("warmup_steps", 0), max_steps)
        for g in opt.param_groups:
            g["lr"] = g["max_lr"] * scale
        tot_loss, mean_patch = 0.0, None
        for micro in range(accum):
            x, y = train_batch(train, seq_len, bs, seed, step * accum + micro, device)
            loss, lengths = step_loss(x, y)
            if not torch.isfinite(loss):
                mf.write(json.dumps({"event": "nonfinite_loss", "step": step}) + "\n")
                mf.close()
                raise FloatingPointError(f"non-finite loss at step {step} (precision={precision})")
            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            tot_loss += loss.item() / accum
            if lengths is not None:
                mean_patch = (x.shape[1] + 1) / (lengths > 0).sum(-1).float().mean().item()
        if scaler:
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], t.get("grad_clip", 1.0))
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        row = {"step": step, "loss": tot_loss, "bpb": tot_loss / LN2, "lr": max_lr * scale,
               "grad_norm": gnorm.item(), "elapsed_s": round(time.time() - t0, 2)}
        if mean_patch is not None:
            row["mean_patch_bytes"] = round(mean_patch, 3)
        if cuda:
            row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        mf.write(json.dumps(row) + "\n")
        mf.flush()
        if step % t.get("log_every", 10) == 0:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if val is not None and t.get("eval_every") and (step + 1) % t["eval_every"] == 0:
            vl = evaluate()
            mf.write(json.dumps({"eval_step": step, "val_loss": vl, "val_bpb": vl / LN2}) + "\n")
            print(f"step {step} val_bpb {vl / LN2:.4f}")
            doc_eval(step)
        if t.get("save_every") and (step + 1) % t["save_every"] == 0:
            save_ckpt(run_dir, step + 1, model, opt, scaler, {})

    extra = {"entropy_model_cfg": cfg.get("entropy_model", {})} if task == "entropy_lm" else {}
    # save_final=false for throwaway runs: a BLTQwen checkpoint with an unfrozen Qwen body
    # (weights + Adam state, fp32) is ~6GB, a third of Kaggle's /kaggle/working quota.
    path = save_ckpt(run_dir, end_step, model, opt, scaler, extra) if t.get("save_final", True) else None
    if task == "entropy_lm":  # small standalone file for patching
        torch.save({"model": model.state_dict(), **extra}, os.path.join(run_dir, "entropy_lm.pt"))
    if val is not None:
        vl = evaluate()
        mf.write(json.dumps({"event": "final_eval", "step": end_step, "val_loss": vl, "val_bpb": vl / LN2}) + "\n")
        print(f"final val_bpb {vl / LN2:.4f}")
        doc_eval(end_step, final=True)
    mf.write(json.dumps({"event": "done", "steps": end_step, "elapsed_s": round(time.time() - t0, 1),
                         "s_per_step": round((time.time() - t0) / max(end_step - start, 1), 3)}) + "\n")
    mf.close()
    print(f"done, checkpoint {path}")


if __name__ == "__main__":
    main()
