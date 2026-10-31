# %% [markdown]
# # BLT-on-Qwen3.5-0.8B-Base: Kaggle smoke test
#
# Pipeline check for `vislm/train_blt_qwen.py` on real hardware (plans/PLAN.md, BLT testbed):
# unit tests incl. the overfit test, a tiny entropy LM, then BLTQwen with the real Qwen3.5
# weights: whitespace patching in fp16 vs fp32, resume from checkpoint, entropy patching,
# and a memory probe of the kaggle_t4 / rtx24gb profiles. Not a result run - every model
# here is trained for minutes. Built with setup/build_kaggle_kernel.py.

# %%
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

W = "/kaggle/working"
print(sys.version)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

_candidates = ["/kaggle/input/vislm-research-code", "/kaggle/input/datasets/nguyennn263/vislm-research-code"]
_candidates += glob.glob("/kaggle/input/**/vislm-research-code", recursive=True)
CODE_DIR = next(p for p in _candidates if os.path.isdir(os.path.join(p, "vislm")))
os.environ["PYTHONPATH"] = CODE_DIR
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
CFG = f"{CODE_DIR}/experiments/blt_qwen/configs"
print("CODE_DIR:", CODE_DIR)

RESULTS = {}


def sh(name, cmd, env=None, timeout_min=30):
    """Run a command, stream its output into the notebook log, record rc + wall time.
    Killed after timeout_min (rc -9) so one stuck step cannot burn the whole GPU quota -
    v1 of this kernel sat 6h in the old `datasets` streaming download."""
    print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=W,
                         env={**os.environ, **(env or {})})
    timer = threading.Timer(timeout_min * 60, p.kill)
    timer.start()
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line.rstrip()])[-15:]
    rc = p.wait()
    timer.cancel()
    RESULTS[name] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
    if rc != 0:
        RESULTS[name]["tail"] = tail
    print(f"===== {name}: rc={rc} in {RESULTS[name]['seconds']}s", flush=True)
    return rc


def train(name, config, *overrides):
    return sh(name, [sys.executable, "-m", "vislm.train_blt_qwen", f"{CFG}/{config}", *overrides])

# %%
sh("pip_transformers", [sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE_DIR}/requirements-blt-qwen.txt"])
# Gated DeltaNet kernels (pure Triton). --no-deps so pip never swaps Kaggle's CUDA torch.
sh("pip_fla", [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core", "einops"])

# tests/ is imported by path: Kaggle's image has an unrelated `tests` package on sys.path.
FLA_CHECK = f"""
import sys; sys.path.insert(0, "{CODE_DIR}/tests")
import torch, test_blt_qwen as T
from vislm.backbones.modules.blt_patcher import whitespace_patch_lengths
m = T.tiny_model().cuda().train()
x = T.tokens_of().cuda()
with torch.autocast("cuda", dtype=torch.float16):
    logits, loss = m(x[:, :-1], whitespace_patch_lengths(x[:, :-1]), targets=x[:, 1:])
loss.backward()
assert torch.isfinite(loss), loss
print("tiny BLTQwen fwd/bwd on cuda ok, loss", loss.item())
"""
if sh("fla_check", [sys.executable, "-c", FLA_CHECK]) != 0:
    print("fla kernels fail on this GPU - removing them, transformers falls back to PyTorch")
    sh("pip_uninstall_fla", [sys.executable, "-m", "pip", "uninstall", "-y", "flash-linear-attention", "fla-core"])
    sh("fla_check_fallback", [sys.executable, "-c", FLA_CHECK])

VERSIONS = subprocess.run([sys.executable, "-c", (
    "import torch, transformers, importlib.util as u; "
    "print(torch.__version__, transformers.__version__, bool(u.find_spec('fla')))")],
    capture_output=True, text=True).stdout.split()
print("torch, transformers, fla:", VERSIONS)

# %%
DATA = f"{W}/data/fineweb2_vi"
sh("data", [sys.executable, f"{CODE_DIR}/setup/download_prepare_data.py", "--target-gb", "0.05", "--out-dir", DATA],
   timeout_min=15)

# %%
sh("pytest", [sys.executable, "-m", "pytest", f"{CODE_DIR}/tests/test_blt_qwen.py", "-q", "-p", "no:cacheprovider",
              "--rootdir", W], env={"VISLM_RUN_OVERFIT": "1"})

# %% [markdown]
# ## Entropy LM (tiny, 200 steps) - feeds the entropy-patching run below

# %%
ENT = f"{W}/runs/entropy_lm_smoke"
train("entropy_lm_smoke", "entropy_lm.yaml", "profile=kaggle_smoke", f"dataset={DATA}", f"run_dir={ENT}")
ENT_CKPT = f"{ENT}/entropy_lm.pt"
shutil.rmtree(f"{ENT}/checkpoints", ignore_errors=True)

# %% [markdown]
# ## BLTQwen + real Qwen3.5-0.8B-Base weights, whitespace patching
# Same seed and data: fp16 (auto on T4) vs fp32 should give close loss curves.
# The fp16 run saves at step 15/30 and is then resumed to step 40.

# %%
B = "blt_qwen35_08b.yaml"
WS16 = f"{W}/runs/blt_ws_fp16"
train("blt_ws_fp16", B, "profile=kaggle_smoke", f"dataset={DATA}", f"run_dir={WS16}")
train("blt_ws_fp16_resume", B, "profile=kaggle_smoke", f"dataset={DATA}", f"run_dir={WS16}",
      f"resume_from={WS16}/checkpoints/latest.pt", "train.max_steps=40", "train.save_final=false")
shutil.rmtree(f"{WS16}/checkpoints", ignore_errors=True)

train("blt_ws_fp32", B, "profile=kaggle_smoke", f"dataset={DATA}", f"run_dir={W}/runs/blt_ws_fp32",
      "train.precision=fp32", "train.save_every=0", "train.save_final=false")

# %% [markdown]
# ## Entropy patching (smoke entropy LM, threshold calibrated to 4.5 bytes/patch)

# %%
train("blt_entropy", B, "profile=kaggle_smoke", f"dataset={DATA}", f"run_dir={W}/runs/blt_entropy",
      "patching.mode=entropy", f"patching.entropy_ckpt={ENT_CKPT}", "train.save_every=0", "train.save_final=false")

# %% [markdown]
# ## Memory / speed probes of the real profiles (10 and 5 steps)
# kaggle_t4 = Qwen body frozen; rtx24gb = everything trained (expected not to fit a 16GB T4).

# %%
train("probe_kaggle_t4", B, "profile=kaggle_t4", f"dataset={DATA}", f"run_dir={W}/runs/probe_kaggle_t4",
      f"patching.entropy_ckpt={ENT_CKPT}", "train.max_steps=10", "train.grad_accum=1", "train.warmup_steps=2",
      "train.eval_batches=4", "train.save_every=0", "train.save_final=false", "train.log_every=1")
train("probe_full_seq2048", B, "profile=kaggle_t4", f"dataset={DATA}", f"run_dir={W}/runs/probe_full_seq2048",
      f"patching.entropy_ckpt={ENT_CKPT}", "train.freeze_global=false", "train.max_steps=5", "train.grad_accum=1",
      "train.warmup_steps=1", "train.eval_batches=2", "train.save_every=0", "train.save_final=false",
      "train.log_every=1")

# %%
def read_metrics(run_dir):
    path = f"{run_dir}/metrics.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    steps = [r for r in rows if "step" in r and "loss" in r]
    out = {k: v for r in rows if r.get("event") in ("start", "patching", "done", "final_eval") for k, v in r.items()
           if k not in ("event",)}
    if steps:
        out.update(first_bpb=round(steps[0]["bpb"], 3), last_bpb=round(steps[-1]["bpb"], 3), n_steps=len(steps),
                   peak_mem_gb=max(r.get("peak_mem_gb", 0) for r in steps),
                   mean_patch_bytes=steps[-1].get("mean_patch_bytes"))
    return out


summary = {"versions": VERSIONS, "commands": RESULTS,
           "runs": {os.path.basename(d): read_metrics(d) for d in sorted(glob.glob(f"{W}/runs/*"))}}
with open(f"{W}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
shutil.rmtree(f"{W}/data", ignore_errors=True)
