# %% [markdown]
# # BLT-on-Qwen: train the Vietnamese byte entropy model (Kaggle T4)
# `entropy_lm.yaml profile=kaggle_t4`: official BLT entropy-model shape (768 dim, 14 layers,
# window 512) on FineWeb-2 vie_Latn bytes. Stops itself at train.max_hours and saves
# runs/entropy_lm/entropy_lm.pt (used by blt_qwen35_08b.yaml patching.mode=entropy).
# Continue in a new session with RESUME pointing at a previous version's latest.pt.

# %%
import glob
import os
import subprocess
import sys

W = "/kaggle/working"
_candidates = ["/kaggle/input/vislm-research-code", "/kaggle/input/datasets/nguyennn263/vislm-research-code"]
_candidates += glob.glob("/kaggle/input/**/vislm-research-code", recursive=True)
CODE_DIR = next(p for p in _candidates if os.path.isdir(os.path.join(p, "vislm")))
env = {**os.environ, "PYTHONPATH": CODE_DIR, "PYTHONDONTWRITEBYTECODE": "1"}
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

DATA = f"{W}/data/fineweb2_vi"
RUN = f"{W}/runs/entropy_lm"
RESUME = None  # e.g. "/kaggle/input/<previous output>/runs/entropy_lm/checkpoints/latest.pt"


def run(cmd, timeout_h):
    print("=====", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, cwd=W, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait(timeout=timeout_h * 3600)
    print("===== rc", rc, flush=True)
    return rc


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE_DIR}/requirements-blt-qwen.txt"], check=True)
# 20000 steps x 32k bytes = 0.66GB seen; 1.2GB keeps it under one epoch, 1% held out for val
run([sys.executable, f"{CODE_DIR}/setup/download_prepare_data.py", "--target-gb", "1.2", "--out-dir", DATA], 0.5)

# %%
args = [sys.executable, "-m", "vislm.train_blt_qwen", f"{CODE_DIR}/experiments/blt_qwen/configs/entropy_lm.yaml",
        "profile=kaggle_t4", f"dataset={DATA}", f"run_dir={RUN}"]
if RESUME:
    args.append(f"resume_from={RESUME}")
run(args, 11.5)

# %%
import shutil
shutil.rmtree(f"{W}/data", ignore_errors=True)  # keep only runs/ in the output
print(open(f"{RUN}/metrics.jsonl").read()[-3000:])
