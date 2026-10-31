# %% [markdown]
# # BLT-on-Qwen: unit tests on a Kaggle T4 with the flash-linear-attention kernels
# tests/test_blt_qwen.py on CUDA (causality, lag fix, padding, overfit), i.e. with the real
# Gated DeltaNet Triton kernels instead of the PyTorch fallback the Mac runs.

# %%
import glob
import os
import subprocess
import sys

_candidates = ["/kaggle/input/vislm-research-code", "/kaggle/input/datasets/nguyennn263/vislm-research-code"]
_candidates += glob.glob("/kaggle/input/**/vislm-research-code", recursive=True)
CODE_DIR = next(p for p in _candidates if os.path.isdir(os.path.join(p, "vislm")))
env = {**os.environ, "PYTHONPATH": CODE_DIR, "PYTHONDONTWRITEBYTECODE": "1", "VISLM_RUN_OVERFIT": "1"}
for cmd in (
    [sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE_DIR}/requirements-blt-qwen.txt"],
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core", "einops"],
):
    subprocess.run(cmd, check=True)
r = subprocess.run([sys.executable, "-m", "pytest", f"{CODE_DIR}/tests/test_blt_qwen.py", "-v", "-p", "no:cacheprovider",
                    "--rootdir", "/kaggle/working"], env=env, capture_output=True, text=True, timeout=3600)
print(r.stdout[-8000:], r.stderr[-3000:])
open("/kaggle/working/pytest.txt", "w").write(r.stdout)
