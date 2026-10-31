# BLT trên Qwen3.5-0.8B-Base

Testbed duy nhất để kiểm chứng BLT từ 2026-10-18 (xem memory `blt-qwen-testbed`). Model:
`vislm/backbones/models/blt_qwen.py`; trainer: `vislm/train_blt_qwen.py`; test:
`tests/test_blt_qwen.py`.

Một script, một config, chạy được ở mọi máy. Máy nào thì chọn `profile=`; tham số lẻ thì
override bằng `key=value`. Precision tự chọn theo GPU: bf16 (Ampere trở lên), fp16 kèm
GradScaler (T4), fp32 (CPU).

| profile | dùng cho | ghi chú |
|---|---|---|
| `kaggle_smoke` | kiểm tra pipeline (vài chục bước) | whitespace patching, hash 50002 |
| `kaggle_t4` | train thật trên Kaggle T4 16GB | đóng băng thân Qwen (stage 1 kiểu Bolmo), hash 50002 |
| `rtx24gb` | train thật trên máy RTX 24GB | train cả thân Qwen, hash 100002 |
| `cpu_plumbing` | kiểm tra trên Mac với Qwen tí hon ngẫu nhiên | chỉ dùng `train.max_steps=0` |

## Thứ tự chạy

```bash
# 0. env (Python >= 3.10, transformers >= 5.20.1); trên GPU cài thêm flash-linear-attention
uv venv .venv-qwen --python 3.11 && uv pip install --python .venv-qwen/bin/python -r requirements-blt-qwen.txt
python setup/download_prepare_data.py --target-gb 5 --out-dir data/prepared/fineweb2_vi

# 1. entropy model tiếng Việt
python -m vislm.train_blt_qwen experiments/blt_qwen/configs/entropy_lm.yaml profile=rtx24gb

# 2. BLT-Qwen (mặc định entropy patching, threshold tự hiệu chỉnh để patch trung bình 4.5 byte)
python -m vislm.train_blt_qwen experiments/blt_qwen/configs/blt_qwen35_08b.yaml profile=rtx24gb \
    patching.entropy_ckpt=runs/blt_qwen_entropy_lm/entropy_lm.pt

# chạy tiếp sau khi bị ngắt (vd. hết giới hạn 12h của Kaggle)
python -m vislm.train_blt_qwen <config> profile=... resume_from=<run_dir>/checkpoints/latest.pt
```

## Kaggle

Notebook được viết thành file `.py` kiểu py-percent trong `kaggle/`, rồi build ra `.ipynb`:

```bash
PATH=$PWD/.venv-qwen/bin:$PATH bash setup/sync_kaggle_code_dataset.sh     # đẩy code lên dataset
python setup/build_kaggle_kernel.py experiments/blt_qwen/kaggle/blt_qwen_smoke.py \
    --slug vislm-blt-qwen-smoke --out /tmp/k
kaggle kernels push -p /tmp/k --accelerator NvidiaTeslaT4                    # P100 đã bị Kaggle bỏ
kaggle kernels output nguyennn263/vislm-blt-qwen-smoke -p runs/<tên-run>
```

Cần Kaggle CLI >= 2.x (có cờ `--accelerator`; đã cài sẵn trong `.venv-qwen`). Thư mục
`/kaggle/working` giới hạn 20GB, mà một checkpoint BLT-Qwen train cả thân Qwen nặng khoảng
6GB. Vì vậy các run thử đặt `train.save_final=false`.
