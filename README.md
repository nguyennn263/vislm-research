# vislm-research

Nghiên cứu SLM tiếng Việt (nhỏ – nhanh – nhẹ – chính xác).

- [`plans/PLAN.md`](plans/PLAN.md) — kế hoạch nghiên cứu đầy đủ: câu hỏi nghiên cứu, giả thuyết, thí nghiệm, tiêu chí đi tiếp/dừng lại.
- [`phases/README.md`](phases/README.md) — lộ trình thực thi theo phase (ánh xạ từ plan sang thứ tự chạy thật + compute cụ thể).

## Cấu trúc repo

```
vislm/            # core library (tokenizers, backbones, moe_lora, alignment, rag, ood, routing)
experiments/      # 1 folder / pillar, config trùng ID câu hỏi con trong plans/PLAN.md
evals/            # harness đánh giá dùng chung nhiều phase
setup/            # tạo env, tải tokenizer/dữ liệu
data/{raw,prepared}/  # không commit dữ liệu thô
runs/             # dump_dir mỗi lần chạy (config + code snapshot + metrics)
```

`data/` bị `.gitignore` chặn hoàn toàn (kể cả file lạc trong đó, ví dụ `kaggle.json`) —
tồn tại local only, tự tạo lại bằng `mkdir -p data/{raw,prepared}`. `raw/` chứa dữ liệu
tải về nguyên bản theo Phụ lục "Nguồn dữ liệu tiếng Việt hiện có" trong `plans/PLAN.md`;
`prepared/` chứa dữ liệu đã làm sạch/tokenize, sinh ra bởi `setup/download_prepare_data.py`.

## Bắt đầu

```bash
bash setup/create_env.sh   # cài requirements.txt — đủ cho Phase 0/1, chạy local
source .venv/bin/activate
python setup/download_tokenizer.py --model phogpt
```

Phase 2 trở đi (cần train) cài thêm `pip install -r requirements-train.txt` trên máy RTX 24GB (SSH) hoặc Kaggle — không cài torch trên máy local.
