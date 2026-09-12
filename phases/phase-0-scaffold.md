# Phase 0 — Hạ tầng &amp; scaffold repo

Xem [`plans/PLAN.md`](../plans/PLAN.md) mục "Phụ lục: Cấu trúc repo tham khảo" cho layout đầy đủ và lý do mượn cấu trúc từ Meta Lingua / mamba_ssm / trl.

## Mục tiêu
Dựng khung repo tối thiểu để Phase 1 chạy được, không viết trước code cho các trụ cột chưa tới lượt (nguyên tắc thực thi #4 trong PLAN.md).

## Việc cần làm
- [x] Đổi tên `plans/Plan.md` → `plans/PLAN.md`, tạo `phases/` liên kết ngược lại.
- [x] Dựng cây thư mục `vislm/`, `experiments/`, `evals/`, `setup/`, `data/`, `runs/`.
- [x] `requirements.txt` (local, Phase 0/1) + `requirements-train.txt` (RTX 24GB/Kaggle, Phase 2+) + script tải tokenizer/dữ liệu.
- [x] `vislm/tokenizers/` với baseline BPE (Arm A) — đủ để chạy phân tích 1.1 ở Phase 1.
- [x] Quy ước `runs/<timestamp>_<phase>_<arm>/{config.yaml, code/, metrics.jsonl, logs/}` (xem `runs/README.md`).
- [x] `pip install -r requirements.txt` đã chạy thử trong `.venv/` — sạch, `import vislm.tokenizers` OK (transformers 4.57.6, không cần torch).
- [ ] 1 lần chạy mẫu thật (VD: 1.1a) ghi ra `runs/` đúng quy ước — để lại cho lúc bắt đầu Phase 1.

## Compute
Local (Apple M5, CPU). Không cần GPU cho phase này.

## Điều kiện xong (exit criteria)
- `pip install -r requirements.txt` chạy được.
- `import vislm.tokenizers` không lỗi.
- Có tối thiểu 1 lần chạy mẫu ghi đúng quy ước vào `runs/`.

## Trạng thái
Đang làm.
