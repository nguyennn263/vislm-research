# data/

Không commit dữ liệu thô vào git (`.gitignore` đã loại `data/raw/*` và `data/prepared/*`).

- `raw/` — dữ liệu tải về nguyên bản theo Phụ lục "Nguồn dữ liệu tiếng Việt hiện có" trong `plans/PLAN.md`.
- `prepared/` — dữ liệu đã làm sạch/tokenize, sinh ra bởi `setup/download_prepare_data.py`.
