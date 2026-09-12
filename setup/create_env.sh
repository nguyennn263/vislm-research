#!/usr/bin/env bash
# Tạo virtualenv cho Phase 0/1 (chạy local hoặc trên máy RTX 24GB / Kaggle notebook).
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Xong. Kích hoạt bằng: source .venv/bin/activate"
