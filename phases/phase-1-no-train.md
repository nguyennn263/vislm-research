# Phase 1 — Tier 1, không cần train

Chi tiết câu hỏi con / tiêu chí đi tiếp: xem [`plans/PLAN.md`](../plans/PLAN.md) — Trụ cột 1 (mục "Câu hỏi con 1.1", "1.2"), Trụ cột 5 (mục "Câu hỏi con 5.1").

## Phạm vi
- **1.1a–d** — phân tích lỗi BPE hiện tại (PhoGPT/ViGPT) trên từ ghép hiếm, tên riêng, số liệu, lỗi chính tả. Không train, chỉ tokenize + đếm.
- **1.2a–b** — so khớp đỉnh entropy byte-tiếp-theo (mGPT/XGLM và PhoGPT) với ranh giới âm tiết. Chỉ forward pass, không train.
- **5.1** — đánh giá retriever tiếng Việt trên cặp đoạn văn gần nghĩa khác sự thật. Chỉ inference embedding.
- **Trụ cột 0** — TẠM HOÃN theo yêu cầu (chưa chọn miền PoC/domain expert). Không chặn phần còn lại của Phase 1 vì 1.1/1.2/5.1 không phụ thuộc domain.

## Compute
- 1.1: local đủ (chỉ cần tokenizer, không cần model weights đầy đủ).
- 1.2, 5.1: cần forward pass qua model — ưu tiên Kaggle (free GPU) để không tốn giờ máy RTX 24GB cho việc thuần suy luận.

## Điều kiện đi tiếp
Theo đúng "Tiêu chí đi tiếp" ghi ở từng câu hỏi con trong PLAN.md (VD: 1.1 — tỷ lệ vỡ token tăng > 2 lần thì xác nhận vấn đề; 1.2 — cả 2 lượt đo trùng khớp > 70%).

## Trạng thái
1.1/1.2 đã chạy baseline đầu tiên trên Kaggle (CPU) — xem
`runs/2026_09_12_pillar1_phase1_kaggle_baseline/metrics.jsonl` và notebook trong
`experiments/pillar1_patch_encoder/kaggle/`.

**Kết quả (n nhỏ, cần mở rộng wordlist trước khi coi là kết luận cuối):**
- 1.1a (từ ghép hiếm): 1.43x — chưa vượt ngưỡng 2x.
- 1.1b (tên riêng): 1.25x — chưa vượt ngưỡng 2x.
- 1.1c (số liệu): 3.76x — **vượt ngưỡng 2x, xác nhận vấn đề.**
- 1.1d (lỗi chính tả): 1.0x — không có khác biệt với tokenizer này.
- 1.2a (XGLM-564M, đa ngôn ngữ): 90.9% đỉnh entropy trùng ranh giới âm tiết — vượt ngưỡng 70%.
- 1.2b (GPT2-Vietnamese, riêng tiếng Việt): 90.0% — vượt ngưỡng 70%, và rất gần 1.2a
  → củng cố giả thuyết "mồi âm tiết" đáng để thử ở 1.3 Arm C.

Lưu ý: tokenizer dùng là `NlpHUST/gpt2-vietnamese`, không phải `vinai/PhoGPT-4B` như
gợi ý gốc trong PLAN.md — PhoGPT-4B lỗi tương thích transformers hiện tại (xem
`vislm/tokenizers/bpe_baseline.py`). 5.1 chưa chạy.
