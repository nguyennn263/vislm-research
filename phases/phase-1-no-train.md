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
1.1/1.2 đã chạy 2 vòng trên Kaggle (CPU): v1 (n nhỏ, tham khảo) và v2 (n lớn hơn,
nhiều mẫu văn bản hơn — kết quả dùng để kết luận). Xem
`runs/2026_09_12_pillar1_phase1_kaggle_baseline_v2/metrics.jsonl` và notebook trong
`experiments/pillar1_patch_encoder/kaggle/`.

**Kết quả v2 (n=30–58 từ/cụm mỗi nhóm, 5 đoạn văn cho 1.2):**
- 1.1a (từ ghép hiếm, n=58): 1.38x — chưa vượt ngưỡng 2x.
- 1.1b (tên riêng, n=52): 1.23x — chưa vượt ngưỡng 2x.
- 1.1c (số liệu, n=30): 3.42x (độ lệch chuẩn cao, 1.67 — một số dạng số vỡ token nhiều
  hơn hẳn) — **vượt ngưỡng 2x, xác nhận vấn đề, và nhất quán với v1 (3.76x).**
- 1.1d (lỗi chính tả, 5 cặp đoạn văn): 1.03x ± 0.04 — nhất quán qua nhiều chủ đề, xác
  nhận **không** có khác biệt đáng kể với tokenizer này (không phải do v1 chỉ có 1 mẫu).
- 1.2a (XGLM-564M, đa ngôn ngữ): 98–100% đỉnh entropy trùng ranh giới âm tiết (5 đoạn
  văn, top20%/top30%) — vượt xa ngưỡng 70%, ổn định qua các mẫu.
- 1.2b (GPT2-Vietnamese, riêng tiếng Việt): 88.6–92% — cũng vượt ngưỡng 70%, gần với
  1.2a (cách nhau ~7-10 điểm %, không phải trùng khớp tuyệt đối nhưng cùng xu hướng)
  → củng cố giả thuyết "mồi âm tiết" đáng để thử ở 1.3 Arm C.

**Kết luận đi tiếp cho 1.1/1.2 (theo tiêu chí trong PLAN.md):** chỉ 1.1c (số liệu) và
1.2 (cả 2 lượt đo) đạt tiêu chí xác nhận vấn đề/đi tiếp; 1.1a, 1.1b, 1.1d chưa đạt
ngưỡng với tokenizer NlpHUST/gpt2-vietnamese — cần cân nhắc có thử lại với tokenizer
khác (VD PhoBERT word-segmented) trước khi kết luận "từ ghép hiếm/tên riêng không phải
vấn đề" một cách chắc chắn, vì kết quả có thể phụ thuộc tokenizer cụ thể.

Lưu ý: tokenizer dùng là `NlpHUST/gpt2-vietnamese`, không phải `vinai/PhoGPT-4B` như
gợi ý gốc trong PLAN.md — PhoGPT-4B lỗi tương thích transformers hiện tại (xem
`vislm/tokenizers/bpe_baseline.py`).

**5.1 — retriever tiếng Việt phân biệt cặp đoạn văn gần nghĩa khác sự thật:** đã chạy
trên Kaggle (CPU), 20 cặp (câu hỏi, đoạn đúng, đoạn gài sai lệch 1 dữ kiện then chốt —
năm/số/tên/địa danh), xem `runs/2026_09_12_pillar5_5_1_kaggle_baseline/metrics.jsonl`.
- `bkai-foundation-models/vietnamese-bi-encoder`: 80% chọn đúng (16/20), margin cosine
  trung bình +0.098 (độ lệch chuẩn 0.12 — khá phân tán, có cặp bị chọn sai hẳn).
- `dangvantuan/vietnamese-embedding`: 90% chọn đúng (18/20), margin +0.079 (ổn định hơn,
  độ lệch chuẩn 0.08).

Chưa có ngưỡng "đủ tin cậy" định trước ở Trụ cột 0 (0.2/0.3) để so — nhưng 80-90% với
n=20 nghĩa là còn 2-4 cặp mỗi model bị lẫn lộn dữ kiện, khớp đúng lo ngại nêu trong RQ5:
retriever hiện tại chưa đủ mạnh để tin cậy hoàn toàn cho cập nhật tri thức dạng "chỉ thay
đổi 1 con số/năm" — cần cân nhắc thêm bước lọc/rerank chứ không chỉ dựa vào cosine
similarity thô.
