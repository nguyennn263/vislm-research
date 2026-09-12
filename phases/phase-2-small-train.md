# Phase 2 — Tier 1, cần train nhỏ

Chi tiết: xem [`plans/PLAN.md`](../plans/PLAN.md) — Trụ cột 1 (mục "Câu hỏi con 1.3"), Trụ cột 2 (toàn bộ), Trụ cột 5 (mục "Câu hỏi con 5.2").

## Phạm vi
- **1.3 Arm A/B/C** — BPE+Transformer vs BLT thuần vs BLT+mồi âm tiết, compute-matched.
- **2.1a–b** — compute-matched ratio sweep SSM:Attention + đối chứng RoPE.
- **2.2a** — tái lập MOHAWK/MambaInLlama trên baseline tiếng Anh (bắt buộc, sanity check trước 2.2b).
- **2.2b** — mở rộng sang checkpoint tiếng Việt (chỉ chạy sau khi 2.2a đạt kết quả tương đương công bố gốc).
- **5.2** — ranh giới RAG vs weight-edit (AdaPLE) cho loại tri thức lặp lại nhiều lần.

## Điều kiện mở phase này
Phase 1 phải có kết luận rõ ràng cho 1.1/1.2 (đạt hoặc không đạt tiêu chí đi tiếp) — không mở Arm B/C của 1.3 khi chưa biết BPE thất bại ở đâu và ranh giới âm tiết có ý nghĩa hay không.

## Compute
Máy RTX 24GB (SSH) cho toàn bộ job train thật. Kaggle chỉ dùng để debug pipeline ở quy mô cực nhỏ trước khi chạy job đầy đủ trên máy 24GB (tránh tốn giờ máy riêng cho lỗi cài đặt).

## Trạng thái
Chưa bắt đầu — chờ Phase 1.
