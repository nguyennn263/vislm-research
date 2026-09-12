# Pillar 1 — Patch encoder / biểu diễn đầu vào

Xem `plans/PLAN.md` — Trụ cột 1.

- `configs/1_1*.yaml`, `1_2*.yaml` — thuộc `phases/phase-1-no-train.md` (chạy được ngay, không cần GPU thật, không phụ thuộc domain).
- `configs/1_3_arm_*.yaml` — thuộc `phases/phase-2-small-train.md` (cần train nhỏ trên máy RTX 24GB, compute-matched giữa 3 arm — xem "Nguyên tắc thiết kế thí nghiệm" trong PLAN.md).

Tên file config trùng ID câu hỏi con trong `plans/PLAN.md` để truy vết 1-1 (quy tắc thực thi #1 trong PLAN.md).
