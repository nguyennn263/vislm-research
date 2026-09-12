# Phase 3 — Tier 2, cần backbone đã chốt

Chi tiết: xem [`plans/PLAN.md`](../plans/PLAN.md) — Trụ cột 3 (MoE-LoRA), Trụ cột 6 (OOD gate).

## Phạm vi
- **Trụ cột 3** (3.1–3.3) — LoRA expert theo miền, chống catastrophic forgetting, router cấp token. **Phần dữ liệu theo miền cụ thể vẫn phụ thuộc quyết định Trụ cột 0 đang tạm hoãn** — cần chọn miền + domain expert trước khi chạy 3.1 thật (có thể chuẩn bị code/pipeline trước bằng dữ liệu placeholder).
- **Trụ cột 6** (6.1–6.2) — cổng OOD, không phụ thuộc domain nên chạy độc lập được ngay khi có checkpoint từ Phase 2.

## Điều kiện mở phase này
Backbone từ Phase 2 (Trụ cột 2) đã chốt kiến trúc + có checkpoint ổn định.

## Compute
RTX 24GB (SSH) — LoRA train nhẹ, vừa với 24GB.

## Trạng thái
Chưa bắt đầu — chờ Phase 2 và (cho phần 3) quyết định miền PoC.
