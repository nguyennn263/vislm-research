# Mục lục Phase

Nguồn sự thật cho RQ/giả thuyết/tiêu chí đi tiếp là [`plans/PLAN.md`](../plans/PLAN.md). Các file trong `phases/` chỉ là lớp điều phối thực thi — mỗi phase gom một nhóm câu hỏi con theo đúng "Sơ đồ phụ thuộc" trong PLAN.md, gắn compute cụ thể, và theo dõi trạng thái.

| Phase | Trụ cột / câu hỏi con | Compute | Trạng thái |
|---|---|---|---|
| [0 — Scaffold](phase-0-scaffold.md) | — (hạ tầng) | local | Hoàn tất |
| [1 — Không cần train](phase-1-no-train.md) | 1.1, 1.2, 5.1 (Trụ cột 0 tạm hoãn) | local + Kaggle | Có kết quả (1.1, 1.2, 5.1) |
| [2 — Train nhỏ](phase-2-small-train.md) | 1.3, 2.1, 2.2, 5.2 | Kaggle GPU (đủ dùng ở scale này) | A/B/C trained ở 4 quy mô (30→500→5000→20000 bước); A>>{B,C}, khoảng cách GIÃN nhất quán qua 3 lần scale — khuyến nghị BPE ở scale nhỏ; C>B bền vững (4-8%); backbone (2.1/2.2) chưa viết |
| [3 — Tier 2, cần backbone](phase-3-tier2-backbone-ready.md) | 3, 6 | RTX 24GB (SSH) | Chưa bắt đầu |
| [4 — Alignment](phase-4-alignment.md) | 4 | RTX 24GB (SSH) | Chưa bắt đầu |
| [5 — Router (optional)](phase-5-router-optional.md) | 7 | TBD | Không nằm trong lộ trình mặc định |

**Quy tắc:** chỉ mở phase kế tiếp khi phase hiện tại đạt tiêu chí đi tiếp ghi trong PLAN.md cho từng câu hỏi con liên quan — không nhảy cóc (nguyên tắc xuyên suốt #2 trong PLAN.md).
