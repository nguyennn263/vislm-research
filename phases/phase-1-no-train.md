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
1.1/1.2 đã chạy 3 vòng trên Kaggle (CPU): v1/v2 dùng `NlpHUST/gpt2-vietnamese` (model
cộng đồng nhỏ, ~535 tải/tháng, không có paper riêng — dùng tạm vì `vinai/PhoGPT-4B` lỗi
tương thích transformers bản Kaggle ship mặc định). **v3 dùng đúng `vinai/PhoGPT-4B`
thật** (VinAI Research, có paper arXiv:2311.02945, "de facto" LLM tiếng Việt) sau khi
tìm ra cách fix: ghim `transformers==4.46.3` trước khi load — **v3 là kết quả dùng để
kết luận**, v1/v2 giữ lại chỉ để đối chiếu. Xem
`runs/2026_09_12_pillar1_phase1_kaggle_baseline_v3_phogpt/metrics.jsonl` và notebook
trong `experiments/pillar1_patch_encoder/kaggle/`.

**Kết quả v3 (vinai/PhoGPT-4B thật, n=30–58 từ/cụm mỗi nhóm, 5 đoạn văn cho 1.2):**
- 1.1a (từ ghép hiếm, n=58): 1.34x so baseline — chưa vượt ngưỡng 2x (nhất quán hướng
  với v2's 1.38x, dù baseline của PhoGPT tokenizer tự nó đã > 1.0, khác gpt2-vietnamese).
- 1.1b (tên riêng, n=52): 1.11x so baseline — chưa vượt ngưỡng 2x, thấp hơn v2 (1.23x)
  → PhoGPT xử lý tên riêng tốt hơn gpt2-vietnamese một chút.
- 1.1c (số liệu, n=30): 2.51x so baseline — **vẫn vượt ngưỡng 2x, xác nhận vấn đề**,
  nhất quán qua cả 3 vòng chạy (v1 3.76x, v2 3.42x, v3 2.51x — hướng luôn rõ ràng dù
  độ lớn khác nhau theo tokenizer).
- 1.1d (lỗi chính tả, 5 cặp đoạn văn): 1.26x ± 0.07 — chưa vượt ngưỡng 2x, nhưng **cao
  hơn rõ rệt** so với gpt2-vietnamese (1.03x) → PhoGPT nhạy với việc bỏ dấu hơn, hợp lý
  vì vocab của nó có nhiều token gắn với ký tự có dấu tiếng Việt hơn.
- 1.2a (XGLM-564M, đa ngôn ngữ): 96.8–100% đỉnh entropy trùng ranh giới âm tiết (5 đoạn
  văn, top20%/top30%) — vượt xa ngưỡng 70%, nhất quán qua các vòng.
- 1.2b (PhoGPT-4B thật, riêng tiếng Việt): 93.8–94.2% — vượt ngưỡng 70%, và **gần XGLM
  hơn hẳn** so với v2 (cách nhau chỉ ~3-6 điểm %, so với ~7-10 điểm % của gpt2-vietnamese)
  → củng cố mạnh hơn giả thuyết "mồi âm tiết" đáng để thử ở 1.3 Arm C, giờ với model
  thật sự đại diện cho "LLM pretrain riêng tiếng Việt."

**Kết luận đi tiếp cho 1.1/1.2 (theo tiêu chí trong PLAN.md, dựa trên v3):** chỉ 1.1c
(số liệu) và 1.2 (cả 2 lượt đo) đạt tiêu chí xác nhận vấn đề/đi tiếp; 1.1a, 1.1b, 1.1d
chưa đạt ngưỡng 2x kể cả với PhoGPT-4B thật — kết luận này giờ đáng tin hơn v1/v2 vì đã
dùng đúng model uy tín PLAN.md đề xuất, không còn phụ thuộc vào một model thay thế ít
được kiểm chứng.

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
