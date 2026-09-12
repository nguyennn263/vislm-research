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

## Dữ liệu
Corpus pretraining đã chốt: **FineWeb2 tiếng Việt** (`HuggingFaceFW/fineweb-2`, config
`vie_Latn` — ~61M dòng, ~130GB text thô, streaming trên HF). `setup/download_prepare_data.py`
stream + subsample thành shard `.jsonl` trong `data/prepared/fineweb2_vi/`, tính theo
dung lượng byte thô (không theo tokenizer cụ thể) để dùng chung, công bằng giữa Arm
A/B/C — script đã test chạy thật (streaming, không tải nguyên 130GB). Dung lượng mục
tiêu (`--target-gb`) và số FLOPs/tham số cụ thể vẫn để `null` trong config — chốt khi
biết rõ thời gian/compute thực tế trên máy RTX 24GB.

## Config động: test nhỏ trên Kaggle trước khi chạy full trên RTX 24GB
Mọi config trong `configs/` đều có `compute: rtx24gb_ssh` là giá trị mặc định cho lần
chạy đầy đủ, nhưng không hard-code — `vislm/args.py` cho phép override qua dot-list
CLI (kiểu Meta Lingua), VD:
```
python train.py configs/1_3_arm_A_bpe.yaml compute=kaggle_debug dataset_target_gb=0.05
```
Cùng 1 file config chạy được cả bản debug nhỏ (Kaggle, vài chục MB) lẫn bản đầy đủ
(RTX 24GB) — chỉ khác giá trị override, không cần sửa tay hay tạo file riêng.

**Đã smoke-test (data only):** stream 50MB từ FineWeb2 tiếng Việt trên Kaggle + tokenize
bằng tokenizer Arm A (PhoGPT-4B) — xem
`runs/2026_09_12_data_pipeline_smoke_test_kaggle/metrics.jsonl`. Kết quả: 7.421 tài
liệu, ~49.7MB text → 9.82M token (~5.06 byte/token), độ dài tài liệu trung vị 731 token.

## Cách chạy: code dùng chung giữa Kaggle và RTX 24GB
Không copy-paste code vào notebook Kaggle — `vislm/` + `experiments/pillar1_patch_encoder/configs/`
+ `setup/` được đóng gói thành Kaggle Dataset riêng (`nguyennn263/vislm-research-code`,
cập nhật bằng `kaggle datasets version -p <export dir>`), mount vào kernel tại
`/kaggle/input/datasets/nguyennn263/vislm-research-code/` (lưu ý: path có thêm tầng
`datasets/<username>/` so với quy ước `/kaggle/input/<slug>/` thường thấy). Notebook chỉ
làm 3 việc: set `PYTHONPATH`, chạy `setup/download_prepare_data.py`, rồi
`python -m vislm.train <config> <overrides>` — **cùng một lệnh, cùng một config**, sẽ
chạy trên máy RTX 24GB sau này, chỉ đổi giá trị override.

**Đã train thật Arm A (transformer_standard) trên Kaggle** — không chỉ smoke-test data
nữa, mà chạy trọn `vislm/train.py` 300 bước trên 50MB FineWeb2 + tokenizer PhoGPT-4B, xem
`runs/2026_09_12_train_arm_a_debug_kaggle/`. Model 10.1M tham số, loss giảm từ 9.998
(≈ ln(20480), đúng random-init cho vocab PhoGPT) xuống dao động 6.6–7.6 — xác nhận có
học thật, pipeline train hoạt động đúng đầu-cuối.

**Đã fix GPU:** Kaggle cấp GPU P100 (sm_60), torch bản mới nhất (Kaggle ship 2.10+cu128)
đã bỏ hỗ trợ sm_60 (Pascal, bỏ từ torch 2.8 trở đi). Notebook giờ tự phát hiện P100 qua
`nvidia-smi` và CHỈ khi đó mới ghim `torch==2.7.1+cu126` (bản mới nhất còn hỗ trợ sm_60)
— nếu Kaggle cấp GPU khác (T4/A100) thì giữ nguyên torch mới nhất sẵn có, không đánh đổi
gì cả. `vislm/train.py`'s `pick_device()` vẫn giữ làm lưới an toàn cuối (rơi về CPU nếu
lỡ vẫn gặp GPU không tương thích).

**Kết quả:** cùng seed, cùng config — chạy GPU (P100 + torch 2.7.1) chỉ mất **~20 giây**
cho 300 bước, so với **~730 giây (12 phút) trên CPU** — nhanh hơn **~33 lần**, loss cuối
gần như giống hệt (6.956 GPU vs 6.956 CPU). Xem
`runs/2026_09_12_train_arm_a_debug_kaggle_gpu/` (so với bản CPU
`runs/2026_09_12_train_arm_a_debug_kaggle/`).

## Arm B/C (BLT) — đã viết và train thật trên Kaggle
`vislm/backbones/modules/entropy_model.py` (byte LM nhỏ, train ngắn rồi freeze, dùng để
tính entropy) + `patching.py` (ranh giới patch theo ngưỡng entropy toàn cục — kiểu đơn
giản hơn trong 2 kiểu patching của paper BLT) + `vislm/backbones/models/blt_lm.py`
(local encoder mean-pool → latent transformer causal theo patch → local decoder tự hồi
quy trong patch). Đây là bản **rút gọn có chủ đích**, không phải port nguyên code
facebookresearch/blt: local encoder dùng mean-pool thay vì cross-attention, xử lý patch
bằng vòng lặp Python theo từng sequence thay vì ragged-batch vector hoá — đã ghi rõ trong
docstring của `blt_lm.py`. Arm C = Arm B + `vislm/tokenizers/syllable_seed.py` (thêm ranh
giới patch ngay sau mỗi byte khoảng trắng, cộng dồn chứ không thay thế ranh giới entropy).

**Đã train thật Arm B trên Kaggle (GPU P100 + torch 2.7.1):** 30 bước debug, model
7.36M tham số (latent 6.48M), loss giảm từ 5.552 (≈ ln(256), đúng random-init cho vocab
byte) xuống 3.82 (min 3.71) — học thật, đúng đầu-cuối. Xem
`runs/2026_09_13_train_arm_b_debug_kaggle_gpu/`. Arm C đã test local (forward+backward
đúng), chưa chạy debug thật trên Kaggle.

**Chưa compute-matched Arm A vs B/C:** latent_params Arm B (6.48M) THẤP hơn Arm A (10.1M)
dù cùng d_model/n_layers/n_heads — vì Arm A có bảng embedding từ vựng PhoGPT (20480 mục,
chiếm phần lớn tham số Arm A), Arm B chỉ cần bảng byte (256 mục). Đây là khác biệt CẤU
TRÚC giữa BPE và byte-level, không phải lỗi cấu hình — nguyên tắc cô lập biến số trong
PLAN.md đòi hỏi so theo **FLOPs**, không phải đếm tham số thô — chưa đo/khớp FLOPs thật.

**Hiệu năng Arm B chậm hơn nhiều so với Arm A** (cùng GPU): ~1.63s/bước so với ~0.067s/bước
của Arm A — do vòng lặp Python theo từng patch (điểm rút gọn nói trên), không phải lỗi. Cần
tối ưu (vector hoá theo batch) trước khi chạy ở quy mô lớn hơn debug.

## Trạng thái
Arm A, Arm B đã viết xong + train thật được trên Kaggle GPU. Arm C viết xong, đã test
local, chưa test debug trên Kaggle. Còn thiếu trước khi so sánh 1.3 có ý nghĩa: (1) đo
FLOPs thật và chỉnh cấu hình để compute-matched giữa 3 arm, (2) tối ưu tốc độ Arm B/C nếu
muốn chạy quy mô lớn hơn debug, (3) quét `entropy_threshold` (đang để tạm 1.5, chưa tune).
2.1/2.2 (Trụ cột 2 — backbone SSM) chưa viết code.
