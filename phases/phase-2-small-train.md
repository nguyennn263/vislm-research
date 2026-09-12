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

**Đã chạy Arm A ở quy mô lớn hơn** (0.5GB FineWeb2, 2000 bước, cùng GPU) — xem
`runs/2026_09_13_train_arm_a_scaled_kaggle_gpu/`. Loss giảm từ 9.998 xuống 5.66 (min
4.76) — hội tụ tốt hơn hẳn bản debug (chỉ tới ~7 sau 300 bước). Train 2000 bước chỉ mất
~135 giây (~0.068s/bước, khớp con số đo trước đó); **bottleneck thật sự giờ là bước
tokenize 0.5GB text (~281 giây, vòng lặp Python gọi `tok.encode()` từng tài liệu)**, không
phải training — nếu scale data lớn hơn nữa, đây là chỗ cần tối ưu trước (batch tokenize),
không phải model.

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

## Compute-matching (đã đo và chỉnh)
`experiments/pillar1_patch_encoder/measure_compute_match.py` (dùng heuristic ~6×params
FLOPs/byte, xem `vislm/backbones/flops.py`) đo được:
- Arm B cùng `d_model=256` với Arm A cho FLOPs/byte gấp **~1.9 lần** Arm A — vì Arm B tốn
  thêm compute cho local encoder/decoder mỗi BYTE, ngoài latent transformer mỗi PATCH
  (Arm A chỉ tốn compute mỗi TOKEN qua 1 tầng duy nhất). latent_params Arm B (6.48M) vốn
  THẤP hơn Arm A (10.1M) dù cùng d_model/n_layers vì Arm A có bảng embedding từ vựng lớn
  (PhoGPT vocab_size=20480) — khác biệt CẤU TRÚC giữa BPE và byte-level, không phải lỗi
  cấu hình, nên phải so FLOPs chứ không phải đếm tham số thô.
- Giảm Arm B/C xuống `d_model=192, n_heads=6` đưa tỷ lệ về **~1.06–1.14x** — trong ngưỡng
  ±20%, dùng làm giá trị hiện tại (đã cập nhật trong `1_3_arm_B_blt.yaml`/`_C_...yaml`).
- Đo entropy thật trên data cũng cho thấy `entropy_threshold: 1.5` (đoán ban đầu) quá
  thấp — entropy trung bình thật ~3.0-3.7, khiến gần như mọi byte tự thành 1 patch. Đã
  chỉnh lên 3.7 (điểm bắt đầu, rất nhạy quanh vùng này, xem comment trong config).
- Đây là ước lượng (heuristic 6N), không phải FLOPs đo bằng profiler — đủ dùng để chỉnh
  kích thước tương đối, không phải con số khoa học cuối cùng.

## Tối ưu tốc độ Arm B (đã làm 1 bước, đã đo lại trên Kaggle GPU)
Local decoder trước đây gọi transformer riêng cho TỪNG patch (hàng trăm lần/sequence) —
giờ đã batch tất cả patch trong 1 sequence lại thành 1 lệnh gọi duy nhất (pad + attention
mask kết hợp causal+padding, xem `build_causal_padding_mask` trong `blt_lm.py`). Đã verify
cho kết quả **giống hệt bit-for-bit** so với bản vòng lặp cũ (cùng seed) — thuần tối ưu
tốc độ, không đổi hành vi.

**Đo lại trên cùng GPU (P100):** ~0.256s/bước, so với ~1.63s/bước trước khi tối ưu —
nhanh hơn **~6.4 lần**. Xem `runs/2026_09_13_train_arm_b_debug_kaggle_gpu_v2_optimized/`
(so với `runs/2026_09_13_train_arm_b_debug_kaggle_gpu/`). Vẫn chậm hơn Arm A (~0.068s/
bước) khoảng 3.8 lần — vòng lặp Python theo từng SEQUENCE trong batch (không phải từng
patch nữa) vẫn còn, là bước tối ưu tiếp theo nếu cần.

## Arm C — đã train thật trên Kaggle, so trực tiếp với Arm B
Cùng model size (4.55M tham số, latent 3.67M), cùng số bước (30), cùng dung lượng data
mục tiêu (50MB) với Arm B — xem `runs/2026_09_13_train_arm_c_debug_kaggle_gpu/`.

| | Arm B (entropy thuần) | Arm C (entropy + mồi âm tiết) |
|---|---|---|
| Loss đầu | 5.608 | 5.620 |
| Loss cuối (bước 30) | 3.955 | **3.645** |
| Loss thấp nhất | 3.917 | **3.630** |
| Tốc độ/bước (P100) | ~0.256s | ~0.094s |

**Arm C hội tụ thấp hơn Arm B rõ rệt ở cùng số bước** — khớp với phát hiện ở 1.2a/1.2b
(Phase 1: đỉnh entropy trùng ranh giới âm tiết ~90%+), gợi ý mồi âm tiết cho patch cấu
trúc hữu ích hơn chỉ dựa entropy. **Đây mới là tín hiệu ở quy mô debug (30 bước/50MB,
1 seed), chưa phải kết luận** — cần chạy dài hơn, nhiều seed hơn, data lớn hơn trước khi
coi là xác nhận cho câu hỏi con 1.3.

## So sánh công bằng cả 3 arm ở quy mô lớn hơn (500 bước / 0.2GB, cùng config đầy đủ)
Cùng `max_seq_len=512, batch_size=8` (không dùng override debug nữa), cùng
`dataset_target_gb=0.2`, cùng 500 bước — xem `runs/2026_09_13_train_arm_{a,b,c}_500_kaggle_gpu/`.
Loss thô KHÔNG so được trực tiếp giữa Arm A (vocab PhoGPT 20480) và Arm B/C (vocab byte
256) — đã quy đổi hết về **bits/byte** (loss ÷ ln 2, và với Arm A chia thêm cho
~5.06 byte/token đo ở Phase 1):

| | Arm A (BPE) | Arm B (entropy thuần) | Arm C (entropy + mồi âm tiết) |
|---|---|---|---|
| bits/byte (TB 20 bước cuối) | **2.054** | 3.583 | 3.375 |
| bits/byte thấp nhất | **1.858** | 3.123 | 2.770 |
| Tốc độ/bước (P100) | **~0.068s** | ~0.817s | ~0.399s |

**Xếp hạng bits/byte: A < C < B** (thấp hơn = tốt hơn). Arm A vẫn dẫn đầu rõ rệt ở quy mô
này — tokenizer PhoGPT được pretrain sẵn trên corpus khổng lồ cho nó lợi thế khởi đầu lớn
mà 1 model byte-level học từ đầu chưa bắt kịp trong 500 bước. Nhưng **trong nhóm
byte-level, Arm C vẫn nhất quán tốt hơn Arm B** (đúng ở cả 30 bước lẫn 500 bước) — tín
hiệu mồi âm tiết hữu ích được củng cố thêm, không phải ngẫu nhiên của lần chạy debug.
Arm C cũng nhanh hơn Arm B ở quy mô này (~0.399s vs ~0.817s/bước).

**Sự cố kỹ thuật gặp phải:** Arm C 500-bước fail 3 lần liên tiếp với lỗi
`FileNotFoundError` trước khi chạy đúng ở lần thứ 4 — hoá ra Kaggle mount Kaggle Dataset
KHÔNG NHẤT QUÁN giữa các lần chạy (có lúc `/kaggle/input/<slug>/`, có lúc thêm tầng
`/kaggle/input/datasets/<owner>/<slug>/`). Đã sửa notebook để tự detect path lúc chạy
(xem code trong `train_arm_c_500/`) thay vì hard-code — cần áp dụng cách này cho mọi
notebook Kaggle mới sau này.

## Trạng thái
Cả 3 arm (A, B, C) đã viết xong, compute-matched (~1.06-1.14x FLOPs/byte giữa B/C và A),
decoder Arm B/C đã tối ưu tốc độ (~6.4x), và đã train thật ở 2 quy mô (debug 30 bước và
500 bước, cùng điều kiện, cùng data target) trên Kaggle GPU. **Tín hiệu nhất quán qua cả
2 quy mô cho câu hỏi con 1.3: trong nhóm byte-level, Arm C (mồi âm tiết) > Arm B (entropy
thuần); nhưng cả hai đều thua Arm A (BPE) ở giai đoạn training ngắn này** — khớp hướng
với 1.2a/1.2b
(entropy trùng ranh giới âm tiết ~90%+) cho phần B-vs-C, nhưng A-vs-{B,C} cho thấy BPE có
lợi thế lớn ở quy mô nhỏ (chưa rõ có giữ khi scale lên nhiều hơn, đây là đúng câu hỏi mà
BLT paper trả lời ở scale lớn hơn nhiều). Còn thiếu trước khi kết luận thật cho 1.3: (1)
quét lại `entropy_threshold` ở scale training thật (số hiện tại đo trên mẫu nhỏ, rất
nhạy), (2) chạy dài hơn/nhiều seed hơn/data lớn hơn để xem A-vs-{B,C} có đổi chiều không
khi byte-level "bắt kịp" (đúng như BLT paper claim ở scale lớn), (3) vector hoá luôn vòng
lặp theo sequence nếu cần scale lớn hơn nữa, (4) áp dụng cách detect CODE_DIR động cho
mọi notebook Kaggle cũ (train_arm_a_debug, train_arm_b_debug, train_arm_c_debug,
data_pipeline_smoke_test) trước khi rerun chúng — hiện chỉ mới sửa cho các notebook 500
bước. 2.1/2.2 (Trụ cột 2 — backbone SSM) chưa viết code.
