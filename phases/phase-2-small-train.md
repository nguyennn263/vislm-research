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
Kaggle GPU (free tier: ~30h/tuần, 12h/session, tối đa 2 session GPU song song) đủ dùng
cho toàn bộ quy mô hiện tại (model vài triệu tham số, data dưới 1GB) — không nhất thiết
phải chờ máy RTX 24GB. RTX 24GB chỉ thật sự cần khi model to hơn nhiều (cần >16GB VRAM)
hoặc khi chạm giới hạn quota/session của Kaggle.

## Hạ tầng training (checkpoint/resume/LR schedule/eval) — `vislm/train.py`
Thêm để training "thuận tiện" hơn, học theo quy ước phổ biến (nanoGPT, Meta Lingua):
- **Checkpoint + resume** (`vislm/checkpoint.py`) — lưu định kỳ (`train.save_every`) và
  cuối run vào `<run_dir>/checkpoints/step_<N>.pt` + `latest.pt`. Resume bằng
  `resume_from=<run_dir>/checkpoints/latest.pt` — quan trọng vì Kaggle giới hạn session
  12h. **Lưu ý đã biết:** resume tiếp tục đúng step counter + LR schedule, nhưng data
  iteration bắt đầu lại từ đầu `train_ids` (không nhớ đúng vị trí batch trước khi ngắt) —
  đơn giản hoá chấp nhận được ở quy mô data hiện tại (không shuffle, không lặp nhiều lần).
- **LR schedule** (`vislm/lr_schedule.py`) — warmup tuyến tính + cosine decay (kiểu
  nanoGPT/Chinchilla), thay vì LR cố định như trước.
- **Gradient clipping** (`train.grad_clip`) — thiếu ở bản đầu, có thể gây training không
  ổn định khi scale lên nhiều bước hơn.
- **Train/val split + eval định kỳ** (`train.val_fraction`, `eval_every`, `eval_steps`) —
  giữ % cuối của data stream làm val (không train trên đó), đo val_loss định kỳ để biết
  model có overfit lên corpus nhỏ khi lặp nhiều epoch không — trước đây chỉ có train loss.
- Tất cả field mới đều có default hợp lý trong `configs/1_3_arm_{A,B,C}_*.yaml`, override
  được qua `vislm/args.py` như các field khác.
- Đã test local: cả Arm A và Arm B (có entropy model bị freeze) resume đúng, LR schedule
  đúng pha khi resume, val_loss giảm cùng chiều train_loss (không có dấu hiệu bug).

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

## Scale-up 5.000 bước / 0.3GB — dùng hạ tầng checkpoint/resume/LR-schedule/eval mới
Cùng model size, cùng data (0.3GB), `entropy_pretrain_steps=500` (tăng từ 100), dùng
**val_loss** (đo trên phần data giữ riêng, không train) thay vì train loss — đáng tin
hơn nhiều cho việc so xu hướng. Xem
`runs/2026_09_13_train_arm_{a,b,c}_5000_kaggle_gpu/`.

| bước | A (bpb) | B (bpb) | C (bpb) | B/A | C/A | C/B |
|---|---|---|---|---|---|---|
| 500 | 1.966 | 3.596 | 3.292 | 1.829 | 1.674 | 0.915 |
| 1000 | 1.733 | 3.430 | 3.168 | 1.979 | 1.828 | 0.924 |
| 2000 | 1.600 | 3.276 | 3.092 | 2.048 | 1.933 | 0.944 |
| 3000 | 1.536 | 3.214 | 3.060 | 2.093 | 1.993 | 0.952 |
| 4000 | 1.496 | 3.164 | 3.017 | 2.115 | 2.017 | 0.953 |
| 4500 | 1.485 | 3.155 | 3.016 | 2.125 | 2.031 | 0.956 |

**Phát hiện quan trọng, đi ngược kỳ vọng ban đầu:**
1. **Khoảng cách A vs {B,C} GIÃN RA theo thời gian train, không thu hẹp** (B/A: 1.83x→2.13x,
   C/A: 1.67x→2.03x). Cả 3 arm đều cải thiện tuyệt đối, nhưng Arm A cải thiện nhanh hơn
   trong cửa sổ 5.000 bước này — giả thuyết "byte-level bắt kịp BPE khi có thêm compute"
   **chưa** thấy xảy ra ở scale này. Có thể do: (a) điểm giao cắt thật sự cần scale lớn
   hơn nhiều (đúng như paper BLT report), (b) bản BLT rút gọn ở đây (mean-pool encoder,
   decoder đơn giản) yếu hơn bản đầy đủ, (c) dùng chung LR/warmup cho cả 2 kiến trúc khác
   nhau có thể không tối ưu cho BLT, (d) `entropy_threshold=3.7` vẫn chưa tune kỹ.
2. **Khoảng cách C vs B (tỷ lệ C/B) lại THU HẸP dần về phía 1.0** (0.915→0.956) — Arm C
   vẫn luôn tốt hơn Arm B suốt quá trình, nhưng lợi thế tương đối giảm dần khi train lâu
   hơn. Cách đọc hợp lý: mồi âm tiết cho model 1 "điểm khởi đầu" tốt hơn, nhưng model có
   thể tự học cấu trúc tương đương từ entropy khi có đủ thời gian train — nghĩa là mồi âm
   tiết giúp **hội tụ nhanh hơn ở giai đoạn đầu**, chưa chắc giúp **giá trị hội tụ cuối
   cùng khác biệt nhiều** nếu train đủ lâu (chưa xác nhận được, cần train tới khi cả 2
   plateau thật sự).
3. **Tốc độ Arm B vs C đảo chiều so với lần 500-bước:** ở 5.000 bước, Arm C
   (~0.450s/bước) chậm hơn Arm B (~0.226s/bước) — ngược với lúc 500 bước (C nhanh hơn B).
   Lý do hợp lý: entropy model được pretrain lâu hơn (500 bước) khiến Arm B tự nhiên tạo
   patch dài hơn (ít điểm cắt hơn) khi đã học tốt, còn Arm C vẫn bị ép cắt tại mọi khoảng
   trắng bất kể entropy nói gì → patch ngắn hơn, nhiều patch hơn mỗi sequence → chậm hơn.
   Đây là hệ quả kiến trúc thật, không phải lỗi.

## Scale-up x4: 20.000 bước / 0.5GB — xác nhận và làm rõ thêm xu hướng
Cùng model size/entropy_pretrain_steps=500, chỉ tăng `max_steps` (200000→20000, gấp 4
lần) và `dataset_target_gb` (0.3→0.5GB, đủ tránh lặp data ở scale này). Xem
`runs/2026_09_13_train_arm_{a,b,c}_20000_kaggle_gpu/`.

| bước | A (bpb) | B (bpb) | C (bpb) | B/A | C/A | C/B |
|---|---|---|---|---|---|---|
| 1000 | 1.720 | 3.421 | 3.159 | 1.989 | 1.837 | 0.923 |
| 5000 | 1.422 | 3.165 | 3.054 | 2.226 | 2.148 | **0.965** |
| 9000 | 1.295 | 3.026 | 2.900 | 2.337 | 2.239 | 0.958 |
| 13000 | 1.224 | 2.937 | 2.794 | 2.399 | 2.283 | 0.951 |
| 17000 | 1.187 | 2.906 | 2.743 | 2.449 | 2.311 | 0.944 |
| 19000 | 1.177 | 2.895 | 2.720 | **2.461** | **2.312** | 0.940 |

**Xác nhận và làm rõ thêm 3 điều so với lần 5.000 bước:**

1. **A vs {B,C} tiếp tục giãn mạnh hơn nữa** — B/A lên tới 2.46x, C/A lên tới 2.31x (so
   với 2.13x/2.03x ở 5.000 bước). Xu hướng này giờ đã nhất quán qua **3 lần scale liên
   tiếp** (500→5.000→20.000 bước), không còn là nhiễu ngẫu nhiên của 1 lần chạy.
2. **Arm B có dấu hiệu chững lại (gần plateau) quanh ~2.9 bpb** từ bước 11.000 trở đi
   (dao động hẹp 2.98→2.89), trong khi **Arm A vẫn cải thiện đều đặn, chưa có dấu hiệu
   plateau** (1.30→1.18, vẫn dốc). Arm C cải thiện chậm lại nhưng ít hơn Arm B (2.85→2.72,
   vẫn còn dốc rõ hơn B). Cách đọc hợp lý: Arm A có nhiều "chỗ chứa" hơn để tiếp tục học
   (phần lớn tham số nằm ở bảng embedding vocab lớn), trong khi model byte-level nhỏ hơn
   có thể đang chạm giới hạn năng lực ở kích thước hiện tại — **đây là gợi ý cần scale
   MODEL SIZE (không chỉ số bước) mới thấy được liệu BLT có bắt kịp không**, phù hợp với
   cách paper BLT gốc thiết kế thí nghiệm scale (tăng cả model size lẫn data).
3. **Sửa lại nhận định trước đó về C/B: KHÔNG đơn điệu thu hẹp về 1.0.** Ở 5.000 bước,
   dữ liệu (0.915→0.956) trông giống đang tiến về 1.0 — nhưng nhìn xa hơn tới 20.000 bước,
   tỷ lệ C/B đạt đỉnh ~0.965 quanh bước 5.000 rồi **giảm trở lại** xuống 0.940 (nghĩa là
   lợi thế của C so B lại tăng nhẹ trở lại). Đây là bài học quan trọng: **đừng ngoại suy
   xu hướng từ 1 đoạn ngắn của đường cong** — kết luận "mồi âm tiết chỉ giúp hội tụ nhanh
   giai đoạn đầu" (rút ra sau lần 5.000 bước) là **quá vội, không đúng** khi nhìn đủ xa.
   Với dữ liệu hiện có, đọc đúng hơn là: Arm C duy trì lợi thế ~4-8% so Arm B khá bền
   vững, dao động chứ không hội tụ về 0.

## Phát hiện: `entropy_threshold` bị hiệu chỉnh sai trong TOÀN BỘ các lần chạy 500/5.000/20.000 bước ở trên (đã sửa)
Sau khi có kết luận sơ bộ ở trên, đi tìm hướng cải thiện BLT thì phát hiện ra lỗi hiệu
chỉnh: `entropy_threshold=3.7` (dùng xuyên suốt mọi lần chạy Arm B/C từ đầu tới giờ, kể cả
3 lần scale-up 500/5.000/20.000 bước) **được chọn khi entropy model CHƯA train đủ** (chỉ
~100-150 bước ad-hoc lúc kiểm tra thủ công ban đầu). Thực tế mọi lần train thật đều dùng
`entropy_pretrain_steps=500` (override qua CLI) — dựng công cụ đo lại có hệ thống
(`experiments/pillar1_patch_encoder/sweep_entropy_threshold.py`, quét 11 giá trị threshold,
đo trên 8 cửa sổ held-out sau khi entropy model đã train đủ 500 bước) thì thấy:

| threshold | avg patch length (byte) |
|---|---|
| 1.5 | ~1.4 |
| 3.0 | **4.33 (gần khớp mục tiêu 5.06)** |
| **3.7 (giá trị đã dùng)** | **~14.89 (gần chạm trần max_patch_len=16)** |
| 4.5 | ~16.0 (chạm trần hoàn toàn) |

**Ý nghĩa:** ở threshold=3.7, một khi entropy model học đủ tốt, entropy hiếm khi vượt
ngưỡng — nghĩa là patch gần như luôn bị cắt bởi giới hạn cứng `max_patch_len=16`, không
phải bởi tín hiệu entropy nữa. Nói cách khác: **toàn bộ kết quả Arm B/C ở 500/5.000/20.000
bước phía trên thực chất đang so sánh BPE với một BLT gần-như-cắt-cố-định-16-byte, không
phải BLT "tự quyết định điểm cắt" như dự định ban đầu.** Điều này KHÔNG làm sai hướng kết
luận chính (BPE vẫn thắng, khoảng cách vẫn giãn ra theo compute — hai điều này không phụ
thuộc vào patch có "động" thật hay không), nhưng có nghĩa là: (a) so sánh Arm C vs Arm B
(mồi âm tiết vs entropy thuần) có thể chưa công bằng, vì "entropy thuần" ở đây gần như
không thật sự phát huy tác dụng; (b) mọi diễn giải kiểu "model tự học cắt theo entropy" ở
các mục trên (khi giải thích cho user) mô tả đúng THIẾT KẾ nhưng KHÔNG đúng HÀNH VI THỰC TẾ
đã xảy ra trong các lần train đó.

**Đã sửa:** `entropy_threshold: 3.7 → 3.0` trong `1_3_arm_B_blt.yaml`, `1_3_arm_C_blt_syllable.yaml`,
`1_3_arm_B2_cross_attention.yaml` (threshold=3.0 cho avg patch ~4.33 byte, khớp mục tiêu
~5.06 byte/patch = bytes/token thật của Arm A). Đo lại compute-matching với threshold mới
(`measure_compute_match.py`) — Arm B vẫn trong ngưỡng (0.80x Arm A), Arm B2 cần tăng
`d_model` (176→200, xem `1_3_arm_B2_cross_attention.yaml`) để về lại 0.97x, Arm D không đổi
(dùng `bpe_guided`, không phụ thuộc `entropy_threshold`). Đã chạy lại regression 10 bước
local cho cả 4 config (B/B2/C/D) — không lỗi. **Chưa train lại 500/5.000/20.000 bước thật
trên Kaggle với threshold đã sửa** — cần làm việc này trước khi coi kết luận 1.3 là chắc
chắn (patch "động" thật sự có thể cho kết quả khác so với "cắt cố định 16 byte" ở trên).

## Hướng cải thiện BLT sau khi có kết luận 1.3 sơ bộ (đã setup, CHƯA train)
Sau khi thấy BPE thắng rõ ở quy mô hiện tại, tìm hướng cải thiện byte-level thay vì bỏ hẳn
— 2 hướng độc lập, mỗi hướng đổi ĐÚNG 1 biến so với Arm B để cô lập tác dụng riêng:

- **Arm B2 — local encoder dùng cross-attention thay vì mean-pool**
  (`1_3_arm_B2_cross_attention.yaml`, `compare_against: 1_3_arm_B_blt`). Đây là thiết kế
  ĐÚNG theo paper BLT gốc (Pagnoni et al.) — bản rút gọn ban đầu ở đây dùng mean-pool đơn
  giản hơn. Thêm `CrossAttention`/`CrossAttentionBlock` (`vislm/backbones/modules/transformer_block.py`)
  và nhánh `encoder_type="cross_attention"` trong `BLTLanguageModel._encode_patches()`
  (`vislm/backbones/models/blt_lm.py`). Đã verify: forward/backward chạy đúng, regression
  local pass, compute-matched 0.97x Arm A ở `d_model=200`.
- **Arm D — dùng ranh giới của BPE tokenizer thật, KHÔNG dùng vocab/embedding của nó**
  (`1_3_arm_D_bpe_guided.yaml`, `compare_against: [1_3_arm_A_bpe, 1_3_arm_B_blt]`). Lấy ý
  tưởng "mượn cắt cụm, không mượn từ vựng" từ paper Super Tiny Language Models
  (arXiv:2405.14159): vẫn byte-level (vocab=256), nhưng dùng chính tokenizer PhoGPT-4B của
  Arm A (`return_offsets_mapping`) để tìm điểm cắt patch thay vì entropy model tự học —
  `vislm/backbones/modules/patching.py::bpe_guided_patch_lengths()`. Lợi ích phụ: bỏ hẳn
  bước `entropy_pretrain_steps` (không cần entropy model), tiết kiệm compute thật —
  `num_latent_params == num_params`, đã verify local. Đã verify: forward/backward đúng,
  regression local pass, compute-matched 0.97x Arm A ở `d_model=224`.

**Đã train thật debug-scale (30 bước / 50MB) trên Kaggle GPU (P100) cho cả 4 config** — xem
`runs/2026_09_20_train_arm_{b,c}_debug_kaggle_gpu_threshold_fix/` (Arm B/C sau khi sửa
threshold) và `runs/2026_09_20_train_arm_{b2,d}_debug_kaggle_gpu/` (2 hướng cải thiện mới).
Cả 4 chạy sạch trên `device=cuda`, không lỗi, loss giảm đều:

| | Arm B (sửa) | Arm C (sửa) | Arm B2 (cross-attn) | Arm D (bpe-guided) |
|---|---|---|---|---|
| params | 4.55M | 4.55M | 5.82M | 4.98M |
| loss đầu → cuối | 5.608 → 4.447 | 5.620 → 4.295 | 5.518 → 4.037 | 5.563 → 4.082 |

Arm D xác nhận đúng thiết kế trên GPU thật: `latent_params == params` (không có entropy
model, không tốn compute pretrain nó). Đây là smoke-test 30 bước — đủ để xác nhận pipeline
chạy đúng đầu-cuối trên GPU thật, chưa đủ để so sánh có ý nghĩa — xem ngay mục dưới cho kết
quả 500 bước.

## So sánh 500 bước / 0.2GB — Arm B/C sau khi sửa threshold, cộng 2 hướng cải thiện mới
Cùng quy mô với lần "so sánh công bằng 3 arm" gốc (`max_seq_len=512, batch_size=8`,
`dataset_target_gb=0.2`, 500 bước) — Arm B/C dùng `entropy_threshold=3.0` (đã sửa) +
`entropy_pretrain_steps=500` (tăng từ 100 trong lần chạy gốc, để khớp giá trị đã dùng hiệu
chỉnh threshold), Arm B2/D chạy lần đầu ở quy mô này. Arm A không đổi gì nên dùng lại kết
quả cũ (`runs/2026_09_13_train_arm_a_500_kaggle_gpu/`). Xem
`runs/2026_09_20_train_arm_{b,c}_500_kaggle_gpu_threshold_fix/`,
`runs/2026_09_20_train_arm_{b2,d}_500_kaggle_gpu/`.

| | A (BPE) | B (entropy, sửa) | C (entropy+âm tiết, sửa) | B2 (cross-attn) | D (bpe-guided) |
|---|---|---|---|---|---|
| bits/byte (TB 20 bước cuối) | **2.054** | 3.453 | 3.367 | 3.343 | **3.338** |
| bits/byte thấp nhất | **1.858** | 2.956 | 2.837 | 2.778 | 2.850 |

**So với lần chạy TRƯỚC KHI sửa threshold** (B: 3.583→**3.453**, C: 3.375→**3.367**):
1. **Arm B cải thiện rõ rệt sau khi sửa threshold** (bits/byte 3.583 → 3.453, ~3.6% tốt
   hơn) — hợp lý: patch giờ mới thật sự "động" theo entropy như thiết kế, thay vì gần như
   cắt cố định mỗi 16 byte như trước.
2. **Arm C gần như không đổi** (3.375 → 3.367) — vì ranh giới âm tiết được CỘNG THÊM vào
   ranh giới entropy (không thay thế), Arm C vốn ít phụ thuộc vào việc entropy model có
   hoạt động đúng hay không.
3. **Hệ quả: khoảng cách C so B THU HẸP đáng kể** (C tốt hơn B ~5.8% trước khi sửa → chỉ
   còn ~2.5% sau khi sửa). Đọc đúng hơn: phần lớn lợi thế "mồi âm tiết" quan sát được trước
   đây thực ra là do **Arm B bị hỏng bởi lỗi threshold**, không phải do mồi âm tiết tự nó
   mạnh đến vậy. Mồi âm tiết vẫn có lợi thế thật, nhưng nhỏ hơn nhiều so với ước tính trước.
4. **Cả 2 hướng cải thiện đều thắng cả Arm B lẫn Arm C** — Arm D (bpe-guided boundaries,
   3.338 bpb) và Arm B2 (cross-attention, 3.343 bpb) gần như ngang nhau, đều tốt hơn Arm C
   (3.367) lẫn Arm B (3.453). Arm D còn có lợi thế phụ: không tốn compute pretrain entropy
   model. Đây là tín hiệu tốt cho hướng cải thiện BLT, dù cách biệt với Arm A (2.054) vẫn
   còn rất lớn ở quy mô này.

## Trạng thái
Cả 3 arm gốc (A, B, C) đã viết xong, compute-matched, decoder Arm B/C đã tối ưu tốc độ
(~6.4x ở scale debug), có checkpoint/resume/LR-schedule/eval, và đã train thật ở nhiều quy
mô (30, 500, 5.000, 20.000 bước) trên Kaggle GPU. `entropy_threshold` từng bị hiệu chỉnh
sai (3.7, gây patch gần-cố-định) đã được sửa (3.0) và **đã rerun ở cả 2 quy mô 30 và 500
bước để xác nhận** — số liệu 5.000/20.000-bước ở các mục trên vẫn giữ nguyên như đã chạy
(chưa rerun với threshold mới, xem "Còn thiếu" bên dưới). 2 hướng cải thiện mới (Arm B2 —
cross-attention, Arm D — bpe-guided boundaries) đã train thật ở 30 và 500 bước, cả 2 đều
cho kết quả tốt hơn Arm B/C ở quy mô 500 bước (xem mục ngay trên).

**Kết luận cho câu hỏi con 1.3 ở phạm vi đã thử nghiệm:**
- **BPE (Arm A) vẫn vượt trội byte-level (mọi biến thể B/C/B2/D) rất nhiều ở quy mô nhỏ
  này** — kết luận này KHÔNG phụ thuộc vào lỗi threshold (cắt cố định hay cắt động thì BPE
  vẫn thắng đậm). Khuyến nghị dùng BPE ở quy mô hiện tại, byte-level chỉ đáng đầu tư tiếp
  nếu có compute để scale MODEL SIZE.
- **"Mồi âm tiết tốt hơn entropy thuần" — vẫn đúng nhưng lợi thế nhỏ hơn nhiều so với ước
  tính trước khi sửa threshold** (2.5% chứ không phải 5.8%) — phần lớn chênh lệch cũ là do
  Arm B bị lỗi cấu hình, không phải do bản thân ý tưởng mồi âm tiết yếu.
- **2 hướng cải thiện mới (cross-attention encoder, bpe-guided boundaries) đều có tác dụng
  thật, gần tương đương nhau, và đều tốt hơn cách làm gốc (entropy thuần lẫn mồi âm tiết)**
  ở quy mô 500 bước — bằng chứng ban đầu ủng hộ cả 2 hướng nghiên cứu là đáng đầu tư tiếp
  nếu muốn thu hẹp khoảng cách với BPE.

Còn thiếu nếu muốn kết luận chắc chắn hơn nữa cho 1.3: (1) rerun 5.000/20.000-bước cho Arm
B/C với threshold đã sửa (mới có 30/500 bước) để xem xu hướng "giãn ra" so Arm A và "thu
hẹp" so C/B có còn đúng ở scale dài hơn không; (2) scale Arm B2/D lên 5.000+ bước để xem
lợi thế của chúng có bền vững không; (3) thử tăng MODEL SIZE cho Arm B/C/B2/D (giữ
compute-matched) thay vì chỉ tăng số bước; (4) thử LR/warmup riêng cho BLT thay vì dùng
chung với Arm A; (5) nhiều seed hơn để chắc chắn xu hướng không phải nhiễu (mới có 1 seed
cho mỗi scale); (6) áp dụng cách detect CODE_DIR động cho các notebook Kaggle cũ hơn còn
sót lại (train_arm_{a,b,c}_debug gốc — 2 cái B/C debug đã sửa, data_pipeline_smoke_test)
nếu cần rerun. 2.1/2.2 (Trụ cột 2 — backbone SSM) chưa viết code.
