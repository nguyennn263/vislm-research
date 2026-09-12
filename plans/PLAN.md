# Kế hoạch nghiên cứu: SLM tiếng Việt (nhỏ – nhanh – nhẹ – chính xác)

## Cách đọc tài liệu này

Mỗi mục theo cấu trúc: **Câu hỏi nghiên cứu (RQ)** → **Câu hỏi con** → **Giả thuyết** → **Thí nghiệm kiểm chứng** → **Bước cụ thể** → **Tiêu chí đi tiếp / dừng lại (go–no-go)**.

Không có mốc thời gian. Thứ tự thực hiện dựa trên **phụ thuộc logic** (cái gì phải xong trước cái gì), không phải lịch. Xem sơ đồ phụ thuộc ở cuối file.

Mỗi trụ cột có thêm mục **"Tài liệu &amp; mã nguồn tham khảo"** — ưu tiên bài báo mới (2024–2026), tác giả/phòng lab uy tín (Meta FAIR, Google DeepMind, Microsoft, AI21, DeepSeek, Princeton NLP…), và hội nghị top-tier (NeurIPS, ICML, ICLR, ACL/EMNLP) hoặc tạp chí Nature khi có. Link GitHub đi kèm khi tác giả công bố code chính thức; nếu không có code chính thức, đề ghi rõ "chưa công bố code chính thức" thay vì đoán một repo không xác thực.

---

## Mức độ ưu tiên (cập nhật sau vòng review đầu tiên)

- **Tier 1 — ưu tiên cao nhất, chẻ nhỏ nhất, bắt đầu ngay:** Trụ cột 1 (patch encoder) và Trụ cột 2 (backbone). Đây là 2 trụ cột có tiềm năng đóng góp/nghiên cứu cao nhất, đồng thời phần lớn thí nghiệm ở đây rẻ (không cần train full model — xem phần chẻ nhỏ bên dưới).
- **Tier 1 hỗ trợ, đã đúng hướng, giữ nguyên:** Trụ cột 0 (định nghĩa mục tiêu) và Trụ cột 5 (RAG). Không cần chẻ thêm nhiều vì bản chất câu hỏi đã khá atomic và không phụ thuộc backbone.
- **Tier 2 — hợp lý về mặt kỹ thuật nhưng nên làm SAU Tier 1:** Trụ cột 3 (MoE-LoRA), Trụ cột 4 (Alignment), Trụ cột 6 (OOD gate). Lý do defer: cả 3 đều cần một checkpoint backbone đã ổn định để thí nghiệm có ý nghĩa — chạy sớm chỉ tốn công làm lại.
- **Tier 3 — optional, không nằm trong lộ trình lõi hiện tại:** Trụ cột 7 (Router/ensemble hệ thống). Hiện đang tham vọng hơn mức cần thiết cho giai đoạn này; chỉ quay lại nếu Tier 1–2 đã ổn định và có nhu cầu sản phẩm cụ thể đòi hỏi nó.

## Nguyên tắc thiết kế thí nghiệm: cô lập biến số

Nhận xét quan trọng: RQ tổng quát ở dưới gộp 7 innovation vào cùng một câu hỏi. Nếu kiểm định bằng cách huấn luyện thẳng 1 model có đủ cả 7 thứ rồi so với baseline, kết quả tốt hơn (nếu có) **không nói lên được cải thiện đến từ đâu** — đây là vấn đề nhiễu biến (confounding) kinh điển trong thiết kế thí nghiệm. Để tránh lặp lại lỗi này ở từng trụ cột:

1. **Không bao giờ kiểm định tổ hợp trước khi từng thành phần đã được kiểm định độc lập.** Mỗi thí nghiệm chỉ được thay đổi ĐÚNG MỘT biến so với baseline (single-variable-at-a-time) — kiến trúc, dữ liệu, compute budget, seed, số bước train phải giữ nguyên ở các arm còn lại. Đây chính là cách các bài báo được trích dẫn trong file này thiết kế thí nghiệm của họ (BLT so sánh compute-matched với BPE; Mamba-2 so sánh compute-matched với Transformer…).
2. **So sánh phải compute-matched, không chỉ "kiến trúc mới thắng."** Ví dụ ở Trụ cột 2: nếu một arm có nhiều lớp attention hơn arm kia, phải giữ tổng tham số/FLOPs bằng nhau — nếu không, không thể phân biệt "kiến trúc lai tốt hơn" với "đơn giản là nhiều tham số hơn."
3. **Khi ghép 2 thành phần đã validate riêng, phải đo hiệu ứng tương tác (interaction effect) bằng thiết kế tối thiểu 4 arm: baseline, chỉ A, chỉ B, A+B.** Không mặc định lợi ích cộng gộp — A+B có thể trung hoà nhau hoặc tệ hơn cả 2 thứ riêng lẻ.
4. **Trước khi mở rộng một kỹ thuật đã công bố sang tiếng Việt, tái lập kết quả gốc trên đúng dữ liệu/ngôn ngữ tác giả gốc dùng trước.** Việc này tách bạch 2 khả năng thất bại khác nhau: "pipeline của mình bị lỗi khi cài lại" và "kỹ thuật không hợp với đặc thù tiếng Việt." (Áp dụng cụ thể ở câu hỏi con 2.2a bên dưới.)
5. **Master RQ ở mục kế tiếp không được kiểm định trực tiếp bằng một model tổng hợp cả 7 trụ cột.** Nó chỉ là định hướng dài hạn — việc tổ hợp chỉ tiến hành sau khi Tier 1 (Trụ cột 0, 1, 2, 5) đã có kết luận độc lập rõ ràng, và mỗi bước tổ hợp phải qua nguyên tắc 3 ở trên.

---

## RQ0 — Câu hỏi nghiên cứu tổng quát (Master Research Question)

> *"Có thể xây một mô hình ngôn ngữ tiếng Việt (SLM) đạt đồng thời 4 mục tiêu — kích thước nhỏ, tốc độ suy luận nhanh, chi phí vận hành thấp, độ chính xác cao trên các tác vụ mục tiêu — bằng cách kết hợp có chọn lọc các kiến trúc: patch encoder không tokenizer, backbone lai SSM–Attention, MoE-LoRA theo miền, alignment tối ưu hoá sở thích, RAG thay cho chỉnh sửa trọng số, và cổng phát hiện out-of-distribution?"*

Đây là câu hỏi **chưa thể trả lời trực tiếp và KHÔNG được kiểm định bằng cách huấn luyện thẳng 1 model gộp cả 7 thứ** (xem "Nguyên tắc thiết kế thí nghiệm" ở trên) — nó phải được chẻ thành 8 nhóm câu hỏi con (Trụ cột 0–7), mỗi nhóm kiểm định độc lập trước, việc tổ hợp là bước riêng và sau cùng. Mỗi trụ cột là một dòng nghiên cứu độc lập có thể thành công hoặc thất bại riêng; kết quả chung phụ thuộc vào việc bao nhiêu trụ cột "sống sót" qua kiểm chứng.

**Tài liệu nền cho toàn bộ kế hoạch (các mô hình SLM tham chiếu để so sánh):**

- *Phi-4 Technical Report* — Abdin et al., Microsoft Research, 2024. [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
- *Gemma 3 Technical Report* — Gemma Team, Google DeepMind, 2025. [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) · [model card](https://ai.google.dev/gemma/docs/core/model_card_3)
- *Qwen3 Technical Report* — Yang et al., Alibaba Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) · GitHub: [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)

---

## Trụ cột 0 — Định nghĩa mục tiêu (phải làm TRƯỚC mọi thứ khác)

**RQ0:** *"Chính xác 99%" đo bằng gì, "nhỏ – nhanh – nhẹ" định lượng ra sao, và tác vụ/miền đầu tiên để làm proof-of-concept là gì?*

Đây không phải câu hỏi kỹ thuật — nó là câu hỏi **đo lường**. Nếu bỏ qua bước này, mọi thí nghiệm phía sau đều vô nghĩa vì không có gì để so sánh.

- **Câu hỏi con 0.1:** Benchmark nào là trọng tài? (VMLU cho tổng quát, nhưng cần thêm 1 benchmark tự tạo cho miền hẹp mà Viwixtech chọn — vì VMLU không đo được nghiệp vụ cụ thể).
  - *Bước cụ thể:* Chọn 1 miền hẹp (ví dụ CSKH, giáo dục — theo phần phân khúc trong báo cáo thị trường). Tự soạn 300–500 câu hỏi + đáp án chuẩn cho miền đó, có người trong ngành duyệt lại.
  - *Tiêu chí dừng:* Nếu không tìm được domain expert để duyệt bộ câu hỏi → chưa nên chọn miền đó làm PoC.
- **Câu hỏi con 0.2:** "Nhỏ/nhanh/nhẹ" là con số nào? (VD: ≤ 4B tham số, latency ≤ 300ms/câu trên 1 GPU phổ thông, chạy được trên máy chủ tầm trung không cần cluster).
  - *Bước cụ thể:* Viết ra 1 "spec sheet" 1 trang: số tham số tối đa, RAM tối đa, latency tối đa, throughput tối thiểu. Đây là ràng buộc cứng cho mọi thí nghiệm sau.
- **Câu hỏi con 0.3:** Ngưỡng "từ chối trả lời" chấp nhận được là bao nhiêu? (Model từ chối 15% câu hỏi nhưng đúng 99% câu còn lại có tốt hơn trả lời 100% câu với độ chính xác 90% không? — tuỳ use-case).

**Kết quả đầu ra bắt buộc trước khi sang Trụ cột 1:** 1 văn bản ngắn định nghĩa rõ 3 câu hỏi con trên. Không có văn bản này thì các thí nghiệm ở dưới không có đường base để so sánh.

**Tài liệu &amp; mã nguồn tham khảo:**

- *VMLU Benchmarks: A Comprehensive Benchmark Toolkit for Vietnamese LLMs* — Bui et al., **ACL 2025** (main). [ACL Anthology 2025.acl-long.563](https://aclanthology.org/2025.acl-long.563/) · GitHub: [ZaloAI-Jaist/VMLU](https://github.com/ZaloAI-Jaist/VMLU) — 10.880 câu hỏi trắc nghiệm, 58 chủ đề, dùng làm trọng tài benchmark tổng quát.
- *VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark* (2025) — [arXiv:2508.13680](https://arxiv.org/abs/2508.13680) — tham khảo nếu sau này mở rộng đa phương thức.

---

## Trụ cột 1 — Patch encoder / biểu diễn đầu vào (thay tokenizer)

**RQ1:** *Một patch encoder byte-level có "mồi" bằng ranh giới âm tiết tiếng Việt có vượt trội hơn BPE truyền thống về (a) hiệu quả nén, (b) độ chính xác trên từ hiếm/lỗi chính tả, (c) tốc độ suy luận?*

- **Câu hỏi con 1.1 — chia thành 4 thí nghiệm độc lập, không cần train, chạy trong vài giờ:** BPE tiếng Việt hiện tại thất bại cụ thể ở đâu?
  - *1.1a — Từ ghép hiếm:* Lấy tokenizer có sẵn (PhoGPT/ViGPT), chạy qua danh sách từ ghép tần suất thấp → đo tỷ lệ token/âm tiết.
  - *1.1b — Tên riêng:* Cùng tokenizer, chạy qua danh sách tên người/địa danh tiếng Việt không phổ biến → đo tỷ lệ tách vỡ.
  - *1.1c — Số liệu:* Chạy qua chuỗi số (ngày tháng, tiền tệ, mã số) → đo tính nhất quán của cách tách.
  - *1.1d — Lỗi chính tả:* Lấy 1 đoạn văn sạch, tạo bản sao có lỗi chính tả/dấu cố ý → so tỷ lệ token giữa 2 bản.
  - *Vì sao tách 4 thí nghiệm riêng thay vì gộp 1 tập test:* mỗi loại lỗi có cơ chế khác nhau (từ vựng hiếm vs. cấu trúc số vs. nhiễu bề mặt) — gộp chung dễ che lấp nguồn gốc cụ thể, làm bước 1.3 sau khó biết nên ưu tiên giải quyết loại lỗi nào.
  - *Tiêu chí đi tiếp (áp dụng riêng từng thí nghiệm):* Nếu tỷ lệ vỡ token tăng &gt; 2 lần so với văn bản "sạch, phổ biến" → xác nhận đúng vấn đề ở khía cạnh đó.
- **Câu hỏi con 1.2 — tách thành 2 lượt đo để kiểm soát yếu tố "chọn model đo entropy":** Ranh giới âm tiết có trùng với ranh giới patch tối ưu theo entropy không?
  - *1.2a:* Dùng 1 model đa ngôn ngữ tổng quát, không train riêng cho tiếng Việt (VD mGPT/XGLM), tính entropy dự đoán byte-tiếp-theo → so khớp đỉnh entropy với ranh giới âm tiết thực tế.
  - *1.2b:* Lặp lại đúng phép đo với 1 model đã pretrain riêng cho tiếng Việt (VD PhoGPT) → so kết quả với 1.2a.
  - *Vì sao cần 2 lượt:* nếu chỉ đo bằng 1 model, không biết độ trùng khớp phản ánh đặc tính của **ngôn ngữ tiếng Việt** hay chỉ là đặc tính riêng của **model dùng để đo**. Hai lượt cho kết quả gần nhau → kết luận về ngôn ngữ đáng tin hơn.
  - *Tiêu chí đi tiếp:* Cả 2 lượt đều cho độ trùng khớp &gt; 70% → có cơ sở "mồi" patch encoder bằng âm tiết.
- **Câu hỏi con 1.3 — thiết kế lại thành bậc thang 3 arm cô lập biến số (không so sánh gộp):**
  - **Biến số cố định giữa mọi arm:** cùng tập dữ liệu, cùng tổng compute (FLOPs), cùng số tham số latent transformer, cùng seed — chỉ đổi đúng 1 thứ mỗi bước.
  - *Arm A (baseline):* BPE + Transformer chuẩn, không đổi gì.
  - *Arm B:* Thay tokenizer bằng BLT thuần entropy (đúng kiến trúc gốc), giữ nguyên phần còn lại → cô lập riêng hiệu ứng "chuyển sang byte-level patching," tách khỏi ý tưởng âm tiết.
  - *Arm C:* BLT + khởi tạo ranh giới patch bằng âm tiết (chỉ thêm đúng 1 thay đổi so với Arm B) → cô lập riêng hiệu ứng "mồi âm tiết," tách khỏi hiệu ứng byte-level nói chung.
  - *Cách đọc kết quả:* B so A cho biết byte-level patching có ích gì không (bất kể ý tưởng âm tiết); C so B cho biết riêng phần "mồi âm tiết" có cộng thêm giá trị ngoài B hay không. **Không kết luận "ý tưởng âm tiết đúng" chỉ từ so sánh C với A** — đó là lỗi nhiễu biến kinh điển.
  - *Tiêu chí thành công cho C:* C hội tụ nhanh hơn B với cùng compute, VÀ B đã chứng minh tốt hơn A trước đó — nếu B không hơn A, việc C hơn A không chứng minh được gì về riêng ý tưởng âm tiết.
  - *Tiêu chí dừng/pivot:* Nếu C không khác biệt đáng kể so với B (dù B đã hơn A) → bỏ phần "mồi âm tiết," dùng thẳng BLT gốc (Arm B).

**Tài liệu &amp; mã nguồn tham khảo:**

- *Byte Latent Transformer: Patches Scale Better Than Tokens* — Pagnoni et al., Meta FAIR, **ACL 2025**. [arXiv:2412.09871](https://arxiv.org/abs/2412.09871) · GitHub chính thức: [facebookresearch/blt](https://github.com/facebookresearch/blt) — bài gốc cho toàn bộ ý tưởng entropy-based dynamic patching.
- *Multiscale Byte Language Models — A Hierarchical Architecture for Causal Million-Length Sequence Modeling* (2025), tổng hợp MegaByte và MambaByte — [arXiv:2502.14553](https://arxiv.org/abs/2502.14553) — nền tảng lý thuyết cho patch/byte hierarchical modeling.
- *KazByte: Adapting Qwen models to Kazakh via Byte-level Adapter* (2025/2026) — [arXiv:2603.27859](https://arxiv.org/abs/2603.27859) — ca tương tự: dùng byte-level adapter để thích nghi 1 LLM có sẵn sang ngôn ngữ ít tài nguyên, mô hình rất gần với việc Viwixtech muốn làm cho tiếng Việt.
- *BamiBERT: A New BERT-based Language Model for Vietnamese* (2025) — [arXiv:2607.02259](https://arxiv.org/abs/2607.02259) — huấn luyện trực tiếp trên raw text tiếng Việt, bỏ phụ thuộc word segmentation ngoài, bằng chứng thực nghiệm gần nhất cho câu hỏi con 1.1–1.2.
- *VSEC: Transformer-based Model for Vietnamese Spelling Correction* (2021) — [arXiv:2111.00640](https://arxiv.org/abs/2111.00640) — phân tích trực tiếp đánh đổi syllable/subword/character level cho tiếng Việt, dùng làm cơ sở lý luận cho câu hỏi con 1.1.

---

## Trụ cột 2 — Backbone: bộ nhớ dài hạn (SSM lai Attention)

**RQ2:** *Kiến trúc lai SSM–Attention có giữ được khả năng nhớ chính xác (associative recall) tốt hơn SSM thuần, trong khi rẻ hơn Transformer thuần, khi áp dụng cho văn bản tiếng Việt dài?*

- **Câu hỏi con 2.1 — tách thành 2 thí nghiệm để không nhầm "tỷ lệ kiến trúc" với "nhiều tham số hơn" hoặc "positional encoding khác nhau":**
  - *2.1a — Compute-matched ratio sweep:* Tạo bộ test needle-in-haystack tiếng Việt (nhiều độ dài khác nhau) → chạy qua vài tỷ lệ SSM:Attention, nhưng **giữ tổng tham số và FLOPs bằng nhau giữa các tỷ lệ** (tăng width SSM để bù khi giảm số lớp attention) → đo % trả lời đúng theo độ dài.
  - *Vì sao phải compute-matched:* Nếu chỉ thêm lớp attention mà không bớt gì ở lớp khác, tỷ lệ có nhiều attention hơn tự động có nhiều tham số hơn — không thể tách "tỷ lệ tối ưu" khỏi "đơn giản là model lớn hơn."
  - *2.1b — Kiểm tra riêng biến số positional encoding:* Vì Mamba không cần RoPE (theo chính bài Jamba đã trích ở trên), cần xác nhận riêng: khác biệt recall giữa các arm ở 2.1a có bị lẫn với việc arm nhiều attention hơn cũng đang dùng RoPE mạnh hơn không? Chạy thêm 1 cặp đối chứng: cùng tỷ lệ SSM:Attention, chỉ đổi có/không RoPE ở phần attention → xem recall đổi bao nhiêu do riêng yếu tố này.
  - *Tiêu chí đi tiếp:* Chọn tỷ lệ giữ được &gt; 90% độ chính xác recall ở độ dài mục tiêu (theo spec Trụ cột 0) với chi phí bộ nhớ thấp nhất, SAU KHI đã loại trừ 2 yếu tố nhiễu ở trên.
- **Câu hỏi con 2.2 — tách thành bước "tái lập trước" và "mở rộng sau" để không nhầm lỗi pipeline với lỗi giả thuyết:**
  - *Đây là chỗ áp dụng đúng tinh thần ý tưởng gốc của bạn (assemble không train lại từ đầu) — nhưng ở mức kiến trúc backbone, không phải mức ensemble nhiều model hoàn chỉnh.*
  - *2.2a — Tái lập trên baseline đã biết trước (sanity check bắt buộc):* Trước khi thử trên tiếng Việt, chạy đúng quy trình chưng cất Transformer→Mamba theo 1 recipe đã công bố (MOHAWK hoặc MambaInLlama, xem tài liệu tham khảo bên dưới) trên đúng dữ liệu/model mà tác giả gốc dùng → xác nhận tái lập được con số tương tự họ công bố.
  - *Vì sao bước này bắt buộc:* nếu bỏ qua và chạy thẳng trên tiếng Việt rồi thất bại, sẽ không biết là do "kỹ thuật không hợp với tiếng Việt" hay "cài đặt lại pipeline bị sai" — 2 chẩn đoán này dẫn tới 2 hướng xử lý hoàn toàn khác nhau.
  - *2.2b — Áp dụng sang checkpoint tiếng Việt:* Chỉ sau khi 2.2a đạt kết quả tương đương công bố gốc, mới lặp lại đúng quy trình đó trên 1 checkpoint Transformer tiếng Việt có sẵn → so đường cong phục hồi hiệu năng (theo số bước fine-tune) với đường cong ở 2.2a.
  - *Tiêu chí đi tiếp:* Nếu đường cong phục hồi ở 2.2b gần với 2.2a (chỉ cần &lt; 10% compute so với train from-scratch để đạt hiệu năng tương đương) → đây là con đường chính, tiết kiệm rất nhiều cho một đội nhỏ.
  - *Tiêu chí dừng/pivot:* Nếu 2.2a thất bại → dừng lại sửa pipeline trước, chưa kết luận gì về tiếng Việt. Nếu 2.2a thành công nhưng 2.2b thất bại rõ rệt (loss không hội tụ dù cùng pipeline đã tái lập được) → đây mới là bằng chứng thật về khó khăn riêng của tiếng Việt, quay lại train from-scratch quy mô nhỏ.

**Tài liệu &amp; mã nguồn tham khảo:**

- *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu &amp; Dao, 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) · GitHub chính thức: [state-spaces/mamba](https://github.com/state-spaces/mamba).
- *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2) — Dao &amp; Gu, **ICML 2024**. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) — cùng repo trên.
- *Jamba: A Hybrid Transformer-Mamba Language Model* — AI21 Labs, 2024, xuất hiện tại **ICLR 2025**. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) · [Jamba-1.5, arXiv:2408.12570](https://arxiv.org/abs/2408.12570) — nguồn trực tiếp cho câu hỏi con 2.1 (tỷ lệ SSM:Attention, KV-cache giảm 8 lần).
- *Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling* (2025) — [arXiv:2510.26912](https://arxiv.org/abs/2510.26912) — bàn trực tiếp về associative recall trong kiến trúc lai, rất sát câu hỏi con 2.1.
- *Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models* (MOHAWK) — Bick et al., **NeurIPS 2024**. [arXiv:2408.10189](https://arxiv.org/abs/2408.10189) · GitHub: [goombalab/phi-mamba](https://github.com/goombalab/phi-mamba) — bằng chứng trực tiếp cho câu hỏi con 2.2: chưng cất Phi-1.5 (Transformer) thành Mamba chỉ với 3B token, không train from-scratch.
- *The Mamba in the Llama: Distilling and Accelerating Hybrid Models* — **NeurIPS 2024**. GitHub: [jxiw/MambaInLlama](https://github.com/jxiw/MambaInLlama) — quy trình chưng cất từng bước (thay từng lớp attention bằng Mamba2) rất phù hợp để thử nghiệm nhanh câu hỏi con 2.2 trên phần cứng hạn chế (8×A100 theo tác giả).
- *Attention to Mamba: A Recipe for Cross-Architecture Distillation* — Apple, 2026. [arXiv:2604.14191](https://arxiv.org/abs/2604.14191) — recipe 2 giai đoạn mới hơn, tránh sụp hiệu năng khi chưng cất trực tiếp.

---

## Trụ cột 3 — MoE-LoRA theo miền (không train lại toàn bộ)

**RQ3:** *Có thể thêm năng lực chuyên môn theo miền (y tế, tài chính, pháp lý…) chỉ bằng LoRA expert + router, mà không làm suy giảm năng lực tổng quát và không train lại toàn bộ model?*

*(Phụ thuộc: cần Trụ cột 2 đã chốt kiến trúc backbone, vì LoRA expert gắn vào lớp FFN cụ thể của backbone đó.)*

- **Câu hỏi con 3.1:** Cần tối thiểu bao nhiêu dữ liệu domain để 1 LoRA expert đạt hiệu năng chấp nhận được?
  - *Thí nghiệm:* Train LoRA expert cho 1 miền với các mức dữ liệu tăng dần (1k, 5k, 20k, 50k mẫu) → vẽ đường cong hiệu năng theo lượng dữ liệu → tìm điểm bão hoà.
- **Câu hỏi con 3.2:** Thêm expert thứ 2 có làm suy giảm expert thứ 1 hoặc năng lực tổng quát không (catastrophic forgetting)?
  - *Thí nghiệm:* Train expert A → đo benchmark tổng quát + benchmark A. Train thêm expert B → đo lại benchmark tổng quát + A + B.
  - *Tiêu chí đi tiếp:* Benchmark tổng quát và expert A không giảm quá X% (định trước ở Trụ cột 0) sau khi thêm B.
  - *Tiêu chí dừng/pivot:* Nếu suy giảm vượt ngưỡng → cần "Knowledge-Preservation Plugin" (đóng băng 1 nhóm base expert) như mô tả trong Med-MoE-LoRA, thử lại.
- **Câu hỏi con 3.3:** Router ở cấp token có định tuyến chính xác không, hay dễ nhầm miền khi câu hỏi lai (VD: câu hỏi vừa liên quan tài chính vừa pháp lý)?
  - *Thí nghiệm:* Tạo 1 tập câu hỏi "lai miền" cố ý → xem router chọn expert nào, đo độ chính xác cuối cùng.

**Tài liệu &amp; mã nguồn tham khảo:**

- *Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs* — Microsoft, 2025. [arXiv:2503.01743](https://arxiv.org/abs/2503.01743) — ví dụ thực tế một SLM sản xuất dùng đúng kiến trúc Mixture-of-LoRAs cho đa miền/đa phương thức.
- *D-MoLE: Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning* — **ICML 2025**. GitHub: [gcd19/D-MoLE](https://github.com/gcd19/D-MoLE) — trực tiếp trả lời câu hỏi con 3.2 (chống catastrophic forgetting khi thêm expert tuần tự), báo cáo +15–20% so với LoRA tĩnh.
- *Mixture of LoRA Experts for Continual Information Extraction with LLMs* — **EMNLP Findings 2025**. GitHub: [nju-websoft/MOLE-CIE](https://github.com/nju-websoft/MOLE-CIE) — case study gần với câu hỏi con 3.3 (router cấp token giữa nhiều tác vụ/miền).
- *Mixture of LoRA Experts* (MoLE gốc) — Wu, Huang, Wei, 2024 — bài nền cho toàn bộ dòng nghiên cứu MoLE ở trên.

---

## Trụ cột 4 — Alignment (ORPO + căn chỉnh theo embedding)

**RQ4:** *Có thể đạt chất lượng alignment tương đương RLHF/DPO nhưng với ít dữ liệu người gán nhãn hơn, bằng ORPO (gộp 1 pha) kết hợp lọc theo cụm embedding (REAL/HEAL)?*

*(Phụ thuộc: cần có 1 checkpoint base model từ Trụ cột 1–2 để bắt đầu alignment.)*

- **Câu hỏi con 4.1:** Cần tối thiểu bao nhiêu cặp preference (chosen/rejected) tiếng Việt để ORPO hội tụ ổn định?
- **Câu hỏi con 4.2:** Lọc trước bằng cụm embedding (loại bỏ cặp nhãn nhiễu trước khi train) có giảm được đáng kể công sức gán nhãn thủ công không?
  - *Thí nghiệm:* So sánh 2 tập dữ liệu cùng kích thước: (i) gán nhãn thủ công 100%, (ii) một nửa gán nhãn thủ công + một nửa lọc tự động bằng embedding clustering → so hiệu năng model sau alignment.
  - *Tiêu chí đi tiếp:* Nếu (ii) đạt ≥ 90% hiệu năng của (i) với chưa tới một nửa công sức gán nhãn → áp dụng REAL/HEAL làm chuẩn quy trình.
- **Câu hỏi con 4.3:** So sánh trực tiếp ORPO vs DPO vs chỉ-SFT trên cùng bộ test tiếng Việt — chênh lệch có đáng để đổi độ phức tạp không?

**Tài liệu &amp; mã nguồn tham khảo:**

- *ORPO: Monolithic Preference Optimization without Reference Model* — Hong, Lee, Thorne, **EMNLP 2024**. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) · Code chính thức: [xfactlab/orpo](https://github.com/xfactlab/orpo), tích hợp sẵn trong [huggingface/trl](https://github.com/huggingface/trl/blob/main/docs/source/orpo_trainer.md).
- *SimPO: Simple Preference Optimization with a Reference-Free Reward* — Meng et al., **NeurIPS 2024**. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734) · GitHub: [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO) — phương án thay thế ORPO nếu cần đơn giản hơn nữa, dùng cho câu hỏi con 4.3.
- *HEAL: Hierarchical Embedding Alignment Loss for Improved Retrieval and Representation Learning* — Bhattarai et al., ACL Workshop KnowledgeNLP **2025**. [arXiv:2412.04661](https://arxiv.org/abs/2412.04661) · GitHub: [lanl/t-elf](https://github.com/lanl/t-elf) — nguồn trực tiếp cho câu hỏi con 4.2 (lọc cặp preference bằng cụm embedding phân cấp).
- *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafailov et al., NeurIPS 2023 — bài nền DPO, dùng làm baseline so sánh ở câu hỏi con 4.3.

---

## Trụ cột 5 — RAG &amp; cập nhật tri thức (không sửa trọng số hàng loạt)

**RQ5:** *Cấu hình RAG nào (retriever, kích thước chunk, tần suất cập nhật) phù hợp cho tri thức tiếng Việt cập nhật liên tục, và khi nào cần thêm chỉnh sửa nhẹ kiểu AdaPLE thay vì chỉ dựa vào RAG?*

*(Đây là trụ cột có thể bắt đầu SỚM NHẤT, độc lập với backbone riêng — có thể thử nghiệm ngay cả trên 1 model có sẵn trong lúc chờ Trụ cột 1–2 hoàn thiện.)*

- **Câu hỏi con 5.1:** Retriever tiếng Việt (dense embedding) hiện có đủ tốt để phân biệt các đoạn văn bản gần nghĩa nhưng khác sự thật (VD: "lãi suất 2024" vs "lãi suất 2025") không?
  - *Thí nghiệm:* Xây bộ test cố ý gài các cặp đoạn văn gần giống nhau nhưng khác dữ kiện then chốt, đo tỷ lệ retriever chọn đúng.
- **Câu hỏi con 5.2:** Với loại thông tin nào thì RAG là đủ, và loại nào thực sự cần sửa trực tiếp vào model (VD: quy tắc ngữ pháp/định dạng cố định lặp lại rất nhiều lần)?
  - *Ghi chú quan trọng:* Không dùng ROME/MEMIT cho cập nhật liên tục (bằng chứng WikiBigEdit đã chỉ rõ suy giảm sau vài trăm lần sửa). Nếu thực sự cần sửa tham số, chỉ dùng cho số lượng rất nhỏ, không thường xuyên, và theo dõi chặt bằng benchmark tổng quát trước/sau mỗi lần sửa.

**Tài liệu &amp; mã nguồn tham khảo:**

- *WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs* — Thede et al., **ICML 2025**. [arXiv:2503.05683](https://arxiv.org/abs/2503.05683) · GitHub chính thức: [ExplainableML/WikiBigEdit](https://github.com/ExplainableML/WikiBigEdit) — bằng chứng chính cho việc loại bỏ ROME/MEMIT khỏi kiến trúc.
- *Locating and Editing Factual Associations in GPT* (ROME) — Meng, Bau, Andonian, Belinkov, **NeurIPS 2022**. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) · GitHub: [kmeng01/rome](https://github.com/kmeng01/rome).
- *Mass-Editing Memory in a Transformer* (MEMIT) — Meng et al., **ICLR 2023**. [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) · GitHub: [kmeng01/memit](https://github.com/kmeng01/memit) — 2 bài trên chỉ để hiểu cơ chế, không khuyến nghị dùng cho cập nhật liên tục.
- *EasyEdit: An Easy-to-use Knowledge Editing Framework for LLMs* — **ACL 2024** (demo track). GitHub: [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) — bộ công cụ hợp nhất nhiều phương pháp editing (ROME/MEMIT/MEND…), tiện để chạy thí nghiệm so sánh nhanh nếu vẫn muốn kiểm chứng câu hỏi con 5.2 bằng thực nghiệm thay vì chỉ dựa trên WikiBigEdit.
- *UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing* (2025) — [arXiv:2505.14679](https://arxiv.org/abs/2505.14679) — hướng thay thế mới hơn ROME/MEMIT, báo cáo ổn định tới hàng triệu lượt sửa; đáng thử nếu RAG không đủ cho một số loại tri thức.
- *Advancing Vietnamese Information Retrieval with Learning Objective and…* (2025) — [arXiv:2503.07470](https://arxiv.org/abs/2503.07470) — đánh giá các embedding model tiếng Việt cho retrieval, trực tiếp phục vụ câu hỏi con 5.1.

---

## Trụ cột 6 — Cổng tin cậy / phát hiện out-of-distribution (OOD)

**RQ6:** *Có thể xây một cơ chế phát hiện "model không chắc" đủ tin cậy để mô hình biết từ chối thay vì đoán sai, với chi phí tính toán thấp phù hợp một SLM?*

*(Phụ thuộc: cần có checkpoint từ Trụ cột 1–2 để hiệu chỉnh/calibrate.)*

- **Câu hỏi con 6.1:** Phương pháp nào rẻ nhất mà vẫn hiệu quả — entropy đầu ra, khoảng cách embedding tới phân bố huấn luyện, hay bất đồng thuận giữa vài forward pass (dropout-based)?
  - *Thí nghiệm:* Thử cả 3 phương pháp trên cùng 1 tập test có trộn câu hỏi "trong phân bố" và câu hỏi "ngoài phân bố" (chủ đề model chưa từng thấy) → vẽ đường ROC cho từng phương pháp.
- **Câu hỏi con 6.2:** Ngưỡng từ chối đặt ở đâu để cân bằng giữa "từ chối quá nhiều" (vô dụng) và "từ chối quá ít" (không đạt độ chính xác 99% như mục tiêu)?
  - *Thí nghiệm:* Vẽ đường cong đánh đổi giữa % câu bị từ chối và độ chính xác trên phần còn lại → chọn điểm theo ngưỡng đã định ở câu hỏi con 0.3.

**Tài liệu &amp; mã nguồn tham khảo:**

- *Detecting Hallucinations in Large Language Models Using Semantic Entropy* — Farquhar, Kossen, Kuhn, Gal (Oxford OATML), **Nature, 2024**. [Nature 630, 625–630](https://www.nature.com/articles/s41586-024-07421-0) · Code: [OATML GitHub](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html) — phương pháp entropy-based hàng đầu, trực tiếp trả lời câu hỏi con 6.1.
- *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs* (2024) — [arXiv:2406.15927](https://arxiv.org/abs/2406.15927) · GitHub: [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes) — bản rẻ hơn 5–10 lần so với semantic entropy gốc, ước tính trực tiếp từ hidden state — phù hợp ràng buộc chi phí của một SLM.
- *Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances* — **ACM Computing Surveys, 2025**. GitHub: [shuolucs/Awesome-Out-Of-Distribution-Detection](https://github.com/shuolucs/Awesome-Out-Of-Distribution-Detection) — bản đồ tổng quan để chọn phương pháp còn lại (embedding distance, dropout-based) cần so sánh ở câu hỏi con 6.1.

---

## Trụ cột 7 — Router hệ thống / ensemble (Tier 3 — OPTIONAL, không nằm trong lộ trình lõi hiện tại)

**RQ7:** *Một cơ chế định tuyến bất đối xứng (chỉ kích hoạt model/ensemble lớn hơn khi cổng OOD báo không chắc) có cải thiện được độ chính xác trên câu khó mà không phá vỡ mục tiêu "nhanh, nhẹ" ở phần lớn câu hỏi thông thường không?*

*(Trụ cột này hiện đang tham vọng hơn mức cần thiết cho giai đoạn hiện tại — chỉ quay lại nếu Tier 1–2 đã ổn định và có nhu cầu sản phẩm cụ thể đòi hỏi nó. Phụ thuộc: cần Trụ cột 6 hoạt động, và cần ít nhất 1 model dự phòng lớn hơn/khác kiến trúc để định tuyến tới.)*

- **Câu hỏi con 7.1:** Các model khác nhau (nếu dùng ensemble) có cần chung không gian embedding không, hay cần một lớp adapter để so khớp? (Đây là điểm kỹ thuật còn thiếu trong ý tưởng ensemble gốc của bạn — "routing encoder nhân tích chập với FFN của nhiều model khác nhau" giả định ngầm là các model đó tương thích không gian biểu diễn, điều này cần kiểm chứng chứ không mặc định đúng.)
- **Câu hỏi con 7.2:** Chi phí trung bình mỗi câu hỏi (latency, compute) tăng bao nhiêu khi thêm tầng router, và mức tăng đó có chấp nhận được theo spec ở Trụ cột 0 không?

**Tài liệu &amp; mã nguồn tham khảo:**

- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — DeepSeek-AI, 2025, xuất bản tại **Nature (2025)**. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) · [Nature 645, 633–638](https://www.nature.com/articles/s41586-025-09422-z) · GitHub chính thức: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — nếu Trụ cột 7 mở rộng sang việc dạy model tự suy luận (không chỉ định tuyến), đây là nguồn kỹ thuật quan trọng nhất hiện nay (RL thuần, không cần SFT reasoning traces).
- Ghi chú: OpenAI o1 không công bố paper/kiến trúc chi tiết hay code, chỉ có blog kỹ thuật ("Learning to reason with LLMs", OpenAI 2024) — dùng để tham khảo ý tưởng test-time scaling nhưng không có gì để tái lập trực tiếp.

---

## Sơ đồ phụ thuộc (thứ tự logic, không phải lịch — khớp với "Mức độ ưu tiên" ở đầu file)

- **Tier 1 — làm trước, độc lập, có thể song song:** Trụ cột 0 (bắt buộc trước tất cả), Trụ cột 1 (đã chẻ thành 1.1a–d, 1.2a–b, 1.3 arm A/B/C — phần lớn không cần backbone hoàn chỉnh), Trụ cột 5 (RAG có thể thử ngay trên model có sẵn).
- **Tier 1 tiếp theo — cần có kết luận sơ bộ từ Trụ cột 1:** Trụ cột 2 (backbone, đã chẻ thành 2.1a–b, 2.2a–b — 2.2a có thể chạy độc lập, không cần chờ Trụ cột 1).
- **Tier 2 — chỉ bắt đầu sau khi Tier 1 đã có kết luận, cần backbone đã chốt:** Trụ cột 3 (MoE-LoRA), Trụ cột 6 (OOD, cần checkpoint để hiệu chỉnh).
- **Tier 2 tiếp theo — cần có checkpoint đã qua Trụ cột 2–3:** Trụ cột 4 (alignment).
- **Tier 3 — optional, chỉ khi có nhu cầu cụ thể:** Trụ cột 7 (router/ensemble) — không đưa vào lộ trình mặc định.

## Lộ trình thực thi theo Phase (ánh xạ Tier ở trên sang các bước thực thi cụ thể)

Phase không phải mốc thời gian cố định — là các cổng (gate) tuần tự, chỉ mở phase sau khi phase trước đạt tiêu chí đi tiếp (khớp "Sơ đồ phụ thuộc" ở trên). Mỗi phase có file riêng trong `phases/`, link ngược lại đúng mục ở file này.

Compute thực tế hiện có: máy local (Apple M5, CPU-only — dùng cho code/scaffold/phân tích nhẹ không cần forward pass nặng), Kaggle (GPU free T4/P100, quota giới hạn theo tuần — dùng cho inference/job ngắn), 1 máy SSH rời có RTX 24GB (dùng cho job train thật, dài hơi hơn quota Kaggle cho phép).

- **[Phase 0 — Hạ tầng & scaffold repo](../phases/phase-0-scaffold.md)** — dựng khung thư mục, requirements, script tải tokenizer/dữ liệu, quy ước `runs/`. Compute: local. Không phụ thuộc trụ cột nào.
- **[Phase 1 — Tier 1, không cần train](../phases/phase-1-no-train.md)** — Trụ cột 1 (câu hỏi con 1.1, 1.2), Trụ cột 5 (câu hỏi con 5.1). Trụ cột 0 tạm hoãn (chưa chọn miền PoC). Compute: local cho 1.1 (chỉ cần tokenizer); Kaggle cho 1.2/5.1 (cần forward pass qua model).
- **[Phase 2 — Tier 1, cần train nhỏ](../phases/phase-2-small-train.md)** — Trụ cột 1 (câu hỏi con 1.3, arm A/B/C), Trụ cột 2 (2.1a–b, 2.2a–b), Trụ cột 5 (5.2). Compute: máy RTX 24GB (Kaggle chỉ dùng để debug phiên bản cực nhỏ trước khi chạy job đầy đủ).
- **[Phase 3 — Tier 2, cần backbone đã chốt](../phases/phase-3-tier2-backbone-ready.md)** — Trụ cột 3 (MoE-LoRA — phần theo miền cụ thể vẫn phụ thuộc quyết định Trụ cột 0 đang hoãn), Trụ cột 6 (OOD, không phụ thuộc domain nên chạy được ngay cả khi Trụ cột 0 chưa xong). Compute: RTX 24GB.
- **[Phase 4 — Tier 2 tiếp theo](../phases/phase-4-alignment.md)** — Trụ cột 4 (Alignment). Compute: RTX 24GB.
- **[Phase 5 — Tier 3, optional](../phases/phase-5-router-optional.md)** — Trụ cột 7. Chỉ mở nếu Phase 1–4 ổn định và có nhu cầu sản phẩm cụ thể.

**Trạng thái hiện tại:** Phase 0 đang thực hiện. Trụ cột 0 bị hoãn theo yêu cầu (chưa chọn miền/domain expert) — cần quay lại trước khi Phase 3 chạy được phần MoE-LoRA theo miền cụ thể.

---

## Nguyên tắc xuyên suốt khi thực hiện

1. Mỗi trụ cột đều có **tiêu chí dừng/pivot** — nếu giả thuyết sai, đó là kết quả nghiên cứu hợp lệ (không phải thất bại), và cần ghi lại lý do trước khi bỏ hướng đó.
2. Không chuyển sang trụ cột phụ thuộc khi trụ cột gốc chưa có kết luận rõ ràng (đạt hoặc không đạt tiêu chí) — tránh xây trên nền chưa kiểm chứng.
3. Mọi con số "thành công" phải được định nghĩa trước khi chạy thí nghiệm (đã làm ở Trụ cột 0), không định nghĩa ngược sau khi thấy kết quả.
4. Mọi thí nghiệm trong Tier 1 (Trụ cột 1, 2) phải tuân thủ nguyên tắc cô lập biến số ở đầu file — nếu một thí nghiệm thay đổi nhiều hơn 1 biến so với baseline, kết quả của nó không được dùng làm căn cứ kết luận cho bất kỳ câu hỏi con nào.

---

## Phụ lục: Nguồn dữ liệu tiếng Việt hiện có

**Cảnh báo chung:** Không có bộ dữ liệu tiếng Việt "sẵn dùng, chất lượng cao, quy mô lớn, hợp pháp rõ ràng" nào công khai — đúng như SWOT gốc của Viwixtech đã tự nhận diện. Các corpus lấy từ Common Crawl (CulturaX, VietVault…) có tình trạng bản quyền từng trang không rõ ràng; cần đánh giá rủi ro pháp lý trước khi dùng cho mục đích thương mại, không chỉ nghiên cứu.

### A. Corpus pretraining thô (phục vụ Trụ cột 1–2)

- **CulturaX** — 6,3 nghìn tỷ token, 167 ngôn ngữ, ghép và lọc kỹ từ mC4 + OSCAR (Nguyen et al., 2023). [arXiv:2309.09400](https://arxiv.org/abs/2309.09400) · HF: [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), subset tiếng Việt riêng tại [vietgpt/CulturaX](https://huggingface.co/datasets/vietgpt/CulturaX).
- **VietVault** — corpus tiếng Việt lọc riêng từ Common Crawl (trước 2023), đã dedup và lọc nội dung độc hại. HF: [nampdn-ai/vietvault](https://huggingface.co/datasets/nampdn-ai/vietvault).
- **binhvq NewsCorpus + Wikipedia + sách + văn bản pháp luật tiếng Việt** — công thức phối trộn phổ biến (~53GB tin tức, ~1,3GB Wikipedia, ~8,5GB sách, ~4,8GB văn bản pháp luật đã dedup), dùng trong continual pretraining các bản Vietnamese-Llama tại cộng đồng.
- **ViSoBERT 14GB corpus** — dữ liệu mạng xã hội (bình luận/bài đăng Facebook + MC4 ecommerce), phục vụ nếu cần model hiểu văn nói/ngôn ngữ đời thường. HF: [5CD-AI/visobert-14gb-corpus](https://huggingface.co/5CD-AI/visobert-14gb-corpus).
- Danh mục sống, cập nhật liên tục: [vndee/awsome-vietnamese-nlp](https://github.com/vndee/awsome-vietnamese-nlp) và [undertheseanlp/NLP-Vietnamese-progress](https://github.com/undertheseanlp/NLP-Vietnamese-progress) — nên kiểm tra định kỳ vì mảng này thay đổi nhanh.

### B. Instruction / SFT (phục vụ Trụ cột 3–4, giai đoạn SFT trước alignment)

- `VietnamAIHub/Vietnamese_llama1_30B_SFT` — hơn 200K instruction tổng hợp từ nhiều nguồn khác nhau.
- `1TuanPham/Vietnamese-OpenO1-SFT` — reasoning traces (toán, code, CoT) dịch sang tiếng Việt từ OpenO1 — nguồn khởi điểm tốt cho distillation reasoning ở Trụ cột 7.
- **Ghi chú quan trọng:** VinaLLaMA và PhoGPT — hai model nền tiếng Việt lớn nhất hiện có — đều **không dùng tập instruction công khai có sẵn**, mà tự sinh (VinaLLaMA công bố khoảng 1 triệu mẫu tổng hợp chất lượng cao, chưng cất từ GPT-4). Nhiều khả năng Viwixtech cũng phải đi theo hướng này: dùng LLM lớn sinh instruction rồi lọc lại, thay vì tải một tập có sẵn.

### C. Preference / alignment data (phục vụ Trụ cột 4, câu hỏi con 4.1)

- **Đây là lỗ hổng thật sự:** không tìm thấy bộ preference (chosen/rejected) tiếng Việt quy mô lớn nào công khai. Các nhóm hiện tại thường dịch UltraFeedback (tiếng Anh) hoặc tự sinh cặp preference bằng LLM giám khảo (LLM-as-judge). Nên coi việc tạo bộ preference tiếng Việt là một **hạng mục nghiên cứu/sản xuất dữ liệu riêng**, không giả định có sẵn.

### D. Dữ liệu theo miền / RAG (phục vụ Trụ cột 5)

- **Zalo AI Challenge 2021 — Legal Text Retrieval** — dữ liệu truy hồi văn bản pháp luật, được nhiều embedding tiếng Việt fine-tune trên đó (vd. `bqbbao6/vietnamese-legal-embedding`, NDCG@10 đạt 0,876 sau fine-tune so với 0,603 của base model).
- `nhminh107/VietEmbed-RAG-Science` — 68.567 cặp query–tài liệu, 7 lĩnh vực khoa học/kỹ thuật, dùng để huấn luyện/đánh giá retriever domain khoa học.
- **VN-MTEB** — benchmark so sánh ~19 model embedding tiếng Việt riêng cho tác vụ retrieval/RAG, dùng để chọn retriever ở câu hỏi con 5.1.

### E. Benchmark đánh giá (phục vụ Trụ cột 0 và 6)

- **VMLU** — 10.880 câu hỏi trắc nghiệm, 58 chủ đề, **ACL 2025**. GitHub: [ZaloAI-Jaist/VMLU](https://github.com/ZaloAI-Jaist/VMLU) (đã nêu ở Trụ cột 0).
- **VLSP** (NER 2016/2018, Sentiment 2016/2018, ABSA) — [vlsp.org.vn/resources](https://vlsp.org.vn/resources-vlsp2018): lưu ý phải điền form và gửi email xin cấp phép, **không tải trực tiếp được**.
- **UIT-ViQuAD, UIT-VSFC, UIT-VSMEC, ViHSD/ViHOS** — bộ benchmark chuyên biệt (đọc hiểu kiểu SQuAD, cảm xúc, hate speech) từ nhóm UIT (ĐH CNTT TP.HCM), hữu ích để đo hồi quy năng lực tổng quát khi thêm domain expert ở câu hỏi con 3.2.

---

## Phụ lục: Cấu trúc repo tham khảo

Dựa trên 3 repo đã kiểm chứng ở phần tài liệu tham khảo phía trên (không tự bịa cấu trúc):

- **Meta Lingua** (`facebookresearch/lingua`) — chính là codebase mà **BLT được xây trên đó** (README của BLT ghi rõ "BLT code is partially based on Meta Lingua"). Ý tưởng: tách `lingua/` (thư viện lõi) khỏi `apps/` (từng thí nghiệm ghép thành phần lại), mỗi lần chạy tạo 1 dump_dir riêng chứa `config.yaml` + snapshot code + `metrics.jsonl`.
- **state-spaces/mamba** — bên trong `mamba_ssm/` tách 3 tầng: `ops/` (kernel) → `modules/` (block) → `models/` (model hoàn chỉnh), cộng `benchmarks/`, `evals/`, `tests/` ở ngoài.
- **huggingface/trl** và **zjunlp/EasyEdit** — quy ước "mỗi phương pháp = 1 file trainer/algorithm riêng, cắm được vào framework chung."

```
vislm-research/
├── README.md
├── plans/
│   └── PLAN.md                      # chính là file kế hoạch này — nguồn sự thật duy nhất cho RQ/giả thuyết/tiêu chí
├── phases/                          # 1 file/phase, mỗi file link ngược lại đúng mục trong plans/PLAN.md
│   ├── README.md                    # mục lục + trạng thái từng phase
│   ├── phase-0-scaffold.md
│   ├── phase-1-no-train.md
│   ├── phase-2-small-train.md
│   ├── phase-3-tier2-backbone-ready.md
│   ├── phase-4-alignment.md
│   └── phase-5-router-optional.md
├── requirements.txt
│
├── setup/                           # ⟵ mượn setup/ của Meta Lingua
│   ├── create_env.sh
│   ├── download_prepare_data.py     # tải + làm sạch theo Phụ lục Dataset A–E ở trên
│   └── download_tokenizer.py
│
├── vislm/                           # CORE LIBRARY — tái dùng, không gắn 1 thí nghiệm cụ thể
│   ├── args.py                      # config dataclass + dot-list override ⟵ lingua/args.py
│   ├── data.py                      # ⟵ lingua/data.py
│   ├── distributed.py               # ⟵ lingua/distributed.py
│   ├── checkpoint.py
│   ├── logger.py
│   │
│   ├── tokenizers/                  # Trụ cột 1
│   │   ├── bpe_baseline.py          # Arm A
│   │   ├── blt_patch_encoder.py     # Arm B — port từ facebookresearch/blt
│   │   └── syllable_seed.py         # Arm C — cắm thêm vào blt_patch_encoder
│   │
│   ├── backbones/                   # Trụ cột 2 ⟵ 3 tầng ops/modules/models của mamba_ssm
│   │   ├── ops/
│   │   ├── modules/                 # transformer_block.py, mamba_block.py, hybrid_block.py
│   │   └── models/
│   │
│   ├── moe_lora/                    # Trụ cột 3
│   ├── alignment/                   # Trụ cột 4 ⟵ quy ước 1-trainer-1-file của huggingface/trl
│   │   ├── orpo_trainer.py
│   │   ├── simpo_trainer.py
│   │   └── embedding_filter.py      # lọc kiểu REAL/HEAL
│   ├── rag/                         # Trụ cột 5
│   ├── ood/                         # Trụ cột 6
│   └── routing/                     # Trụ cột 7 — OPTIONAL, chỉ tạo khi thật sự cần
│
├── experiments/                     # ⟵ apps/ của Meta Lingua: mỗi thí nghiệm 1 folder
│   ├── pillar0_target_spec/
│   ├── pillar1_patch_encoder/
│   │   └── configs/
│   │       ├── 1_1a_rare_compounds.yaml
│   │       ├── 1_1b_proper_nouns.yaml
│   │       ├── 1_1c_numbers.yaml
│   │       ├── 1_1d_typos.yaml
│   │       ├── 1_2a_entropy_multilingual_model.yaml
│   │       ├── 1_2b_entropy_vietnamese_model.yaml
│   │       ├── 1_3_arm_A_bpe.yaml
│   │       ├── 1_3_arm_B_blt.yaml
│   │       └── 1_3_arm_C_blt_syllable.yaml
│   ├── pillar2_backbone/
│   │   └── configs/
│   │       ├── 2_1a_ratio_sweep_compute_matched.yaml
│   │       ├── 2_1b_rope_ablation.yaml
│   │       ├── 2_2a_replicate_mohawk_english.yaml   # tái lập trước — bắt buộc
│   │       └── 2_2b_extend_vietnamese.yaml
│   ├── pillar3_moe_lora/  pillar4_alignment/  pillar5_rag/
│   ├── pillar6_ood_gate/  và pillar7_router_OPTIONAL/
│
├── evals/                           # ⟵ evals/ của mamba + apps/main/eval.py của lingua
│   ├── vmlu_harness.py              # ZaloAI-Jaist/VMLU
│   ├── needle_in_haystack_vi.py     # câu hỏi con 2.1a
│   └── ood_roc.py                   # câu hỏi con 6.1
│
├── data/{raw,prepared}/             # KHÔNG commit dữ liệu thô vào git
│
└── runs/                            # dump_dir ⟵ nguyên bản của Meta Lingua
    └── 2026_xx_xx_pillar1_arm_B/
        ├── config.yaml              # bằng chứng "chỉ đổi đúng 1 biến"
        ├── code/                    # snapshot code lúc chạy
        ├── checkpoints/
        ├── metrics.jsonl
        └── logs/

```

**4 quy tắc thực thi đi kèm (vận hành đúng nguyên tắc cô lập biến số):**

1. Tên file config trùng ID câu hỏi con trong file này (`1_1a…`, `2_2b…`) — truy vết 1-1 giữa tài liệu nghiên cứu và code.
2. `config.yaml` là nguồn sự thật duy nhất, không hard-code hyperparameter trong code — diff giữa 2 config chính là bằng chứng "chỉ đổi đúng 1 biến."
3. Mỗi lần chạy là 1 `runs/...` bất biến, không ghi đè — có code snapshot nên tái lập được nhiều tháng sau, log go/no-go lưu ngay cạnh kết quả.
4. Không tạo sẵn code cho `pillar3–6` khi Tier 1 chưa xong — tránh cám dỗ nhảy cóc.

