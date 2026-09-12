# runs/

Quy ước dump_dir (mượn từ Meta Lingua, xem `plans/PLAN.md` mục "Phụ lục: Cấu trúc repo tham khảo"):

```
runs/<YYYY_MM_DD>_<pillar>_<arm>/
├── config.yaml     # nguồn sự thật duy nhất cho hyperparameter — diff giữa 2 config.yaml là bằng chứng "chỉ đổi đúng 1 biến"
├── code/           # snapshot code lúc chạy
├── checkpoints/
├── metrics.jsonl
└── logs/
```

Mỗi lần chạy là 1 thư mục bất biến, không ghi đè.
