"""Stream + subsample FineWeb2 Vietnamese (HuggingFaceFW/fineweb-2, config vie_Latn)
into data/prepared/, sized by raw UTF-8 bytes so the same shards feed any tokenizer/arm
in the Pillar 1 compute-matched comparison (see plans/PLAN.md, phases/phase-2-small-train.md).

Uses streaming mode - never downloads the full ~130GB config, just what's needed.
"""
import argparse
import json
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-gb", type=float, default=1.0, help="raw text to collect, in GB")
    parser.add_argument("--out-dir", default="data/prepared/fineweb2_vi")
    parser.add_argument("--shard-size-mb", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-2", "vie_Latn", split="train", streaming=True)

    target_bytes = int(args.target_gb * 1e9)
    shard_bytes = int(args.shard_size_mb * 1e6)
    total = 0
    shard_idx = 0
    shard_written = 0

    def open_shard(i):
        return open(os.path.join(args.out_dir, f"shard_{i:04d}.jsonl"), "w", encoding="utf-8")

    f = open_shard(shard_idx)
    for row in ds:
        line = json.dumps({"text": row["text"]}, ensure_ascii=False)
        n = len(line.encode("utf-8")) + 1
        f.write(line + "\n")
        total += n
        shard_written += n
        if shard_written >= shard_bytes:
            f.close()
            shard_idx += 1
            shard_written = 0
            f = open_shard(shard_idx)
        if total >= target_bytes:
            break
    f.close()
    print(f"wrote {total / 1e9:.3f} GB across {shard_idx + 1} shard(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
