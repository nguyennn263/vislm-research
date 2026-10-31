"""Stream + subsample FineWeb2 Vietnamese (HuggingFaceFW/fineweb-2, config vie_Latn)
into data/prepared/, sized by raw UTF-8 bytes so the same shards feed any tokenizer/arm
in the Pillar 1 compute-matched comparison (see plans/PLAN.md, phases/phase-2-small-train.md).

Never downloads the full ~130GB config, just what's needed. Two sources:
  parquet (default)  read row groups of data/vie_Latn/train/*.parquet straight over HTTP
                     (HfFileSystem range reads). No repo listing, so it does not stall on
                     the Hub's rate limit for unauthenticated clients - the `datasets`
                     streaming path hung for 6h in a Kaggle kernel on 2026-10-23.
  stream             the original `datasets` streaming loader.
  --parquet-dir DIR  read local *.parquet files instead (e.g. the filtered FineWeb2 copy on
                     the RTX box), in sorted file order.
--max-docs N stops after N documents (quick tests), whichever of --target-gb/--max-docs comes first.
Set HF_TOKEN for higher Hub rate limits either way.
"""
import argparse
import json
import os

DATASET = "HuggingFaceFW/fineweb-2"
PARQUET_FILE = "datasets/HuggingFaceFW/fineweb-2/data/vie_Latn/train/{i:03d}_00000.parquet"
N_PARQUET = 30  # 000_00000 ... (checked 2026-10-23); files are ~4.8GB each


def iter_parquet_texts():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for i in range(N_PARQUET):
        with fs.open(PARQUET_FILE.format(i=i), "rb", block_size=16 * 2**20) as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                    yield text


def iter_local_parquet_texts(parquet_dir):
    import glob

    import pyarrow.parquet as pq

    for path in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for batch in pf.iter_batches(row_groups=[rg], columns=["text"], batch_size=4096):
                yield from batch.column("text").to_pylist()


def iter_stream_texts():
    from datasets import load_dataset

    for row in load_dataset(DATASET, "vie_Latn", split="train", streaming=True):
        yield row["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-gb", type=float, default=1.0, help="raw text to collect, in GB")
    parser.add_argument("--out-dir", default="data/prepared/fineweb2_vi")
    parser.add_argument("--shard-size-mb", type=int, default=200)
    parser.add_argument("--source", choices=["parquet", "stream"], default="parquet")
    parser.add_argument("--parquet-dir", default=None, help="local parquet files instead of the Hub")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.parquet_dir:
        texts = iter_local_parquet_texts(args.parquet_dir)
    else:
        texts = iter_parquet_texts() if args.source == "parquet" else iter_stream_texts()

    target_bytes = int(args.target_gb * 1e9)
    shard_bytes = int(args.shard_size_mb * 1e6)
    total = 0
    shard_idx = 0
    shard_written = 0

    def open_shard(i):
        return open(os.path.join(args.out_dir, f"shard_{i:04d}.jsonl"), "w", encoding="utf-8")

    f = open_shard(shard_idx)
    n_docs = 0
    for text in texts:
        line = json.dumps({"text": text}, ensure_ascii=False)
        n = len(line.encode("utf-8")) + 1
        f.write(line + "\n")
        total += n
        shard_written += n
        if shard_written >= shard_bytes:
            f.close()
            shard_idx += 1
            shard_written = 0
            f = open_shard(shard_idx)
        n_docs += 1
        if total >= target_bytes or (args.max_docs and n_docs >= args.max_docs):
            break
    f.close()
    print(f"wrote {n_docs} docs, {total / 1e9:.3f} GB across {shard_idx + 1} shard(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
