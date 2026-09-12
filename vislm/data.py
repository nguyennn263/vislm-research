"""Read prepared text shards (jsonl, one {"text": ...} per line) produced by
setup/download_prepare_data.py. Shared across Pillar 1 arms so they train on the exact
same bytes (required for the compute-matched comparison in plans/PLAN.md).
"""
import glob
import json
import os


def shard_paths(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "shard_*.jsonl")))


def iter_texts(data_dir: str):
    for path in shard_paths(data_dir):
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def total_bytes(data_dir: str) -> int:
    return sum(os.path.getsize(p) for p in shard_paths(data_dir))
