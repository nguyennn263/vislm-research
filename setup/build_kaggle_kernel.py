"""Build a Kaggle kernel folder (notebook + kernel-metadata.json) from a py-percent script,
so Kaggle notebooks live in the repo as reviewable .py files instead of hand-edited .ipynb.

    python setup/build_kaggle_kernel.py experiments/blt_qwen/kaggle/blt_qwen_smoke.py \
        --slug vislm-blt-qwen-smoke --out /tmp/kernel
    kaggle kernels push -p /tmp/kernel --accelerator NvidiaTeslaT4

Cells are separated by `# %%`; `# %% [markdown]` cells have their leading "# " stripped.
The kernel mounts the code dataset built by setup/sync_kaggle_code_dataset.sh.
"""
import argparse
import json
import os
import re

OWNER = "nguyennn263"
CODE_DATASET = f"{OWNER}/vislm-research-code"


def to_cells(src: str):
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            lines = [re.sub(r"^# ?", "", line) for line in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": body})
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    with open(args.script, encoding="utf-8") as f:
        cells = to_cells(f.read())
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                       "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(args.out, exist_ok=True)
    nb_name = f"{args.slug}.ipynb"
    with open(os.path.join(args.out, nb_name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    meta = {
        "id": f"{OWNER}/{args.slug}", "title": args.slug, "code_file": nb_name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": not args.no_gpu, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [CODE_DATASET], "competition_sources": [], "kernel_sources": [],
    }
    with open(os.path.join(args.out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(cells)} cells to {args.out}/{nb_name}")


if __name__ == "__main__":
    main()
