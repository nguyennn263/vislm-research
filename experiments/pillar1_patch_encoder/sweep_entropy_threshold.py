"""Systematic entropy_threshold sweep for Arm B/C (plans/PLAN.md question 1.3).

Previous runs used a threshold (3.7) picked from one ad-hoc manual check - this script
formalizes that into a repeatable tool: pretrain the entropy model properly (same
entropy_pretrain_steps as the real config), then sweep thresholds and report avg patch
length, targeting Arm A's own bytes/token (~5.06) as a natural common granularity.

    python -m experiments.pillar1_patch_encoder.sweep_entropy_threshold
"""
import statistics

import torch
import yaml

from vislm.backbones.modules.entropy_model import ByteEntropyModel
from vislm.backbones.modules.patching import entropy_to_patch_lengths
from vislm.data import iter_texts
from vislm.train import batches

TARGET_BYTES_PER_PATCH = 5.06  # Arm A's measured bytes/token (see runs/..._v3_phogpt/)
THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5]


def main():
    with open("experiments/pillar1_patch_encoder/configs/1_3_arm_B_blt.yaml") as f:
        cfg = yaml.safe_load(f)
    e, m, t = cfg["entropy_model"], cfg["model"], cfg["train"]

    torch.manual_seed(cfg.get("seed", 42))
    entropy_model = ByteEntropyModel(e["d_model"], e["n_layers"], e["n_heads"], m["max_seq_len"])

    text = " ".join(iter_texts(cfg["dataset"]))
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    # NOTE: config's own entropy_pretrain_steps default (50) turned out too low to learn
    # anything useful (every threshold degenerated to avg_patch_len=1.0 - see
    # phases/phase-2-small-train.md) - actual training runs already override this to
    # 500, so sweep with that instead of blindly trusting the (stale) config default.
    steps = max(t.get("entropy_pretrain_steps", 100), 500)
    opt = torch.optim.AdamW(entropy_model.parameters(), lr=t["lr"])
    for step, (xs, ys) in enumerate(batches(ids, m["max_seq_len"], t["batch_size"], "cpu")):
        _, loss = entropy_model(xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step >= steps:
            break
    print(f"pretrained entropy model {step + 1} steps, final loss {loss.item():.3f}\n")

    # measure over several held-out windows, not just one, for a less noisy estimate
    n_windows = 8
    window_len = m["max_seq_len"]
    windows = [ids[i * window_len : (i + 1) * window_len] for i in range(n_windows)]

    print(f"{'threshold':>10} {'avg_patch_len':>14} {'|diff from target|':>20}")
    best_threshold, best_diff = None, float("inf")
    for threshold in THRESHOLDS:
        all_lengths = []
        for w in windows:
            entropy = entropy_model.next_byte_entropy(w.unsqueeze(0))[0]
            all_lengths.extend(entropy_to_patch_lengths(entropy, threshold, cfg["patching"]["max_patch_len"]))
        avg_len = statistics.mean(all_lengths)
        diff = abs(avg_len - TARGET_BYTES_PER_PATCH)
        print(f"{threshold:>10} {avg_len:>14.2f} {diff:>20.2f}")
        if diff < best_diff:
            best_threshold, best_diff = threshold, diff

    print(f"\nclosest to target ({TARGET_BYTES_PER_PATCH} bytes/patch): threshold={best_threshold}")


if __name__ == "__main__":
    main()
