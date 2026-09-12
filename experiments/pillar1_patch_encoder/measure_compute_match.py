"""Print estimated FLOPs/byte for Arm A vs Arm B so their configs can be adjusted to
compute-match before running the real 1.3 comparison (plans/PLAN.md). See
vislm/backbones/flops.py for what this estimate does and doesn't capture.

    python -m experiments.pillar1_patch_encoder.measure_compute_match
"""
import os

import torch
import yaml

from vislm.backbones.flops import arm_a_flops_per_byte, blt_flops_per_byte, measure_avg_patch_length
from vislm.backbones.models.blt_lm import BLTLanguageModel
from vislm.backbones.models.transformer_lm import TransformerLM
from vislm.data import iter_texts
from vislm.train import batches

# measured in runs/2026_09_12_pillar1_phase1_kaggle_baseline_v3_phogpt/metrics.jsonl (1.1a-d)
MEASURED_BYTES_PER_PHOGPT_TOKEN = 5.06

SAMPLE_TEXT = (
    "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á, có bờ biển dài và nền văn hóa "
    "lâu đời. Người dân nơi đây rất cần cù và hiếu khách, luôn chào đón du khách bốn "
    "phương với sự nồng hậu đặc trưng của mình."
)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    arm_a_cfg = load_yaml("experiments/pillar1_patch_encoder/configs/1_3_arm_A_bpe.yaml")
    arm_b_cfg = load_yaml("experiments/pillar1_patch_encoder/configs/1_3_arm_B_blt.yaml")

    a_model = TransformerLM(
        vocab_size=20480,  # vinai/PhoGPT-4B
        d_model=arm_a_cfg["model"]["d_model"],
        n_layers=arm_a_cfg["model"]["n_layers"],
        n_heads=arm_a_cfg["model"]["n_heads"],
        max_seq_len=arm_a_cfg["model"]["max_seq_len"],
    )
    a_flops = arm_a_flops_per_byte(a_model, MEASURED_BYTES_PER_PHOGPT_TOKEN)

    e, p = arm_b_cfg["entropy_model"], arm_b_cfg["patching"]
    b_model = BLTLanguageModel(
        d_model=arm_b_cfg["model"]["d_model"],
        n_layers=arm_b_cfg["model"]["n_layers"],
        n_heads=arm_b_cfg["model"]["n_heads"],
        max_seq_len=arm_b_cfg["model"]["max_seq_len"],
        entropy_d_model=e["d_model"],
        entropy_n_layers=e["n_layers"],
        entropy_n_heads=e["n_heads"],
        entropy_threshold=p["entropy_threshold"],
        max_patch_len=p["max_patch_len"],
    )
    # A randomly-initialized entropy model has near-uniform (high) entropy everywhere,
    # so almost every byte becomes its own patch - not representative. Briefly
    # pretrain it first so avg_patch_len reflects a model that's learned *something*.
    pretrain_dataset = arm_b_cfg["dataset"]
    if os.path.isdir(pretrain_dataset) and os.listdir(pretrain_dataset):
        text = " ".join(iter_texts(pretrain_dataset))
    else:
        text = SAMPLE_TEXT * 50
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    opt = torch.optim.AdamW(b_model.entropy_model.parameters(), lr=3e-4)
    for step, (xs, ys) in enumerate(batches(ids, seq_len=128, batch_size=8, device="cpu")):
        _, loss = b_model.entropy_model(xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step >= 100:
            break
    print(f"(pretrained entropy model {step + 1} steps on {'real data' if os.path.isdir(pretrain_dataset) else 'repeated sample text'}, final loss {loss.item():.3f})")

    # measure avg_patch_len on the SAME data the entropy model was just pretrained on
    # (a separate hand-written sample gave a very different, misleading number here -
    # the entropy distribution is sensitive to the specific text).
    byte_ids = ids[: arm_b_cfg["model"]["max_seq_len"]]
    avg_patch_len = measure_avg_patch_length(b_model, byte_ids)
    b_flops = blt_flops_per_byte(b_model, avg_patch_len)

    print(f"Arm A: {a_model.num_params():,} params, ~{a_flops:,.0f} FLOPs/byte")
    print(
        f"Arm B: {b_model.num_latent_params():,} latent params, "
        f"avg patch length {avg_patch_len:.1f} bytes, ~{b_flops:,.0f} FLOPs/byte"
    )
    ratio = b_flops / a_flops
    print(f"Arm B is ~{ratio:.2f}x Arm A's FLOPs/byte")
    if abs(ratio - 1.0) > 0.2:
        direction = "reduce" if ratio > 1.0 else "increase"
        print(
            f"-> not compute-matched (outside +/-20%). Consider adjusting Arm B's "
            f"model.d_model/n_layers ({direction} them) and re-running this script."
        )
    else:
        print("-> within +/-20%, close enough to treat as compute-matched for a debug-scale test.")


if __name__ == "__main__":
    main()
