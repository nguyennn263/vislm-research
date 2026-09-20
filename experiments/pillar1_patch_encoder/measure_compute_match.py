"""Print estimated FLOPs/byte for Arm A vs every BLT-family config (B, B2, C, D, ...) so
they can be adjusted to compute-match before running the real 1.3 comparison
(plans/PLAN.md). See vislm/backbones/flops.py for what this estimate does and doesn't
capture.

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

CONFIG_DIR = "experiments/pillar1_patch_encoder/configs"
BLT_CONFIGS = [
    "1_3_arm_B_blt.yaml",
    "1_3_arm_B2_cross_attention.yaml",
    "1_3_arm_D_bpe_guided.yaml",
]

SAMPLE_TEXT = (
    "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á, có bờ biển dài và nền văn hóa "
    "lâu đời. Người dân nơi đây rất cần cù và hiếu khách, luôn chào đón du khách bốn "
    "phương với sự nồng hậu đặc trưng của mình."
)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_blt_from_cfg(cfg):
    m, p = cfg["model"], cfg["patching"]
    e = cfg.get("entropy_model", {})
    boundary_mode = "bpe_guided" if cfg["architecture"] == "blt_bpe_guided" else "entropy"
    return BLTLanguageModel(
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        max_seq_len=m["max_seq_len"],
        max_patch_len=p["max_patch_len"],
        boundary_mode=boundary_mode,
        entropy_d_model=e.get("d_model", 128),
        entropy_n_layers=e.get("n_layers", 4),
        entropy_n_heads=e.get("n_heads", 4),
        entropy_threshold=p.get("entropy_threshold", 1.5),
        bpe_tokenizer_name=p.get("bpe_tokenizer_name"),
        encoder_type=cfg.get("encoder_type", "mean_pool"),
        n_encoder_layers=cfg.get("n_encoder_layers", 2),
    )


def get_sample_text(dataset_dir):
    if os.path.isdir(dataset_dir) and os.listdir(dataset_dir):
        return " ".join(iter_texts(dataset_dir))
    return SAMPLE_TEXT * 50


def main():
    arm_a_cfg = load_yaml(f"{CONFIG_DIR}/1_3_arm_A_bpe.yaml")
    a_model = TransformerLM(
        vocab_size=20480,  # vinai/PhoGPT-4B
        d_model=arm_a_cfg["model"]["d_model"],
        n_layers=arm_a_cfg["model"]["n_layers"],
        n_heads=arm_a_cfg["model"]["n_heads"],
        max_seq_len=arm_a_cfg["model"]["max_seq_len"],
    )
    a_flops = arm_a_flops_per_byte(a_model, MEASURED_BYTES_PER_PHOGPT_TOKEN)
    print(f"Arm A: {a_model.num_params():,} params, ~{a_flops:,.0f} FLOPs/byte\n")

    for config_name in BLT_CONFIGS:
        cfg = load_yaml(f"{CONFIG_DIR}/{config_name}")
        model = build_blt_from_cfg(cfg)
        text = get_sample_text(cfg["dataset"])
        ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

        if model.entropy_model is not None:
            # 500 steps, matching entropy_pretrain_steps actually used in real runs -
            # fewer steps (e.g. 100) under-trains the entropy model and gives a
            # misleadingly small avg_patch_len (see sweep_entropy_threshold.py findings
            # in phases/phase-2-small-train.md).
            opt = torch.optim.AdamW(model.entropy_model.parameters(), lr=3e-4)
            for step, (xs, ys) in enumerate(batches(ids, seq_len=128, batch_size=8, device="cpu")):
                _, loss = model.entropy_model(xs, ys)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step >= 500:
                    break

        byte_ids = ids[: cfg["model"]["max_seq_len"]]
        avg_patch_len = measure_avg_patch_length(model, byte_ids)
        flops = blt_flops_per_byte(model, avg_patch_len)
        ratio = flops / a_flops

        print(f"{config_name}:")
        print(
            f"  {model.num_latent_params():,} latent params, "
            f"avg patch length {avg_patch_len:.1f} bytes, ~{flops:,.0f} FLOPs/byte"
        )
        print(f"  {ratio:.2f}x Arm A's FLOPs/byte", end=" ")
        if abs(ratio - 1.0) > 0.2:
            direction = "reduce" if ratio > 1.0 else "increase"
            print(f"-> NOT compute-matched (outside +/-20%), consider {direction} model.d_model/n_layers")
        else:
            print("-> within +/-20%, close enough")
        print()


if __name__ == "__main__":
    main()
