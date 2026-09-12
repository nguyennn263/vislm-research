"""Compute-matching helper for Pillar 1 question 1.3 (plans/PLAN.md). Uses the
~6-FLOPs-per-parameter-per-token heuristic (Kaplan et al. 2020, "Scaling Laws for
Neural Language Models") as a CONSISTENT, comparable estimate across arms - not an
exact FLOP count. Good enough to size Arm B/C to roughly match Arm A's compute before
running the real comparison; profiling actual FLOPs is a possible follow-up refinement.

Comparison unit is FLOPs per BYTE of raw text (not per token/patch), since the dataset
is byte-budgeted (setup/download_prepare_data.py) and bytes are the one unit shared by
BPE (Arm A) and byte-level (Arm B/C) architectures.
"""
from vislm.backbones.modules.transformer_block import count_params


def flops_per_step(n_params: int) -> int:
    return 6 * n_params


def arm_a_flops_per_byte(model, bytes_per_token: float) -> float:
    """model: TransformerLM. bytes_per_token: measured empirically (see
    runs/2026_09_12_pillar1_phase1_kaggle_baseline_v3_phogpt/ - ~5.06 for PhoGPT-4B
    on FineWeb2 Vietnamese)."""
    return flops_per_step(model.num_params()) / bytes_per_token


def blt_flops_per_byte(model, avg_bytes_per_patch: float) -> float:
    """model: BLTLanguageModel. avg_bytes_per_patch: measured by running the patcher
    over a real sample (depends on entropy_threshold/max_patch_len) - see
    measure_avg_patch_length() below."""
    latent_params = sum(count_params(b) for b in model.latent_blocks)
    decoder_params = sum(count_params(b) for b in model.decoder_blocks)
    per_byte_params = decoder_params + count_params(model.local_encoder_proj)
    return flops_per_step(latent_params) / avg_bytes_per_patch + flops_per_step(per_byte_params)


def measure_avg_patch_length(model, byte_ids_1d) -> float:
    lengths = model._patch_lengths_for(byte_ids_1d)
    return sum(lengths) / len(lengths)
