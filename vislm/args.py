"""Config loader: YAML defaults + dot-list CLI overrides (Meta Lingua style).

Lets one config.yaml drive both a tiny Kaggle smoke test and the full RTX 24GB
run, no hand-edited copies:
    python train.py configs/1_3_arm_A_bpe.yaml compute=kaggle dataset_target_gb=0.01
"""
import sys

import yaml


def _cast(value: str):
    if value.lower() in ("null", "none"):
        return None
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _set_nested(config: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_args(argv=None) -> dict:
    """argv[0] is a config.yaml path, argv[1:] are `key=value` overrides."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "=" in argv[0]:
        raise ValueError("first argument must be a config.yaml path")

    with open(argv[0]) as f:
        config = yaml.safe_load(f)

    for arg in argv[1:]:
        key, _, raw_value = arg.partition("=")
        _set_nested(config, key, _cast(raw_value))

    return config
