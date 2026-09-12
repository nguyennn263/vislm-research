"""Save/load training checkpoints (model + optimizer + step), so a run interrupted by
Kaggle's 12h session limit (or anything else) can resume instead of restarting."""
import os

import torch


def save(run_dir: str, step: int, model, optimizer, extra: dict = None) -> str:
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},
    }
    path = os.path.join(run_dir, "checkpoints", f"step_{step}.pt")
    torch.save(state, path)
    # latest.pt always points at the most recent checkpoint, for a fixed resume path
    torch.save(state, os.path.join(run_dir, "checkpoints", "latest.pt"))
    return path


def load(path: str, model, optimizer=None, map_location: str = "cpu") -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
