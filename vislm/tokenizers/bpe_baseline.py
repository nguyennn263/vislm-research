"""Arm A (baseline) for pillar 1 - current Vietnamese BPE, unchanged."""
from transformers import AutoTokenizer


def load(pretrained_name: str = "vinai/PhoGPT-4B"):
    """Load a tokenizer only (no model weights).

    On Kaggle's default transformers build, PhoGPT-4B's MPT config raises
    StrictDataclassFieldValidationError (attn_pdrop int vs float) - pin
    `transformers==4.46.3` first (see experiments/pillar1_patch_encoder/kaggle/).
    Works as-is with transformers>=4.57 locally.
    """
    return AutoTokenizer.from_pretrained(pretrained_name)
