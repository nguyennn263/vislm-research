"""Arm A (baseline) for pillar 1 - current Vietnamese BPE, unchanged."""
from transformers import AutoTokenizer


def load(pretrained_name: str = "NlpHUST/gpt2-vietnamese"):
    """Load a tokenizer only (no model weights).

    Default is NOT vinai/PhoGPT-4B (the name suggested in plans/PLAN.md): its MPT
    config raises StrictDataclassFieldValidationError on current transformers
    (attn_pdrop int vs float), confirmed when running the Kaggle baseline in
    experiments/pillar1_patch_encoder/kaggle/. NlpHUST/gpt2-vietnamese is plain
    GPT-2 BPE trained on Vietnamese text, no custom modeling code.
    """
    return AutoTokenizer.from_pretrained(pretrained_name)
