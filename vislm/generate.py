"""Load a trained checkpoint and generate text - the direct way to see what a model's
loss/bits-per-byte number actually means (see phases/phase-2-small-train.md).

    python -m vislm.generate <config> <checkpoint.pt> "Hôm nay trời"
"""
import sys

import torch
import yaml

from vislm.backbones.models.blt_lm import BLTLanguageModel
from vislm.backbones.models.transformer_lm import TransformerLM
from vislm.train import build_model, load_tokenizer


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 60, temperature: float = 0.8):
    model.eval()
    ids = tok.encode(prompt, add_special_tokens=False)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.max_seq_len :]], dtype=torch.long)
        logits, _ = model(x)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    return tok.decode(ids)


def main():
    config_path, ckpt_path, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tok = load_tokenizer(cfg)
    model = build_model(cfg, vocab_size=tok.vocab_size)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    print(f"loaded checkpoint at step {state['step']}")

    torch.manual_seed(0)
    print(generate(model, tok, prompt))


if __name__ == "__main__":
    main()
