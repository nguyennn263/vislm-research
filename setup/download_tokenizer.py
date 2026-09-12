"""Download a tokenizer for pillar 1 (question 1.1/1.2). Tokenizer only, no model weights."""
import argparse

from transformers import AutoTokenizer

# vinai/PhoGPT-4B's MPT config breaks on current transformers (see vislm/tokenizers/bpe_baseline.py)
DEFAULT_MODELS = {
    "gpt2_vi": "NlpHUST/gpt2-vietnamese",
    "phobert": "vinai/phobert-base",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2_vi", choices=list(DEFAULT_MODELS))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    name = DEFAULT_MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name, cache_dir=args.cache_dir)
    print(f"{name}: vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
