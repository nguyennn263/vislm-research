"""Download a tokenizer for pillar 1 (question 1.1/1.2). Tokenizer only, no model weights."""
import argparse

from transformers import AutoTokenizer

DEFAULT_MODELS = {
    "phogpt": "vinai/PhoGPT-4B",  # on Kaggle, pin transformers==4.46.3 first (see vislm/tokenizers/bpe_baseline.py)
    "phobert": "vinai/phobert-base",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="phogpt", choices=list(DEFAULT_MODELS))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    name = DEFAULT_MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name, cache_dir=args.cache_dir)
    print(f"{name}: vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
