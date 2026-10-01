"""Causality guards for every Pillar 1 arm (plans/PLAN.md question 1.3, step 0a/0b).

0a - logits at position p (trained to predict byte/token p+1) must depend on inputs <= p
     and NOT on inputs > p. The pre-fix BLT decoder failed the first half (logits at p
     never saw byte p - a skip-one-byte task).
0b - patching must be incremental (BLT paper: f_p(x_<i) = f_p(x)_<i): a boundary at i
     may only depend on bytes <= i. Entropy and syllable patching satisfy this; BPE-guided
     patching (Arm D) does not - see test_bpe_guided_patching_leaks_future_known_issue.

    PYTHONPATH=. python tests/test_blt_causal.py      # no pytest needed
    PYTHONPATH=. pytest tests/test_blt_causal.py
"""
import glob
import json
import os

import torch

from vislm.backbones.models.blt_lm import BLTLanguageModel
from vislm.backbones.models.transformer_lm import TransformerLM
from vislm.tokenizers import syllable_seed

SEQ_LEN = 96
PROBE_POSITIONS = [5, 23, 47, 70]
PHOGPT = "vinai/PhoGPT-4B"
FALLBACK_TEXT = (
    "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á, có bờ biển dài và nền văn hóa "
    "lâu đời. Người dân nơi đây rất cần cù và hiếu khách."
)


def _sample_bytes(n: int) -> torch.Tensor:
    """Real Vietnamese text when data/prepared exists (BPE boundaries need real text),
    else a fixed sentence - both are valid UTF-8 at the start of the window."""
    shards = sorted(glob.glob("data/prepared/fineweb2_vi/shard_*.jsonl"))
    text = FALLBACK_TEXT * 4
    if shards:
        with open(shards[0], encoding="utf-8") as f:
            text = json.loads(f.readline())["text"] + " " + text
    return torch.tensor(list(text.encode("utf-8"))[:n], dtype=torch.long)


def _perturb(x: torch.Tensor, q: int) -> torch.Tensor:
    y = x.clone()
    y[0, q] = (y[0, q] + 77) % 256
    return y


def _logits_changed(model, x, p: int, q: int) -> bool:
    with torch.no_grad():
        base, _ = model(x)
        out, _ = model(_perturb(x, q))
    return (out[0, p] - base[0, p]).abs().max().item() > 1e-6


def _small_blt(**kwargs) -> BLTLanguageModel:
    torch.manual_seed(0)
    return BLTLanguageModel(d_model=64, n_layers=2, n_heads=4, max_seq_len=SEQ_LEN, **kwargs).eval()


def _with_median_threshold(model: BLTLanguageModel, x: torch.Tensor) -> BLTLanguageModel:
    """An untrained entropy model gives ~ln(256) everywhere; thresholding near the median
    of this input's entropies yields genuinely variable patch lengths to test against.
    The threshold sits in the widest gap between sorted entropies of the middle half, not
    on a value itself: entropies of a prefix vs the full sequence differ by ~1e-6 (float
    noise from different attention lengths), which would flip a boundary sitting exactly
    on the threshold and look like a (fake) causality leak."""
    with torch.no_grad():
        ent = model.entropy_model.next_byte_entropy(x)[0]
    s = ent.sort().values
    lo, hi = len(s) // 4, 3 * len(s) // 4
    gaps = s[lo + 1 : hi] - s[lo : hi - 1]
    k = lo + gaps.argmax().item()
    assert gaps.max().item() > 1e-4, "entropies too tightly packed for a noise-safe threshold"
    model.entropy_threshold = ((s[k] + s[k + 1]) / 2).item()
    return model


def _blt_variants(x: torch.Tensor):
    yield "mean_pool", _with_median_threshold(_small_blt(encoder_type="mean_pool"), x)
    yield "cross_attention", _with_median_threshold(_small_blt(encoder_type="cross_attention"), x)
    yield "syllable_seeded", _with_median_threshold(
        _small_blt(seed_boundaries_fn=syllable_seed.seed_boundaries), x
    )


def _assert_causal(name, model, x):
    for p in PROBE_POSITIONS:
        for q in (p - 1, p):
            assert _logits_changed(model, x, p, q), f"{name}: logits[{p}] ignore input {q} (<= p)"
        for q in (p + 1, p + 9):
            assert not _logits_changed(model, x, p, q), f"{name}: logits[{p}] see input {q} (> p)"


def test_transformer_lm_is_causal():
    torch.manual_seed(0)
    model = TransformerLM(vocab_size=256, d_model=64, n_layers=2, n_heads=4, max_seq_len=SEQ_LEN).eval()
    _assert_causal("TransformerLM", model, _sample_bytes(SEQ_LEN).unsqueeze(0))


def test_blt_is_causal():
    x = _sample_bytes(SEQ_LEN).unsqueeze(0)
    for name, model in _blt_variants(x):
        lengths = model._patch_lengths_for(x[0])
        assert len(set(lengths)) > 1, f"{name}: patching degenerate ({lengths}), test is vacuous"
        _assert_causal(name, model, x)


def _patch_ends(lengths):
    ends, total = set(), 0
    for length in lengths:
        total += length
        ends.add(total - 1)
    return ends


def _is_incremental(model, ids_1d) -> bool:
    """Boundaries strictly inside every prefix must match the full sequence's boundaries
    there (the prefix's last patch is cut at the prefix end, so that one is excluded)."""
    full = _patch_ends(model._patch_lengths_for(ids_1d))
    for n in range(8, len(ids_1d)):
        pref = _patch_ends(model._patch_lengths_for(ids_1d[:n]))
        if {e for e in pref if e < n - 1} != {e for e in full if e < n - 1}:
            return False
    return True


def test_entropy_and_syllable_patching_are_incremental():
    x = _sample_bytes(SEQ_LEN).unsqueeze(0)
    for name, model in _blt_variants(x):
        assert _is_incremental(model, x[0]), f"{name}: patch boundaries depend on future bytes"


def _phogpt_cached() -> bool:
    cache = os.path.expanduser("~/.cache/huggingface/hub/models--vinai--PhoGPT-4B")
    return os.path.isdir(cache) or os.path.isdir("/kaggle/input")


def test_bpe_guided_patching_leaks_future_known_issue():
    """Documents why Arm D (bpe_guided) results cannot be trusted: BPE boundaries depend on
    characters after the boundary. When Arm D moves to a causal BPE-boundary predictor
    (the official repo's `bpe_patcher` mode - PLAN.md 1.3 step 0b), replace this with a
    positive incremental test."""
    if not _phogpt_cached():
        print("  skipped: PhoGPT-4B tokenizer not cached")
        return
    x = _sample_bytes(SEQ_LEN).unsqueeze(0)
    model = _small_blt(boundary_mode="bpe_guided", bpe_tokenizer_name=PHOGPT)
    assert not _is_incremental(model, x[0]), (
        "bpe_guided patching now looks incremental - update this test and PLAN.md 1.3"
    )


if __name__ == "__main__":
    for test_name, fn in list(globals().items()):
        if test_name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {test_name}")
