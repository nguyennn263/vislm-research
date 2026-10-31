"""Checks for the BLT-on-Qwen testbed (vislm/backbones/models/blt_qwen.py) against the
reference semantics of facebookresearch/blt @ 9774ed4. Runs with a tiny random Qwen3.5 body
(forward passes only) on CUDA when available - required once flash-linear-attention is
installed, its Triton kernels reject CPU tensors - else CPU. Needs transformers >= 5.

    PYTHONPATH=. .venv-qwen/bin/python -m pytest tests/test_blt_qwen.py -q

test_overfit_one_batch trains a tiny model for a few hundred steps; it is skipped unless
VISLM_RUN_OVERFIT=1 (run it on Kaggle/RTX, not on the local Mac).
"""
import os

import pytest
import torch

from vislm.backbones.models.blt_qwen import BLTQwen, BLTQwenConfig, QwenLatentTransformer
from vislm.backbones.modules.blt_layers import (
    BOS_ID,
    EOS_ID,
    OFFSET,
    byte_ngram_hash_ids,
    cross_attn_mask,
    encode_bytes,
    local_block_causal_mask,
    patch_ids_from_lengths,
)
from vislm.backbones.modules.blt_patcher import entropy_patch_lengths, whitespace_patch_lengths

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-4 if DEVICE == "cuda" else 1e-5  # chunked GPU kernels reorder float sums
TEXT = "Hà Nội là thủ đô của Việt Nam. Người dân nơi đây rất hiếu khách và cần cù."


def tiny_qwen_config(hidden: int = 64):
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=8, hidden_size=hidden, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )


def tiny_model(seed: int = 0, **overrides) -> BLTQwen:
    torch.manual_seed(seed)
    cfg = BLTQwenConfig(dim_local=32, n_layers_encoder=1, n_layers_decoder=2, n_heads_local=4,
                        cross_attn_nheads=4, cross_attn_k=2, hash_vocab=1009, local_attention_window=16)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    model = BLTQwen(cfg, QwenLatentTransformer.from_config(tiny_qwen_config(cfg.dim_local * cfg.cross_attn_k)))
    return model.to(DEVICE).eval()


def tokens_of(text: str = TEXT, device: str = "cpu") -> torch.Tensor:
    return torch.tensor([encode_bytes(text, add_eos=False)], device=device)


# --- tokenizer / hashing -------------------------------------------------------------------
def test_byte_ids_have_offset_and_bos():
    ids = encode_bytes("Hà", add_eos=True)
    assert ids == [BOS_ID, ord("H") + OFFSET, 0xC3 + OFFSET, 0xA0 + OFFSET, EOS_ID]


def test_ngram_hash_only_looks_back():
    x = tokens_of()
    for n in (3, 8):
        base = byte_ngram_hash_ids(x, n, 0, 500002)
        y = x.clone()
        y[0, 20] = 99
        changed = (byte_ngram_hash_ids(y, n, 0, 500002) != base)[0].nonzero().flatten().tolist()
        assert changed and min(changed) >= 20 and max(changed) <= 20 + n - 1


# --- patching: official semantics -------------------------------------------------------
def test_entropy_patching_matches_official_docstring_example():
    """get_blt_input docstring, "Handle the last byte properly": 14 tokens, include next
    token -> patch_lengths [1, 1, 3, 2, 2, 1, 2, 2, 1]. entropies[j] > t makes byte j+1
    start a patch, so high entropy at 1, 4, 6, 8, 9, 11, 13."""
    e = torch.zeros(1, 14)
    e[0, [1, 4, 6, 8, 9, 11, 13]] = 5.0
    lengths = entropy_patch_lengths(e, threshold=1.0, include_next_token=True)
    assert lengths.tolist() == [[1, 1, 3, 2, 2, 1, 2, 2, 1]]
    e0 = e.clone()
    e0[0, 0] = 9.0  # entropies[0] is never used (bytes 0 and 1 always start patches)
    assert torch.equal(entropy_patch_lengths(e0, 1.0), lengths)
    assert entropy_patch_lengths(e, 1.0, include_next_token=False).sum() == 14


def test_entropy_patching_monotonic_and_max_length():
    e = torch.tensor([[0.0, 0.0, 0.0, 3.0, 3.1, 0.0, 2.0, 2.0, 2.0, 2.0]])
    # monotonic: mask[0]=True (byte 2 starts), then e_t - e_{t-1} > 1 at t=3, 6 -> bytes 4, 7
    assert entropy_patch_lengths(e, 1.0, monotonicity=True).tolist() == [[1, 1, 2, 3, 4]]
    long = entropy_patch_lengths(torch.zeros(1, 20), 1.0, max_patch_length=6)
    assert long.tolist() == [[1, 6, 6, 6, 2]]


def test_patching_is_incremental():
    x = tokens_of()
    full = whitespace_patch_lengths(x)[0]
    for cut in (10, 25, 40):
        prefix = whitespace_patch_lengths(x[:, :cut])[0]
        starts_full = set(torch.cumsum(full, 0)[:-1].tolist()) | {0}
        starts_prefix = set(torch.cumsum(prefix, 0)[:-1].tolist()) | {0}
        assert {s for s in starts_full if s <= cut} == {s for s in starts_prefix if s <= cut}


def test_whitespace_patching_keeps_vietnamese_letters_whole():
    x = tokens_of("Ông ấy nói tiếng Việt.")
    lengths = whitespace_patch_lengths(x)[0].tolist()
    raw = x[0].tolist()
    pieces, i = [], 0
    for n in lengths:
        pieces.append(raw[i : i + n])
        i += n
    # pieces[0] is BOS, the last one is the 1-slot patch of the not-yet-seen next byte
    text_pieces = [bytes(b - OFFSET for b in p).decode("utf-8") for p in pieces[1:-1]]
    assert text_pieces == ["Ông ", "ấy ", "nói ", "tiếng ", "Việt."]


# --- masks ---------------------------------------------------------------------------
def test_local_block_causal_mask_window_and_documents():
    t = torch.tensor([[10, 11, EOS_ID, 12, 13, 14]])
    m = local_block_causal_mask(t, window=2)[0, 0]
    assert m[1].tolist() == [True, True, False, False, False, False]
    assert m[2].tolist() == [False, True, True, False, False, False]  # window 2
    assert m[3].tolist() == [False, False, False, True, False, False]  # new document after EOS


def test_lag_fix_invariant():
    """Decoder byte i uses latent j only if encoder patch j ends at or before i."""
    x = tokens_of()
    n = x.shape[1]
    lengths = whitespace_patch_lengths(x)
    enc_ids = patch_ids_from_lengths(lengths, n)[0]
    dec_ids = patch_ids_from_lengths(lengths[:, 1:], n)[0]
    patch_end = {j: max(i for i in range(n) if enc_ids[i] == j) for j in set(enc_ids.tolist())}
    for i in range(n):
        j = dec_ids[i].item()
        assert patch_end[j] <= i
        assert j + 1 not in patch_end or patch_end[j + 1] > i  # and it is the latest such patch
    dec_mask = cross_attn_mask(dec_ids[None], lengths.shape[1], 2, patches_as_queries=False)
    assert dec_mask.shape == (1, 1, n, lengths.shape[1] * 2)
    assert dec_mask[0, 0].sum(-1).eq(2).all()


# --- full model ------------------------------------------------------------------------
def test_forward_shapes_and_loss():
    model = tiny_model()
    x = tokens_of(device=DEVICE)
    lengths = whitespace_patch_lengths(x)
    logits, loss = model(x[:, :-1], whitespace_patch_lengths(x[:, :-1]), targets=x[:, 1:])
    assert logits.shape == (1, x.shape[1] - 1, 260)
    assert torch.isfinite(loss)
    assert lengths.shape[1] < x.shape[1]  # actually patched


def test_causal_and_sees_current_byte():
    """Logits at p (predicting byte p+1) must depend on bytes <= p, including p itself,
    and on nothing after p. Patches are recomputed from the perturbed bytes each time."""
    model = tiny_model()
    x = tokens_of(device=DEVICE)
    n = x.shape[1]
    with torch.no_grad():
        base, _ = model(x, whitespace_patch_lengths(x))
        assert torch.isfinite(base).all()
        for p in (3, 17, 30, n - 2):
            y = x.clone()
            y[0, p + 1 :] = (y[0, p + 1 :] - OFFSET + 37) % 256 + OFFSET  # change the future
            y[0, p + 1] = ord(" ") + OFFSET  # and force a different patch layout there
            out, _ = model(y, whitespace_patch_lengths(y))
            assert torch.allclose(out[0, : p + 1], base[0, : p + 1], atol=ATOL), f"future leaks into <= {p}"
            z = x.clone()
            z[0, p] = (z[0, p] - OFFSET + 1) % 256 + OFFSET  # change the current byte
            out, _ = model(z, whitespace_patch_lengths(z))
            assert not torch.allclose(out[0, p], base[0, p]), f"logits at {p} ignore byte {p}"


def test_decoder_reads_the_patch_that_just_ended():
    """The lag fix itself: when patch j ends at byte p, logits at p must read latent j
    (a decoder one patch behind would not), and must not read latent j+1."""
    model = tiny_model()
    x = tokens_of(device=DEVICE)
    n = x.shape[1]
    lengths = whitespace_patch_lengths(x)
    enc_ids = patch_ids_from_lengths(lengths, n)[0]
    real_global = model.global_model.forward

    def run(zero_latent=None):
        def fwd(h):
            out = real_global(h)
            if zero_latent is not None:
                out = out.clone()
                out[:, zero_latent] = 0
            return out
        model.global_model.forward = fwd
        with torch.no_grad():
            logits, _ = model(x, lengths)
        model.global_model.forward = real_global
        return logits[0]

    base = run()
    ends = [i for i in range(1, n - 1) if enc_ids[i + 1] != enc_ids[i]][:5]
    assert ends
    for p in ends:
        j = enc_ids[p].item()
        assert not torch.allclose(run(j)[p], base[p]), f"logits at {p} ignore latent {j} (patch ending at {p})"
        assert torch.allclose(run(j + 1)[: p + 1], base[: p + 1], atol=ATOL), f"latent {j + 1} leaks into <= {p}"


def test_batch_rows_with_different_patch_counts():
    """Zero-length padding patches must not produce NaNs or change the real rows."""
    model = tiny_model()
    a = tokens_of("Xin chào các bạn, hôm nay trời đẹp.", DEVICE)
    b = tokens_of("Xin chào các bạn,hômnaytrờiđẹp...", DEVICE)
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    la, lb = whitespace_patch_lengths(a), whitespace_patch_lengths(b)
    width = max(la.shape[1], lb.shape[1])
    pad = lambda l: torch.nn.functional.pad(l, (0, width - l.shape[1]))
    with torch.no_grad():
        both, _ = model(torch.cat([a, b]), torch.cat([pad(la), pad(lb)]))
        alone, _ = model(b, lb)
    assert torch.isfinite(both).all()
    assert torch.allclose(both[1], alone[0], atol=ATOL)


def test_double_residual_flag_changes_output():
    x = tokens_of(device=DEVICE)
    with torch.no_grad():
        a, _ = tiny_model(seed=1)(x, whitespace_patch_lengths(x))
        b, _ = tiny_model(seed=1, cross_attn_double_residual=False)(x, whitespace_patch_lengths(x))
    assert not torch.allclose(a, b)


@pytest.mark.skipif(os.environ.get("VISLM_RUN_OVERFIT") != "1", reason="training; run on Kaggle/RTX")
def test_overfit_one_batch():
    model = tiny_model().train()
    x = tokens_of(device=DEVICE)
    inp, tgt = x[:, :-1], x[:, 1:]
    lengths = whitespace_patch_lengths(inp)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(inp, lengths, targets=tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert BLTQwen.bits_per_byte(loss).item() < 0.5
