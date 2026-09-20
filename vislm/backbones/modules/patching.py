"""Ways to group bytes into variable-length patches for Arm B/C/D (plans/PLAN.md
question 1.3)."""
import torch


def entropy_to_patch_lengths(entropy: torch.Tensor, threshold: float, max_patch_len: int):
    """entropy: 1D tensor [seq_len], entropy of predicting the NEXT byte at each
    position. A patch ends at position t (inclusive) when entropy[t] > threshold
    (model is uncertain about what follows - good place to cut) or when the patch
    hits max_patch_len (safety cap for the local encoder/decoder). Returns a list of
    patch lengths summing to seq_len. Used by Arm B/C (entropy_threshold boundary mode)."""
    lengths = []
    cur = 0
    for e in entropy.tolist():
        cur += 1
        if e > threshold or cur >= max_patch_len:
            lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return lengths


def bpe_guided_patch_lengths(byte_ids_1d: torch.Tensor, bpe_tokenizer, max_patch_len: int):
    """Patch boundaries taken from a real BPE tokenizer's token spans, instead of a
    learned entropy model - "Arm D" idea (see phases/phase-2-small-train.md): borrow
    BPE's proven chunking without paying for its large vocab/embedding table (still
    byte-level vocab=256 underneath). Decodes the byte window to text (best-effort;
    a window edge can land mid-UTF8-character, dropped via errors="ignore" - a minor,
    documented approximation, not a bug) and converts the tokenizer's character offsets
    to byte offsets.
    """
    text = bytes(byte_ids_1d.tolist()).decode("utf-8", errors="ignore")
    offsets = bpe_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)[
        "offset_mapping"
    ]

    lengths = []
    prev_byte_end = 0
    for char_start, char_end in offsets:
        if char_end <= char_start:
            continue
        byte_end = len(text[:char_end].encode("utf-8"))
        length = byte_end - prev_byte_end
        while length > max_patch_len:
            lengths.append(max_patch_len)
            length -= max_patch_len
        if length > 0:
            lengths.append(length)
        prev_byte_end = byte_end

    total = sum(lengths)
    if total < len(byte_ids_1d):
        lengths.append(len(byte_ids_1d) - total)  # bytes lost to decode errors at the tail
    return lengths
