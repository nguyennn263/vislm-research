"""Arm C - BLT with syllable-seeded patch boundaries, plugs into blt_lm.py (Arm B) via
BLTLanguageModel(seed_boundaries_fn=seed_boundaries).

Vietnamese syllables are already whitespace-separated in Quoc Ngu script (the same
assumption question 1.2's entropy-vs-syllable-boundary analysis used), so "syllable
boundary" == "position right after a space byte" (0x20, same in UTF-8 as ASCII - no
need to decode). A position becomes a patch boundary if EITHER the entropy model
(Arm B) flags it OR it follows a space - this only adds splits, never removes an
entropy-based one.
"""


def seed_boundaries(byte_ids_1d, lengths, max_patch_len: int):
    space_boundaries = {i + 1 for i, b in enumerate(byte_ids_1d.tolist()) if b == 0x20}

    new_lengths = []
    offset, cur = 0, 0
    for length in lengths:
        for _ in range(length):
            cur += 1
            offset += 1
            if offset in space_boundaries or cur >= max_patch_len:
                new_lengths.append(cur)
                cur = 0
    if cur > 0:
        new_lengths.append(cur)
    return new_lengths
