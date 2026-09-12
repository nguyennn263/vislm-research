"""Group bytes into variable-length patches via a global entropy threshold - the
simpler of the two patching schemes in Pagnoni et al. (BLT), used for Arm B/C."""
import torch


def entropy_to_patch_lengths(entropy: torch.Tensor, threshold: float, max_patch_len: int):
    """entropy: 1D tensor [seq_len], entropy of predicting the NEXT byte at each
    position. A patch ends at position t (inclusive) when entropy[t] > threshold
    (model is uncertain about what follows - good place to cut) or when the patch
    hits max_patch_len (safety cap for the local encoder/decoder). Returns a list of
    patch lengths summing to seq_len."""
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
