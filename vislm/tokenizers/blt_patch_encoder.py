"""Arm B - byte-level 'tokenizer': UTF-8 bytes, no learned vocab. Patching itself
(entropy model + patch boundaries) lives in vislm/backbones/models/blt_lm.py; this
module only provides the byte encode/decode interface vislm/train.py expects."""

VOCAB_SIZE = 256


class ByteTokenizer:
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(text.encode("utf-8"))

    def decode(self, byte_ids):
        return bytes(byte_ids).decode("utf-8", errors="ignore")


def load(pretrained_name: str = None) -> ByteTokenizer:
    return ByteTokenizer()
