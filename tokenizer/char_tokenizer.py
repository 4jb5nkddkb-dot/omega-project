"""字符级 / 字节级分词器（纯 NumPy + 标准库）。

从单体文件 ``dual_transformer_numpy.py``（第 75–161 行）原样抽出，
行为完全一致：

- ``CharTokenizer``：字符级，ID 0=BOS、1=EOS、2=UNK，3.. 为按码点排序的字符；
  字符表存在检查点里，续跑 / 生成可复现。
- ``ByteTokenizer``：无损 UTF-8 字节级，ID 0..255 为字节，256=BOS、257=EOS。
- ``tokenizer_from_meta``：按检查点 ``extra.tokenizer`` 元数据恢复分词器，
  无字符表时回退到字节分词器。

本包不依赖项目内任何其他模块（无循环导入）。
"""
from __future__ import annotations

# 字节分词器常量：ID 0..255 为 UTF-8 字节；256=BOS，257=EOS。
BOS_ID = 256
EOS_ID = 257
VOCAB_SIZE = 258


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer. Simple and robust, but not compression-efficient."""
    vocab_size = VOCAB_SIZE
    bos_id = BOS_ID
    eos_id = EOS_ID

    def to_meta(self):
        return {"type": "byte"}

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        raw = bytearray()
        for item in ids:
            i = int(item)
            if i in (BOS_ID, EOS_ID) and skip_special:
                continue
            if 0 <= i <= 255:
                raw.append(i)
            elif not skip_special:
                raw.extend({BOS_ID: b"<BOS>", EOS_ID: b"<EOS>"}.get(i, f"<tok:{i}>".encode()))
        return raw.decode("utf-8", errors="replace")


class CharTokenizer:
    """Character-level tokenizer built from the training text.

    IDs: 0 = BOS, 1 = EOS, 2 = UNK, 3.. = characters in sorted order.
    The character table is stored in the checkpoint, so generation can be reproduced after resume.
    """
    BOS, EOS, UNK = 0, 1, 2

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i + 3 for i, c in enumerate(self.chars)}
        self.itos = {i + 3: c for i, c in enumerate(self.chars)}
        self.vocab_size = 3 + len(self.chars)
        self.bos_id, self.eos_id = self.BOS, self.EOS

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(c, self.UNK) for c in text]
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        names = {self.BOS: "<BOS>", self.EOS: "<EOS>", self.UNK: "<UNK>"}
        out = []
        for item in ids:
            i = int(item)
            if i < 3:
                if not skip_special:
                    out.append(names[i])
                continue
            out.append(self.itos.get(i, f"<tok:{i}>"))
        return "".join(out)

    def to_meta(self):
        return {"type": "char", "chars": self.chars}


def tokenizer_from_meta(meta: dict):
    """Rebuild the tokenizer a checkpoint was trained with (byte tokenizer if none is stored)."""
    tok = (meta.get("extra") or {}).get("tokenizer")
    if tok and tok.get("type") == "char":
        return CharTokenizer(tok["chars"])
    return ByteTokenizer()
