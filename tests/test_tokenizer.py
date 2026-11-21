"""分词器测试：CharTokenizer 在 generalist_vocab.json 上的往返。

不依赖 torch，可独立运行。
"""
import json
from pathlib import Path

from tokenizer import ByteTokenizer, CharTokenizer, tokenizer_from_meta

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "processed" / "generalist_vocab.json"


def _load_tok():
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    assert vocab["type"] == "char"
    return CharTokenizer(vocab["chars"]), vocab


def test_vocab_size_is_4622():
    tok, vocab = _load_tok()
    assert len(vocab["chars"]) == 4619
    assert tok.vocab_size == 4622  # 4619 字 + BOS/EOS/UNK


def test_special_ids():
    tok, _ = _load_tok()
    assert (tok.BOS, tok.EOS, tok.UNK) == (0, 1, 2)
    assert tok.bos_id == 0 and tok.eos_id == 1


def test_round_trip_chinese_and_code():
    tok, _ = _load_tok()
    for text in [
        "孙悟空来到花果山。",
        "def add(a, b):\n    return a + b  # 加法",
        "问：三加五等于几？答案是八。",
        "你在花果山。现在是白天。你带着金箍棒。",
    ]:
        assert tok.decode(tok.encode(text)) == text


def test_bos_eos_flags():
    tok, _ = _load_tok()
    ids = tok.encode("你好", add_bos=True, add_eos=True)
    assert ids[0] == 0 and ids[-1] == 1
    assert tok.decode(ids) == "你好"  # 默认跳过特殊符


def test_unknown_char_is_unk():
    tok, _ = _load_tok()
    # 词表外字符（此处用一个极不可能在 4619 字里的字符）应编为 UNK=2。
    # 若未来词表扩充导致该字符入表，本测试会失败——那时换一个字符。
    probe = "\U0001F600"  # 😀
    if probe in tok.stoi:
        import pytest
        pytest.skip("探针字符已入词表，换一个词表外字符")
    assert tok.encode(probe) == [2]


def test_tokenizer_from_meta_round_trip():
    tok, vocab = _load_tok()
    rebuilt = tokenizer_from_meta({"extra": {"tokenizer": vocab}})
    assert isinstance(rebuilt, CharTokenizer)
    assert rebuilt.chars == tok.chars
    assert rebuilt.encode("花果山") == tok.encode("花果山")
    # 无字符表时回退到字节分词器
    assert isinstance(tokenizer_from_meta({}), ByteTokenizer)
