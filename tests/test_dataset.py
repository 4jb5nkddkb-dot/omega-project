"""数据集测试：用 staging 的 NumPy 参考实现（training/dataset.py）验证
live 语料加载与分层采样。

本测试可独立运行（只依赖 numpy + tokenizer，不依赖 torch / model /
training 的 PyTorch 移植）。它测的是参考实现本身；CONTRACT.md 的
torch 版 GeneralistDataset / get_dataloader 由 Agent B 移植，届时
本文件追加 torch 版断言（batch 为 torch 张量、DataLoader 能迭代）。

live 语料 = 旧仓库的实时文件（~/workspace/dual_sft + ~/workspace/omega-chat），
只读；新布局 data/ 的等价性由 data/processed/MANIFEST.md 的计数保证。
"""
import numpy as np
import pytest

from tokenizer import CharTokenizer
from training.dataset import (
    GATE_LEFT_DOMAINS,
    GATE_RIGHT_DOMAINS,
    RATIOS,
    _batch_counts,
    domain_texts,
    live_corpus_paths,
    make_batch_fn,
)
import json
from pathlib import Path


def _tok():
    vocab = json.loads((Path(__file__).resolve().parent.parent
                        / "data" / "processed" / "generalist_vocab.json").read_text())
    return CharTokenizer(vocab["chars"])


def test_ratios_and_gate_domains():
    assert [n for n, _ in RATIOS] == ["经典", "代码", "对话", "常识", "推理", "世界"]
    assert sum(w for _, w in RATIOS) == pytest.approx(1.0)
    assert set(GATE_LEFT_DOMAINS) == {"经典", "代码", "对话", "常识", "推理"}
    assert set(GATE_RIGHT_DOMAINS) == {"世界"}


def test_batch_counts_largest_remainder():
    # batch=8：0.25/0.20/0.10/0.15/0.15/0.15 -> 2/2/1/1/1/1
    assert list(_batch_counts([w for _, w in RATIOS], 8)) == [2, 2, 1, 1, 1, 1]


def test_live_corpus_loads_and_batch_shapes():
    tok = _tok()
    ids = domain_texts(tok, paths=live_corpus_paths())
    assert set(ids) == {"经典", "代码", "对话", "常识", "推理", "世界"}
    for name, arr in ids.items():
        assert len(arr) > 200, f"域 {name} 太短"
    batch_fn = make_batch_fn(ids, seq_len=128, batch_size=8, seed=7)
    rng = np.random.default_rng(0)
    inputs, targets, gate = batch_fn(rng)
    assert inputs.shape == (8, 128)
    assert targets.shape == (8, 128)
    assert gate.shape == (8, 128)
    # targets 是 inputs 右移一位（同一窗口内）
    assert np.array_equal(targets[:, :-1], inputs[:, 1:])
    # 门目标：左支域 1.0，世界域 0.0；每批 7 行左 / 1 行右
    assert set(np.unique(gate)) <= {0.0, 1.0}
    assert int((gate[:, 0] == 1.0).sum()) == 7
    assert int((gate[:, 0] == 0.0).sum()) == 1
    # 同一批内每行门值恒定（每序列一个门监督目标）
    assert all(len(np.unique(row)) == 1 for row in gate)
    # token id 在词表内
    assert inputs.min() >= 0 and inputs.max() < tok.vocab_size
