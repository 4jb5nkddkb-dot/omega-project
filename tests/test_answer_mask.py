"""答案区 loss 掩码测试（先读后答，第 6 块开训前接线）。

测三件事（都不依赖模型，只依赖分词器 + 数据集 + loss）：
1. domain_texts(return_answer_masks=True) 的世界域掩码与逐字符
   长度一致，且 365 条 read_drills 记录都在世界域里。
2. make_batch_fn(answer_masks=...) 返回四元组；世界域行掩码
   只留答案区 token（每行 >0 个），非世界域行全 True。
3. masked_loss 只在掩码位置上算平均交叉熵（与手工重算一致），
   且与全序列 loss 不同。
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tokenizer import CharTokenizer
from training.dataset import (WORLD_RECORD_FILES, domain_texts,
                              live_corpus_paths, make_batch_fn)
from training.loss import cross_entropy_loss, masked_loss


def _tok():
    vocab = json.loads((Path(__file__).resolve().parent.parent
                        / "data" / "processed" / "generalist_vocab.json").read_text())
    return CharTokenizer(vocab["chars"])


def _drill_count():
    n = 0
    for line in (Path(__file__).resolve().parent.parent
                 / "data" / "processed" / "read_drills.jsonl").read_text(
                     encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            assert rec["text"][rec["answer_start"]:] == rec["answer"]
            n += 1
    return n


def _chunk7_count():
    n = 0
    for line in (Path(__file__).resolve().parent.parent
                 / "data" / "processed" / "chunk7_drills.jsonl").read_text(
                     encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            assert rec["text"][rec["answer_start"]:] == rec["answer"]
            n += 1
    return n


def _chunk8_count():
    n = 0
    for line in (Path(__file__).resolve().parent.parent
                 / "data" / "processed" / "chunk8_drills.jsonl").read_text(
                     encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            assert rec["text"][rec["answer_start"]:] == rec["answer"]
            n += 1
    return n


def _chunk9_records():
    recs = []
    for line in (Path(__file__).resolve().parent.parent
                 / "data" / "processed" / "chunk9_drills.jsonl").read_text(
                     encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            assert rec["text"][rec["answer_start"]:] == rec["answer"]
            recs.append(rec)
    return recs


def test_world_mask_aligns_and_includes_drills():
    assert "read_drills.jsonl" in WORLD_RECORD_FILES
    assert _drill_count() == 365
    assert "chunk7_drills.jsonl" in WORLD_RECORD_FILES
    assert _chunk7_count() == 135
    assert "chunk8_drills.jsonl" in WORLD_RECORD_FILES
    assert _chunk8_count() == 133
    assert "chunk9_drills.jsonl" in WORLD_RECORD_FILES
    assert len(_chunk9_records()) == 25
    tok = _tok()
    ids, masks = domain_texts(tok, paths=live_corpus_paths(),
                              return_answer_masks=True)
    assert set(masks) == {"世界"}
    assert len(masks["世界"]) == len(ids["世界"])
    assert masks["世界"].dtype == torch.bool
    # 答案区约占世界域一半（旧记录整条计 loss，QA 记录只计答案区）
    frac = float(masks["世界"].float().mean())
    assert 0.3 < frac < 0.7, f"answer fraction {frac:.3f} out of range"


def test_batch_fn_returns_answer_mask():
    tok = _tok()
    ids, masks = domain_texts(tok, paths=live_corpus_paths(),
                              return_answer_masks=True)
    batch_fn = make_batch_fn(ids, seq_len=128, batch_size=8, seed=7,
                             answer_masks=masks)
    rng = np.random.default_rng(0)
    x, y, gt, lm = batch_fn(rng)
    assert x.shape == y.shape == gt.shape == lm.shape == (8, 128)
    assert lm.dtype == torch.bool
    world_rows = (gt[:, 0] == 0.0)
    assert int(world_rows.sum()) == 1  # 每批恰 1 行世界域
    assert bool(lm[~world_rows].all()), "非世界域行掩码应全 True"
    assert int(lm[world_rows].sum()) > 0, "世界域行掩码全 False"
    assert int(lm[world_rows].sum()) < 128, "世界域行掩码全 True（未掩码）"


def test_masked_loss_only_counts_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    mask = torch.tensor([[True, False, True, False, False],
                         [False, False, True, True, False]])
    got = masked_loss(logits, targets, mask, label_smoothing=0.1)
    manual = F.cross_entropy(
        logits.reshape(-1, 11)[mask.reshape(-1)],
        targets.reshape(-1)[mask.reshape(-1)], label_smoothing=0.1)
    assert abs(float(got) - float(manual)) < 1e-6
    full = cross_entropy_loss(logits, targets, label_smoothing=0.1)
    assert abs(float(got) - float(full)) > 1e-6
    with pytest.raises(ValueError):
        masked_loss(logits, targets, torch.zeros_like(mask))


def test_chunk9_memory_prefix_format():
    """第 9 块：25 条全带记忆前缀，格式与 evaluation/sidecar.py
    的 format_examples 一致（每行 = 提问 + 正确回复，\\n 连接，
    末尾带 \\n）；S0 的 7 条覆盖环在各步的真实缓存前缀；其余
    6 地各 3 条（Q0/Q2/Q3，前缀 [Q0,Q1]）。裸 Q0/Q1/Q2/Q3 为 0 条。"""
    recs = _chunk9_records()
    assert len(recs) == 25
    questions = ["你在什么地方。", "在场人物有谁。",
                 "随身物品有什么。", "现在是白天还是夜晚。"]
    for r in recs:
        prefix = r["memory_prefix"]
        assert prefix, "第 9 块记录必须带记忆前缀"
        assert r["text"].startswith(prefix), "记录文本必须以记忆前缀开头"
        assert prefix.endswith("\n")
        for line in prefix.rstrip("\n").split("\n"):
            assert any(line.startswith(q) for q in questions), \
                f"前缀行必须以提问开头：{line!r}"
            assert line.endswith("。"), f"前缀行必须以句号结尾：{line!r}"
            assert len(line) <= 64, f"前缀行超 64 字（sidecar 截断上限）：{line!r}"
    s0 = [r for r in recs if r["state"]["loc"] == "取经路"
          and r["state"]["time_step"] == 0]
    assert len(s0) == 7
    # S0 的 7 条：Q0×1、Q1×2、Q2×2、Q3×2，且前缀覆盖环的缓存态。
    from collections import Counter
    assert Counter(r["question"] for r in s0) == Counter({
        "你在什么地方。": 1, "在场人物有谁。": 2,
        "随身物品有什么。": 2, "现在是白天还是夜晚。": 2})
    # 环第 5/9 步的 Q0 前缀必须是 [Q1,Q2,Q3] 三行。
    q0_s0 = [r for r in s0 if r["question"] == "你在什么地方。"][0]
    assert q0_s0["memory_prefix"].count("\n") == 3
    assert q0_s0["memory_prefix"].startswith("在场人物有谁。")
    others = [r for r in recs if r["state"]["loc"] != "取经路"]
    assert len(others) == 18
    for loc in ["花果山", "水帘洞", "高老庄", "流沙河", "南海", "天宫"]:
        sub = [r for r in others if r["state"]["loc"] == loc]
        assert len(sub) == 3
        assert {r["question"] for r in sub} == {
            "你在什么地方。", "随身物品有什么。", "现在是白天还是夜晚。"}
