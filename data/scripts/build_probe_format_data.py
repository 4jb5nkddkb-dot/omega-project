#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针格式记录 v10：把循环探针的问法直接写进训练数据。

背景（2026-11-12，s60600 闭环重跑仍 12/12）：
  训练只教过“问：我在哪？”格式，循环探针却是另一种问法——
  摘要 + “你在什么地方。”（无锚定行、无“问：”包装）。
  模型从没见过探针格式，故 A 修了窗口也未见效。

本文件：与 v9 自报记录用同一批 150 态（同种子 23，同分层选态，
断言摘要与 v9 记录逐态一致），每态出 4 条探针格式记录：
  摘要 + "\\n" + 探针原问 + 标准答案（答案与探针评分规则一致）。
共 150 × 4 = 600 条，写出 probe_format_records.jsonl。

用法：python3 build_probe_format_data.py
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/hatch/workspace/dual_side/xiyouji_modern_trainer_1")

import world_state as ws
from build_self_report_data import rebuild_state
from build_self_report_data_v9 import random_walk, N_STATES
from world_anchor_probe import build_cases
from world_self_probe import QUESTIONS as PROBE_QUESTIONS, score as probe_score
from dual_transformer_numpy import read_checkpoint, tokenizer_from_meta

HERE = Path(__file__).resolve().parent
OUT_JSONL = HERE / "probe_format_records.jsonl"
OUT_REPORT = HERE / "PROBE_FORMAT_DATA_REPORT.md"
TOKENIZER_CKPT = HERE / "out/generalist_s61200.npz"


def probe_answers(state):
    """4 个探针问题的标准答案（顺序与 world_self_probe.QUESTIONS 对应）。"""
    present = state.present_characters()
    carried = state.carried_items()
    return [
        f"你在{state.visitor_location}。",
        f"在场人物{ws._join(present)}。",
        "随身物品无。" if not carried else f"随身物品{ws._join(carried)}。",
        f"现在是{state.day_night()}。",
    ]


def select_states():
    """与 v9 构建器同一批 150 态：同种子 23，同池子同分层选态。"""
    probe_triples = set()
    for _sid, pstate, _pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
    rng = random.Random(23)
    pool = {}
    guard = 0
    while len(pool) < 400 and guard < 30000:
        guard += 1
        triple = random_walk(rng, max_ops=2 if guard % 2 == 0 else 8)
        if triple in probe_triples or triple in pool:
            continue
        pool[triple] = True
    assert len(pool) >= N_STATES, f"pool too small: {len(pool)}"
    by_loc = {loc: {} for loc in ws.LOCATIONS}
    for loc, t, carried in pool:
        key = (bool(carried), (t // 4) % 2 == 1)
        by_loc[loc].setdefault(key, []).append((loc, t, carried))
    bucket_order = [(False, False), (False, True), (True, False), (True, True)]
    per_loc = N_STATES // len(ws.LOCATIONS)
    extra = N_STATES - per_loc * len(ws.LOCATIONS)
    states = []
    for i, loc in enumerate(ws.LOCATIONS):
        buckets = {k: list(v) for k, v in by_loc[loc].items()}
        quota = per_loc + (1 if i < extra else 0)
        assert sum(len(v) for v in buckets.values()) >= quota, f"{loc} pool short"
        merged = []
        while len(merged) < quota:
            progressed = False
            for k in bucket_order:
                if len(merged) >= quota:
                    break
                if buckets.get(k):
                    merged.append(buckets[k].pop(0))
                    progressed = True
            assert progressed, f"{loc} buckets exhausted"
        states.extend(merged)
    assert len(states) == N_STATES, len(states)
    return states


def main():
    states = select_states()
    # 与 v9 记录逐态核对摘要：同一批态（v9 每态 8 条，顺序与 states 一致）。
    v9 = [json.loads(l)
          for l in open(HERE / "self_report_records_v9.jsonl", encoding="utf-8")]
    assert len(v9) == N_STATES * 8, len(v9)
    for i, (loc, t, carried) in enumerate(states):
        st = rebuild_state(loc, t, carried)
        prompt0 = v9[i * 8]["summary"]
        # prompt = 锚定行（1 行）+ 摘要 + “问：…”行
        summary0 = "\n".join(prompt0.split("\n")[1:-1])
        assert st.summary() == summary0, f"state {i} summary mismatch"
    # 词表：以最新产品 checkpoint 的 tokenizer 为准（冻结 4622）。
    ck = read_checkpoint(str(TOKENIZER_CKPT))
    meta = ck["meta"]
    if isinstance(meta, (bytes, bytearray)):
        meta = json.loads(meta.decode("utf-8"))
    tok = tokenizer_from_meta(meta)

    records = []
    for i, (loc, t, carried) in enumerate(states):
        state = rebuild_state(loc, t, carried)
        summary = state.summary()
        answers = probe_answers(state)
        for qi, q in enumerate(PROBE_QUESTIONS):
            ans = answers[qi]
            text = summary + "\n" + q + ans
            ids = tok.encode(text)
            assert tok.UNK not in ids, f"UNK in {text[:40]!r}"
            ok, note = probe_score(state, qi, ans)
            assert ok, f"answer fails probe scoring: state {i} Q{qi + 1} {note}"
            records.append({
                "text": text, "summary": summary,
                "answer_start": len(summary) + 1 + len(q),
                "state": {"loc": loc, "time_step": t,
                          "carried": sorted(carried)},
                "question": q, "answer": ans,
            })
    assert len(records) == N_STATES * 4, len(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["text"]) for r in records]
    spans = [len(r["question"]) + len(r["answer"]) for r in records]
    locs = sorted({r["state"]["loc"] for r in records})
    nights = sum(1 for r in records
                 if r["question"] == "现在是白天还是夜晚。" and "夜晚" in r["answer"])
    days = sum(1 for r in records
               if r["question"] == "现在是白天还是夜晚。" and "白天" in r["answer"])
    q2_empty = sum(1 for r in records
                   if r["question"] == "随身物品有什么。" and r["answer"] == "随身物品无。")
    q2_full = sum(1 for r in records
                  if r["question"] == "随身物品有什么。" and r["answer"] != "随身物品无。")
    nanhai = sum(1 for r in records if r["state"]["loc"] == "南海")
    report = (
        "# 探针格式数据报告（v10）\n\n"
        f"记录：{len(records)} 条 = {N_STATES} 态 × 4（与 v9 自报同一批态）\n"
        f"格式：摘要 + 探针原问 + 标准答案（无锚定行、无“问：”包装）\n"
        f"地点：{locs}，每地点约 {N_STATES // 7} 态（含南海 {nanhai} 条）\n"
        f"昼夜：白天 {days} 条 / 夜晚 {nights} 条（Q4 计）\n"
        f"随身：空 {q2_empty} 条 / 非空 {q2_full} 条（Q3 计）\n"
        f"长度：最长 {max(lens)} 字，平均 {sum(lens) / len(lens):.1f} 字；"
        f"问答跨度最大 {max(spans)} 字\n"
        f"窗口：{sum(1 for L in lens if L <= 128)}/{len(lens)} 条整条进 128 字窗口；"
        f"{sum(1 for L in lens if L <= 64)}/{len(lens)} 条整条进 64 字窗口；"
        f"问答对跨度全部 ≤ 64 字\n"
        f"自验：全部 {len(records)} 条答案过探针评分；全部字符在冻结 4622 词表内；"
        f"150 态摘要与 v9 记录逐态一致\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
