#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补丁二轮数据：从全量重补（绝不从一轮补丁起），把随身补上。

一轮教训：一轮补丁治好了无意图 Q1/Q5，却把随身带崩了
（标记随身 5/5→1/5，无意图随身真实 25/25→10/25）。
推测原因：随身在补丁池里只靠锚定样本带了很少量，
300 步全学习率把列表生成带漂了（推测，未证实）。

二轮构成（1200 条，标记格式不变）：
  1. 沿用一轮无意图 Q1 250 条 + 无意图 Q5 250 条（原样复用，
     防止二轮把一轮的修复丢掉——锚定样本里没有无意图记录）。
  2. 新增随身 400 条：40 个工具真实走出的新状态 × 5 人格 × 2 种形状
     ——无意图随身 200 条 + 有意图随身 200 条（同态双形状，
     教模型随身答案只取决于状态，与意图句在不在无关）。
  3. 有意图锚定样本 300 条（全量轮随机抽，种子固定）。
  新状态排除探针三元组与一轮的 50 态；有意图的意图取自全量轮
  意图池（已留出探针意图，仍断言复核）；随身答案按探针评分自验。

用法：python3 build_self_report_patch_data2.py
  写出 self_report_records_patch2.jsonl（1200 条）与
  SELF_REPORT_DATA_PATCH2_REPORT.md。
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_state as ws
from world_anchor import SelfState, PERSONAS
from build_self_report_data import rebuild_state, answers_for
from build_self_report_data_full import random_walk
from world_anchor_probe import build_cases, score as probe_score

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
PATCH1_JSONL = HERE / "self_report_records_patch.jsonl"
FULL_JSONL = HERE / "self_report_records_full.jsonl"
OUT_JSONL = HERE / "self_report_records_patch2.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_PATCH2_REPORT.md"

N_NEW_STATES = 40


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    probe_triples = set()
    probe_intents = set()
    for _sid, pstate, pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
        probe_intents.add(pself.intent)

    # 1. 沿用一轮无意图 Q1/Q5（原样复用）。
    patch1 = [json.loads(l) for l in PATCH1_JSONL.open(encoding="utf-8")]
    reused_q1 = [r for r in patch1 if r["source"] == "patch-q1-nointent"]
    reused_q5 = [r for r in patch1 if r["source"] == "patch-q5-nointent"]
    assert len(reused_q1) == 250 and len(reused_q5) == 250, \
        (len(reused_q1), len(reused_q5))
    patch1_triples = set()
    for r in patch1:
        if r["source"].startswith("patch"):
            st = r["state"]
            patch1_triples.add((st["loc"], st["time_step"],
                                frozenset(st["carried"])))

    # 2. 新增随身：40 新状态 × 5 人格 × 有意图/无意图。
    full_recs = [json.loads(l) for l in FULL_JSONL.open(encoding="utf-8")]
    intent_pool = sorted({r["intent"] for r in full_recs if r["intent"]})
    assert intent_pool and not (set(intent_pool) & probe_intents), \
        "intent pool leaks probe intents"
    rng = random.Random(41)
    states = []
    seen = set()
    guard = 0
    while len(states) < N_NEW_STATES and guard < 20000:
        guard += 1
        triple = random_walk(rng, max_ops=8)
        if triple in probe_triples or triple in patch1_triples or triple in seen:
            continue
        seen.add(triple)
        states.append(triple)
    assert len(states) == N_NEW_STATES, f"only collected {len(states)} states"

    intent_rng = random.Random(11)
    records = list(reused_q1) + list(reused_q5)
    for (loc, t, carried) in states:
        state = rebuild_state(loc, t, carried)
        for persona in PERSONAS:
            # 无意图随身。
            me0 = SelfState(persona=persona, intent="")
            base0 = me0.anchor_lines() + "\n" + state.summary()
            prompt0 = base0 + "\n" + "问：随身物品有什么？"
            ans = answers_for(state, me0)[5]
            ok, note = probe_score(state, me0, 5, ans)
            assert ok, f"Q6 no-intent answer fails scoring: {note}"
            records.append({
                "text": prompt0 + ans, "summary": prompt0,
                "answer_start": len(prompt0),
                "state": {"loc": loc, "time_step": t,
                          "carried": sorted(carried)},
                "persona": me0.persona, "intent": "",
                "question": "问：随身物品有什么？", "answer": ans,
                "source": "patch2-q6-nointent",
            })
            # 有意图随身（同态同答案，意图句在锚点里）。
            intent = intent_rng.choice(intent_pool)
            me1 = SelfState(persona=persona, intent=intent)
            base1 = me1.anchor_lines() + "\n" + state.summary()
            prompt1 = base1 + "\n" + "问：随身物品有什么？"
            ans1 = answers_for(state, me1)[5]
            assert ans1 == ans, "Q6 answer must not depend on intent"
            ok, note = probe_score(state, me1, 5, ans1)
            assert ok, f"Q6 with-intent answer fails scoring: {note}"
            records.append({
                "text": prompt1 + ans1, "summary": prompt1,
                "answer_start": len(prompt1),
                "state": {"loc": loc, "time_step": t,
                          "carried": sorted(carried)},
                "persona": me1.persona, "intent": intent,
                "question": "问：随身物品有什么？", "answer": ans1,
                "source": "patch2-q6-withintent",
            })
    # 3. 有意图锚定样本 300 条。
    anchor = random.Random(17).sample(full_recs, 300)
    for r in anchor:
        r = dict(r)
        r["source"] = "anchor-with-intent"
        records.append(r)
    assert len(records) == 1200, len(records)
    for r in records:
        if r["source"].startswith("patch2"):
            bad = [c for c in r["text"] if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in {r['text'][:40]!r}"
    random.Random(13).shuffle(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    counts = Counter(r["source"] for r in records)
    report = (
        "# 补丁二轮数据报告（从全量重补）\n\n"
        f"记录：{len(records)} 条，各来源：{dict(counts)}\n"
        f"新状态：{N_NEW_STATES} 个（排除探针三元组与一轮 50 态）\n"
        f"随身：同态双形状（有意图/无意图各 200 条），答案与意图无关（已断言）\n"
        f"沿用：一轮无意图 Q1/Q5 原样复用 500 条，防止二轮丢修复\n"
        f"自验：随身答案过探针评分；新记录全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
