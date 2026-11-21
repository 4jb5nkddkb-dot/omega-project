#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打招呼补丁数据：修锚点下打招呼翻人称（你是欧米伽）。

病因（推测，已实测佐证）：协议问候记录（问：你好？→你好。）训练时
没有锚点行 + 摘要前缀；锚点格式下一问候，模型就滑向身份句式，
还翻成人称（你是欧米伽）。修法：把问候写进锚点格式，5 个人格、
有意图/无意图双形状全铺，让“问候→你好。”与锚点无关。

构成（1200 条）：
  1. 问候 600 条：30 个工具真实走出的新状态 × 5 人格 × 2 问法
     （你好/你好啊）× 2 形状（无意图/有意图同态同答案）。
  2. 回归 600 条（沿用补丁二轮原样复用，防止补丁带偏旧线）：
     无意图 Q1 150 + 无意图 Q5 150 + 随身无意图 100 + 随身有意图 100
     + 有意图锚定 100。
  新状态排除探针三元组、一轮 50 态、二轮 40 态；有意图取自全量轮
  意图池（已留出探针意图，仍断言复核）；新记录全部字符在语料字符集内。

用法：python3 build_self_report_greeting_data.py
  写出 self_report_records_greeting.jsonl（1200 条）与
  SELF_REPORT_DATA_GREETING_REPORT.md。
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_state as ws
from world_anchor import SelfState, PERSONAS
from build_self_report_data import rebuild_state
from build_self_report_data_full import random_walk
from world_anchor_probe import build_cases

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
PATCH2_JSONL = HERE / "self_report_records_patch2.jsonl"
FULL_JSONL = HERE / "self_report_records_full.jsonl"
OUT_JSONL = HERE / "self_report_records_greeting.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_GREETING_REPORT.md"

N_NEW_STATES = 30
FORMS = (("问：你好？", "你好。"), ("问：你好啊？", "你好啊。"))


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    probe_triples = set()
    probe_intents = set()
    for _sid, pstate, pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
        probe_intents.add(pself.intent)

    patch2 = [json.loads(l) for l in PATCH2_JSONL.open(encoding="utf-8")]
    patch2_triples = set()
    for r in patch2:
        st = r["state"]
        patch2_triples.add((st["loc"], st["time_step"],
                            frozenset(st["carried"])))
    full_recs = [json.loads(l) for l in FULL_JSONL.open(encoding="utf-8")]
    intent_pool = sorted({r["intent"] for r in full_recs if r["intent"]})
    assert intent_pool and not (set(intent_pool) & probe_intents), \
        "intent pool leaks probe intents"

    rng = random.Random(51)
    states = []
    seen = set()
    guard = 0
    while len(states) < N_NEW_STATES and guard < 20000:
        guard += 1
        triple = random_walk(rng, max_ops=8)
        if triple in probe_triples or triple in patch2_triples or triple in seen:
            continue
        seen.add(triple)
        states.append(triple)
    assert len(states) == N_NEW_STATES, f"only collected {len(states)} states"

    intent_rng = random.Random(31)
    records = []
    for (loc, t, carried) in states:
        state = rebuild_state(loc, t, carried)
        for persona in PERSONAS:
            for shape, me in (("nointent", SelfState(persona=persona, intent="")),
                              ("withintent", SelfState(
                                  persona=persona,
                                  intent=intent_rng.choice(intent_pool)))):
                base = me.anchor_lines() + "\n" + state.summary()
                for q, ans in FORMS:
                    prompt = base + "\n" + q
                    records.append({
                        "text": prompt + ans, "summary": prompt,
                        "answer_start": len(prompt),
                        "state": {"loc": loc, "time_step": t,
                                  "carried": sorted(carried)},
                        "persona": me.persona, "intent": me.intent,
                        "question": q, "answer": ans,
                        "source": "greet-" + shape,
                    })
    assert len(records) == 600, len(records)

    # 回归：沿用补丁二轮原样复用。
    by_source = {}
    for r in patch2:
        by_source.setdefault(r["source"], []).append(r)
    sampler = random.Random(23)
    picks = [("patch-q1-nointent", 150), ("patch-q5-nointent", 150),
             ("patch2-q6-nointent", 100), ("patch2-q6-withintent", 100),
             ("anchor-with-intent", 100)]
    for src, n in picks:
        pool = by_source.get(src, [])
        assert len(pool) >= n, (src, len(pool), n)
        for r in sampler.sample(pool, n):
            r = dict(r)
            r["source"] = "regress-" + src
            records.append(r)
    assert len(records) == 1200, len(records)
    for r in records:
        if r["source"].startswith("greet-"):
            bad = [c for c in r["text"] if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in {r['text'][:40]!r}"
    random.Random(29).shuffle(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in records)
    report = (
        "# 打招呼补丁数据报告\n\n"
        f"记录：{len(records)} 条，各来源：{dict(counts)}\n"
        f"问候：{N_NEW_STATES} 新状态 × 5 人格 × 2 问法 × 2 形状 = 600 条\n"
        f"回归：补丁二轮原样复用 600 条（Q1/Q5 无意图各 150，随身各 100，锚定 100）\n"
        f"新状态排除探针三元组与补丁二轮状态；新记录全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
