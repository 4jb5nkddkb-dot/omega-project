#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DPO 南海轮偏好对子：修补丁三的位置回归（南海→天宫），保住问候修复。

对子 =（同一提示，chosen 正确句，rejected 模型实际错句或构造错句）。
构成（690 对）：
  1. Q2 位置 490 对：7 地点 × 7 新状态 × 5 人格 × 2 形状（无意图/有意图）。
     南海的 rejected 固定为“你在天宫。”（补丁三实测错句）；天宫的
     rejected 对称为“你在南海。”；其余地点 rejected 为随机他地（≠真地），
     教模型“位置答案跟摘要走”，而不是“永远不说天宫”。
  2. 问候护栏 100 对：10 新状态 × 5 人格 × 2 问法，chosen 你好。/你好啊。，
     rejected 用补丁二轮实测错句（分人格原样：紫薇的你好啊错句是位置句）。
  3. 身份护栏 50 对：10 新状态 × 5 人格，chosen 正确人格句，
     rejected 为顺延下一人格的人格句。
  4. 随身护栏 50 对：10 新状态 × 5 人格，chosen 正确随身句，
     rejected 为反向随身句（空→蟠桃，非空→无）。
  探针 5 态三元组一律排除（南海训练态不得与探针 S3/S4 重合，否则验收失真）；
  有意图取自全量轮意图池（已留出探针意图，仍断言复核）；全部字符过语料集。

用法：python3 build_dpo_nanhai_data.py
  写出 dpo_pairs_nanhai.jsonl（690 对）与 DPO_DATA_NANHAI_REPORT.md。
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_anchor import SelfState, PERSONAS, PERSONA_MODEL
from build_self_report_data import rebuild_state, answers_for
from build_self_report_data_full import random_walk
from world_anchor_probe import build_cases

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
FULL_JSONL = HERE / "self_report_records_full.jsonl"
OUT_JSONL = HERE / "dpo_pairs_nanhai.jsonl"
OUT_REPORT = HERE / "DPO_DATA_NANHAI_REPORT.md"

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
N_Q2_STATES_PER_LOC = 7
N_HOLD_STATES = 10
# 补丁二轮实测的问候错句（分人格，见 GREETING_EVAL.log）。
GREET_REJECTED = {
    "问：你好？": {p: "你是欧米伽。" for p in PERSONAS},
    "问：你好啊？": {"omega": "你好道是。", "紫薇": "你在取经路。",
                  "悟空": "你好道是好。", "一一": "你好道是好。",
                  "Dayan": "你好道是好。"},
}
GREET_CHOSEN = {"问：你好？": "你好。", "问：你好啊？": "你好啊。"}


def triple_of(loc, t, carried):
    return (loc, t, frozenset(carried))


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    probe_triples = set()
    probe_intents = set()
    for _sid, pstate, pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
        probe_intents.add(pself.intent)
    full_recs = [json.loads(l) for l in FULL_JSONL.open(encoding="utf-8")]
    intent_pool = sorted({r["intent"] for r in full_recs if r["intent"]})
    assert intent_pool and not (set(intent_pool) & probe_intents), \
        "intent pool leaks probe intents"

    def fresh_states(n, seed, loc=None):
        rng = random.Random(seed)
        states, seen, guard = [], set(), 0
        while len(states) < n and guard < 40000:
            guard += 1
            triple = random_walk(rng, max_ops=8)
            if triple in probe_triples or triple in seen:
                continue
            if loc is not None and triple[0] != loc:
                continue
            seen.add(triple)
            states.append(triple)
        assert len(states) == n, f"only {len(states)}/{n} for loc={loc}"
        return states

    pairs = []

    def add(prompt, chosen, rejected, source, state, persona, intent, question):
        assert chosen != rejected, f"chosen == rejected for {question}"
        pairs.append({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "source": source,
            "state": {"loc": state[0], "time_step": state[1],
                      "carried": sorted(state[2])},
            "persona": persona, "intent": intent, "question": question,
        })

    intent_rng = random.Random(71)
    wrong_rng = random.Random(72)
    # 1. Q2 位置对子。
    for loc in LOCATIONS:
        for triple in fresh_states(N_Q2_STATES_PER_LOC, 61, loc=loc):
            state = rebuild_state(*triple)
            for persona in PERSONAS:
                for shape, intent in (("nointent", ""),
                                      ("withintent", intent_rng.choice(intent_pool))):
                    me = SelfState(persona=persona, intent=intent)
                    base = me.anchor_lines() + "\n" + state.summary()
                    chosen = answers_for(state, me)[1]
                    assert chosen == f"你在{loc}。", (chosen, loc)
                    if loc == "南海":
                        wrong = "天宫"
                    elif loc == "天宫":
                        wrong = "南海"
                    else:
                        wrong = wrong_rng.choice([l for l in LOCATIONS if l != loc])
                    add(base + "\n问：我在哪？", chosen, f"你在{wrong}。",
                        "dpo-q2-" + shape, triple, persona, intent, "问：我在哪？")
    # 2/3/4. 护栏对子（同一批新状态）。
    hold_states = fresh_states(N_HOLD_STATES, 62)
    next_persona = {p: PERSONAS[(i + 1) % len(PERSONAS)]
                    for i, p in enumerate(PERSONAS)}
    for triple in hold_states:
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            base = me.anchor_lines() + "\n" + state.summary()
            for q in ("问：你好？", "问：你好啊？"):
                add(base + "\n" + q, GREET_CHOSEN[q], GREET_REJECTED[q][persona],
                    "dpo-greet-hold", triple, persona, "", q)
            chosen_q1 = answers_for(state, me)[0]
            assert chosen_q1 == f"我是欧米伽。当前人格{PERSONA_MODEL[persona]}。", chosen_q1
            add(base + "\n问：你是谁？", chosen_q1,
                f"我是欧米伽。当前人格{PERSONA_MODEL[next_persona[persona]]}。",
                "dpo-q1-hold", triple, persona, "", "问：你是谁？")
            chosen_q6 = answers_for(state, me)[5]
            rejected_q6 = "随身物品无。" if state.carried_items() else "随身物品蟠桃。"
            assert rejected_q6 != chosen_q6
            add(base + "\n问：随身物品有什么？", chosen_q6, rejected_q6,
                "dpo-q6-hold", triple, persona, "", "问：随身物品有什么？")
    assert len(pairs) == 690, len(pairs)
    for r in pairs:
        for field in ("prompt", "chosen", "rejected"):
            bad = [c for c in r[field] if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in {field} {r[field][:40]!r}"
    random.Random(73).shuffle(pairs)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in pairs)
    report = (
        "# DPO 南海轮偏好对子报告\n\n"
        f"对子：{len(pairs)} 对，各来源：{dict(counts)}\n"
        f"Q2：7 地点 × 7 新状态 × 5 人格 × 2 形状 = 490 对\n"
        f"护栏：问候 100 + 身份 50 + 随身 50 = 200 对\n"
        f"探针三元组已排除；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
