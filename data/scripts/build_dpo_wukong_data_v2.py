#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DPO 第二轮 v2 偏好对子：v1 配比再平衡（用户 2026-10-30 点“改数据再试”）。

v1 教训（三个试跑一致）：Q1 身份 240 对占 40%，长答案梯度
压倒位置护栏，DPO1 里本就脆弱的南海竞争（南只比铭高 2.4 分，
铭埋伏第三）被掀翻到铭；beta 0.1/0.3、lr 1e-4/2e-5 全翻，
试跑配比里南海正对子 22 对反对子 4 对仍翻——是方向问题，
不是步长或抽样问题。Q1 修复本身很强（最小步长、val 零漂移
都修好 5/5），所以可以削 Q1 保南海。

v2 改动（其余结构与 v1 完全一致，新种子取全新状态）：
  1. Q1 身份 240 → 120 对：10 新状态 × 5 人格 × 2 形状＝100
     ＋ 悟空双重对比 20 对（Q1 占比 40% → 23%）。
  2. Q2 南海护栏 100 → 150 对：15 新状态 × 5 人格 × 2 形状
     （南海占比 17% → 28%；Q1:南海 从 2.4:1 → 0.8:1）。
  3. 其余不动：他地 60、问候 100、随身 50、无意图意图 50。
  总计 530 对。探针 5 态三元组一律排除；全部字符过语料集。

用法：python3 build_dpo_wukong_data_v2.py
  写出 dpo_pairs_wukong_v2.jsonl（530 对）与 DPO_DATA_WUKONG_V2_REPORT.md。
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
OUT_JSONL = HERE / "dpo_pairs_wukong_v2.jsonl"
OUT_REPORT = HERE / "DPO_DATA_WUKONG_V2_REPORT.md"

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
N_Q1_STATES = 10
N_NANHAI_STATES = 15
N_HOLD_STATES = 10
GREET_REJECTED = {
    "问：你好？": {p: "你是欧米伽。" for p in PERSONAS},
    "问：你好啊？": {"omega": "你好道是。", "紫薇": "你在取经路。",
                  "悟空": "你好道是好。", "一一": "你好道是好。",
                  "Dayan": "你好道是好。"},
}
GREET_CHOSEN = {"问：你好？": "你好。", "问：你好啊？": "你好啊。"}


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

    def add(prompt, chosen, rejected, source, triple, persona, intent, question):
        assert chosen != rejected, f"chosen == rejected for {question}"
        pairs.append({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "source": source,
            "state": {"loc": triple[0], "time_step": triple[1],
                      "carried": sorted(triple[2])},
            "persona": persona, "intent": intent, "question": question,
        })

    intent_rng = random.Random(91)
    wrong_rng = random.Random(92)
    next_persona = {p: PERSONAS[(i + 1) % len(PERSONAS)]
                    for i, p in enumerate(PERSONAS)}
    # 1. Q1 身份对子（v2：10 态，削半）。
    for triple in fresh_states(N_Q1_STATES, 91):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            for intent in ("", intent_rng.choice(intent_pool)):
                me = SelfState(persona=persona, intent=intent)
                base = me.anchor_lines() + "\n" + state.summary()
                chosen = answers_for(state, me)[0]
                assert chosen == f"我是欧米伽。当前人格{PERSONA_MODEL[persona]}。", chosen
                if persona == "omega":
                    rejected = f"我是欧米伽。当前人格{PERSONA_MODEL[next_persona[persona]]}。"
                else:
                    rejected = "我是欧米伽。当前人格欧米伽。"
                add(base + "\n问：你是谁？", chosen, rejected,
                    "dpo2v2-q1-default", triple, persona, intent, "问：你是谁？")
                if persona == "悟空":
                    add(base + "\n问：你是谁？", chosen,
                        f"我是欧米伽。当前人格{PERSONA_MODEL[next_persona[persona]]}。",
                        "dpo2v2-q1-wukong", triple, persona, intent, "问：你是谁？")
    # 2. Q2 位置护栏（v2：南海 15 态，加护）。
    q2_states = [(loc, t) for loc, t in
                 [("南海", s) for s in fresh_states(N_NANHAI_STATES, 93, loc="南海")]]
    for loc in LOCATIONS:
        if loc == "南海":
            continue
        q2_states.append((loc, fresh_states(1, 94, loc=loc)[0]))
    for loc, triple in q2_states:
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            for intent in ("", intent_rng.choice(intent_pool)):
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
                    "dpo2v2-q2-hold", triple, persona, intent, "问：我在哪？")
    # 3/4/5. 问候、随身、无意图意图护栏（同一批新状态）。
    for triple in fresh_states(N_HOLD_STATES, 95):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            base = me.anchor_lines() + "\n" + state.summary()
            for q in ("问：你好？", "问：你好啊？"):
                add(base + "\n" + q, GREET_CHOSEN[q], GREET_REJECTED[q][persona],
                    "dpo2v2-greet-hold", triple, persona, "", q)
            chosen_q6 = answers_for(state, me)[5]
            rejected_q6 = "随身物品无。" if state.carried_items() else "随身物品蟠桃。"
            assert rejected_q6 != chosen_q6
            add(base + "\n问：随身物品有什么？", chosen_q6, rejected_q6,
                "dpo2v2-q6-hold", triple, persona, "", "问：随身物品有什么？")
            add(base + "\n问：我要干什么？", "我没有意图。", "我要找金箍棒。",
                "dpo2v2-q5-hold", triple, persona, "", "问：我要干什么？")
    assert len(pairs) == 530, len(pairs)
    for r in pairs:
        for field in ("prompt", "chosen", "rejected"):
            bad = [c for c in r[field] if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in {field} {r[field][:40]!r}"
    random.Random(96).shuffle(pairs)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in pairs)
    report = (
        "# DPO 第二轮 v2（悟空·再平衡）偏好对子报告\n\n"
        f"对子：{len(pairs)} 对，各来源：{dict(counts)}\n"
        f"Q1 身份：10 新状态 × 5 人格 × 2 形状 = 100 对（被拒默认句）"
        f"＋ 悟空双重对比 20 对（v1 为 240 对，削半）\n"
        f"Q2 护栏：南海 15 态 × 5 人格 × 2 形状＝150（v1 为 100）"
        f"＋ 他地 6 态 × 5 × 2＝60\n"
        f"护栏：问候 100 ＋ 随身 50 ＋ 无意图意图 50（与 v1 一致）\n"
        f"探针三元组已排除；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
