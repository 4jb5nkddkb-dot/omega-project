#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DPO 第二轮偏好对子：修 DPO1 的悟空身份塌缩（无意图下悟空人格全答成欧米伽）。

上一轮教训：490 对位置 + 100 对问候都在教“人格不影响答案”，
50 对身份对子被淹没，模型在身份问题上也偷懒用默认人格。
本轮把身份对比提到主力，并把被拒句写成模型实际错句（欧米伽默认），
不再只用顺延人格。

构成（600 对）：
  1. Q1 身份 240 对：20 新状态 × 5 人格 × 2 形状（无意图/有意图）。
     每组一对：被拒 = 默认欧米伽句（omega 本人则被拒顺延紫薇句）；
     悟空每组再加一对：被拒 = 顺延一一句（双重对比，专治塌缩）。
  2. Q2 位置护栏 160 对：南海 10 新状态 × 5 人格 × 2 形状
     （被拒固定天宫）＋ 其余 6 地点各 1 新状态 × 5 人格 × 2 形状
     （被拒随机他地），保住第一轮的南海修复。
  3. 问候护栏 100 对：10 新状态 × 5 人格 × 2 问法，被拒用补丁二轮
     实测分人格错句原样。
  4. 随身护栏 50 对：10 新状态 × 5 人格，被拒为反向随身句（保答案正确；
     续写干净度不在 DPO 答案段射程内，本轮不追）。
  5. 无意图意图护栏 50 对：10 新状态 × 5 人格，无意图锚点，
     chosen 我没有意图。，被拒为已知意图句（保 Q5 诚实答案）。
  探针 5 态三元组一律排除；有意图取自全量轮意图池（已留出探针意图，
  仍断言复核）；全部字符过语料集。

用法：python3 build_dpo_wukong_data.py
  写出 dpo_pairs_wukong.jsonl（600 对）与 DPO_DATA_WUKONG_REPORT.md。
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
OUT_JSONL = HERE / "dpo_pairs_wukong.jsonl"
OUT_REPORT = HERE / "DPO_DATA_WUKONG_REPORT.md"

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
N_Q1_STATES = 20
N_NANHAI_STATES = 10
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

    intent_rng = random.Random(81)
    wrong_rng = random.Random(82)
    next_persona = {p: PERSONAS[(i + 1) % len(PERSONAS)]
                    for i, p in enumerate(PERSONAS)}
    # 1. Q1 身份对子。
    for triple in fresh_states(N_Q1_STATES, 81):
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
                    "dpo2-q1-default", triple, persona, intent, "问：你是谁？")
                if persona == "悟空":
                    add(base + "\n问：你是谁？", chosen,
                        f"我是欧米伽。当前人格{PERSONA_MODEL[next_persona[persona]]}。",
                        "dpo2-q1-wukong", triple, persona, intent, "问：你是谁？")
    # 2. Q2 位置护栏。
    q2_states = [(loc, t) for loc, t in
                 [("南海", s) for s in fresh_states(N_NANHAI_STATES, 83, loc="南海")]]
    for loc in LOCATIONS:
        if loc == "南海":
            continue
        q2_states.append((loc, fresh_states(1, 84, loc=loc)[0]))
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
                    "dpo2-q2-hold", triple, persona, intent, "问：我在哪？")
    # 3/4/5. 问候、随身、无意图意图护栏（同一批新状态）。
    for triple in fresh_states(N_HOLD_STATES, 85):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            base = me.anchor_lines() + "\n" + state.summary()
            for q in ("问：你好？", "问：你好啊？"):
                add(base + "\n" + q, GREET_CHOSEN[q], GREET_REJECTED[q][persona],
                    "dpo2-greet-hold", triple, persona, "", q)
            chosen_q6 = answers_for(state, me)[5]
            rejected_q6 = "随身物品无。" if state.carried_items() else "随身物品蟠桃。"
            assert rejected_q6 != chosen_q6
            add(base + "\n问：随身物品有什么？", chosen_q6, rejected_q6,
                "dpo2-q6-hold", triple, persona, "", "问：随身物品有什么？")
            add(base + "\n问：我要干什么？", "我没有意图。", "我要找金箍棒。",
                "dpo2-q5-hold", triple, persona, "", "问：我要干什么？")
    assert len(pairs) == 600, len(pairs)
    for r in pairs:
        for field in ("prompt", "chosen", "rejected"):
            bad = [c for c in r[field] if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in {field} {r[field][:40]!r}"
    random.Random(86).shuffle(pairs)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in pairs)
    report = (
        "# DPO 第二轮（悟空）偏好对子报告\n\n"
        f"对子：{len(pairs)} 对，各来源：{dict(counts)}\n"
        f"Q1 身份：20 新状态 × 5 人格 × 2 形状 = 200 对（被拒默认句）"
        f"＋ 悟空双重对比 40 对\n"
        f"Q2 护栏：南海 10 态 × 5 人格 × 2 形状＝100 ＋ 他地 6 态 × 5 × 2＝60\n"
        f"护栏：问候 100 ＋ 随身 50 ＋ 无意图意图 50\n"
        f"探针三元组已排除；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
