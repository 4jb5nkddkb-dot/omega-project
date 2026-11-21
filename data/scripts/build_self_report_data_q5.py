#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""意图泛化轮数据：与全量轮唯一变量 = 意图池。

全量轮的诊断（标记探针 26/30，训练态 38/40，交叉 10/10）：
  人格（5 个封闭值）抄对了，意图（16 句）只对见过的抄对；
  探针 5 句新意图全被换成最近的训练句（回花果山/找金箍棒/
  去取经路看看）或截断（去南海）。模型学的是“意图 = 16 句里挑”，
  不是“抄锚点里的原句”。

本轮只改意图池：16 句 → 29 句组合句——
  去{地点}看看（6：除花果山）、找{物品}（5：除蟠桃）、
  回{地点}（6：除取经路）、在{地点}休息（6：除南海）、
  见{人物}（6：全部）。
  组合空间大了，死记 16 句的策略失效，唯一解是逐字抄锚点。
状态（150 态，种子与分层与全量轮一字不动，经断言与全量轮态全同）、
人格轮换、记录结构、标记格式全部不动，于是本轮与全量轮的
唯一差别就是意图池——A/B 干净。

防污染（硬断言）：池中任何一句不得等于探针 5 句，也不得包含
探针 5 句作子串（否则模型输出训练句即含探针句，探针会误判为对）；
探针句也不得等于池中任何一句。

用法：python3 build_self_report_data_q5.py
  写出 self_report_records_q5.jsonl（1200 条）与
  SELF_REPORT_DATA_Q5_REPORT.md。
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_state as ws
from world_anchor import SelfState
from build_self_report_data import rebuild_state, answers_for
from world_anchor_probe import build_cases, score as probe_score

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
FULL_JSONL = HERE / "self_report_records_full.jsonl"
OUT_JSONL = HERE / "self_report_records_q5.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_Q5_REPORT.md"

N_STATES = 150
QUESTIONS = ["我是谁。", "我在哪。", "现在何时。",
             "在场人物有谁。", "我要干什么。", "随身物品有什么。"]
PERSONAS = ["omega", "紫薇", "悟空", "一一", "Dayan"]
# 组合意图池（29 句）：模板 × 取值，探针 5 句已从模板取值中剔除。
POOL = ([f"去{loc}看看" for loc in ws.LOCATIONS if loc != "花果山"]
        + [f"找{item}" for item in ws.ITEMS if item != "蟠桃"]
        + [f"回{loc}" for loc in ws.LOCATIONS if loc != "取经路"]
        + [f"在{loc}休息" for loc in ws.LOCATIONS if loc != "南海"]
        + [f"见{name}" for name in ws.CHARACTERS])


def random_walk(rng, max_ops=8):
    """一次随机游走（0-max_ops 步有效工具），返回（地点，时间步，随身）。"""
    st = ws.WorldState()
    for _ in range(rng.randint(0, max_ops)):
        ops = []
        for loc in ws.LOCATIONS:
            if loc != st.visitor_location:
                ops.append(("move", loc))
        for item in st.present_items():
            ops.append(("take", item))
        for name in st.present_characters():
            ops.append(("talk", name))
        ops.append(("look", ""))
        kind, arg = ops[rng.randrange(len(ops))]
        if kind == "move":
            st.move(arg)
        elif kind == "take":
            st.take(arg)
        elif kind == "talk":
            st.talk(arg)
        else:
            st.look(arg)
    return (st.visitor_location, st.time_step, frozenset(st.carried_items()))


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    probe_triples = set()
    probe_intents = set()
    for _sid, pstate, pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
        probe_intents.add(pself.intent)
    assert len(POOL) == 29, len(POOL)
    assert len(set(POOL)) == 29, "pool has duplicates"
    for it in POOL:
        for pi in probe_intents:
            assert it != pi, f"pool intent equals probe intent: {it}"
            assert pi not in it, f"pool intent contains probe intent: {it} ⊃ {pi}"
    print(f"intent pool: {len(POOL)} sentences, probe-exclusion asserts pass ✓")

    # 与全量轮同种子、同分层 → 同 150 态（断言验证）。
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
    # 与全量轮态全同（A/B 保证）。
    full_recs = [json.loads(l) for l in FULL_JSONL.open(encoding="utf-8")]
    full_states = [(r["state"]["loc"], r["state"]["time_step"],
                    frozenset(r["state"]["carried"]))
                   for r in full_recs[0::8]]
    assert full_states == states, "states differ from full round!"
    print("states identical to full round ✓")

    records = []
    for i, (loc, t, carried) in enumerate(states):
        state = rebuild_state(loc, t, carried)
        p_a = PERSONAS[i % len(PERSONAS)]
        p_b = PERSONAS[(i + 2) % len(PERSONAS)]
        it_a = POOL[i % len(POOL)]
        it_b = POOL[(i + 7) % len(POOL)]
        assert p_a != p_b and it_a != it_b
        for me, qidxs in ((SelfState(persona=p_a, intent=it_a), range(6)),
                           (SelfState(persona=p_b, intent=it_b), (0, 4))):
            base = me.anchor_lines() + "\n" + state.summary()
            for qi in qidxs:
                q_old = QUESTIONS[qi]
                q_core = q_old[:-1]
                prompt = base + "\n" + "问：" + q_core + "？"
                ans = answers_for(state, me)[qi]
                text = prompt + ans
                bad = [c for c in text if c not in corpus_chars]
                assert not bad, f"non-vocab chars {bad} in {text[:40]!r}"
                ok, note = probe_score(state, me, qi, ans)
                assert ok, f"answer fails probe scoring: state {i} Q{qi + 1} {note}"
                records.append({
                    "text": text, "summary": prompt,
                    "answer_start": len(prompt),
                    "state": {"loc": loc, "time_step": t,
                              "carried": sorted(carried)},
                    "persona": me.persona, "intent": me.intent,
                    "question": "问：" + q_core + "？", "answer": ans,
                })
    assert len(records) == N_STATES * 8, len(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    intent_counts = Counter(r["intent"] for r in records
                            if r["question"] == "问：我要干什么？")
    report = (
        "# 意图泛化轮数据报告\n\n"
        f"记录：{len(records)} 条 = {N_STATES} 态 × 8（与全量轮结构全同）\n"
        f"状态：与全量轮 150 态全同（已断言）；人格轮换与全量轮全同\n"
        f"意图池：{len(POOL)} 句组合句（去×6、找×5、回×6、在×6、见×6），"
        f"每句作 Q5 答案约 {min(intent_counts.values())}-{max(intent_counts.values())} 次\n"
        f"防污染：池中无探针 5 句、无含探针句作子串者（已断言）\n"
        f"自验：全部 {len(records)} 条答案过探针评分；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
