#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量自报数据：针对冒烟 2 的诊断下药。

冒烟 2 的诊断（标记探针 12/30，训练态 18/30）：
  模型学会了问答模板，但只会抄地点；其余全塌向多数默认值——
  Q1 恒答“当前人格紫薇”，Q3 恒答“夜晚”，Q6 默认“无”，
  Q4 在南海也只列 5 人队伍（漏观音），Q5 输出背过的训练意图
  而不是锚点里的意图。锚点才有的答案（人格、意图）一个没抄上：
  同一状态只配一个人格/意图时，模型可以靠“状态→答案”死记，
  根本不用读锚点。

本轮三处对症设计（格式与冒烟 2 的标记格式一字不动）：
  1. 同态双人格/双意图：每态配 2 个人格 + 2 个意图；6 问全做于人格 A，
     Q1/Q5 再做于人格 B。同一状态对应两个不同的人格答案，死记
     “状态→人格”必错，唯一解是读锚点。记录 150 态 × 8 = 1200 条。
  2. 地点分层：7 地点各约 21 态，南海（观音）不再是少数。
  3. 均衡：昼夜约 5:5（四桶轮取，实测白天 77/夜晚 73）；
     随身空/非空均有覆盖（长游走天然偏非空，实测空 55/非空 95：
     非空例覆盖 6 种物品已足够学抄写，空例 55 条亦非少数，
     故不再强求 5:5——抄写能力取决于覆盖而非精确比例）。
  探针 5 态三元组与探针 5 句意图继续留出；人格、意图轮换保证均衡。

用法：python3 build_self_report_data_full.py
  写出 self_report_records_v9.jsonl（1200 条）与
  SELF_REPORT_DATA_V9_REPORT.md。
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_state as ws
from world_anchor import SelfState, PERSONA_MODEL
from build_self_report_data import rebuild_state, answers_for
from world_anchor_probe import build_cases, score as probe_score

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
OUT_JSONL = HERE / "self_report_records_v9.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_V9_REPORT.md"

N_STATES = 150
QUESTIONS = ["我是谁。", "我在哪。", "现在何时。",
             "在场人物有谁。", "我要干什么。", "随身物品有什么。"]
PERSONAS = ["omega", "紫薇", "悟空", "一一", "Dayan"]
INTENTS = ["去水帘洞看看", "找金箍棒", "去高老庄看看", "找九齿钉耙",
           "去流沙河看看", "找月牙铲", "去天宫看看", "回花果山",
           "在取经路休息", "找锦襕袈裟", "找紫金钵盂", "在花果山休息",
           "去取经路看看", "在高老庄休息", "在流沙河休息", "在天宫休息"]


def random_walk(rng, max_ops=8):
    """一次随机游走（0-8 步有效工具），返回（地点，时间步，随身 frozenset）。"""
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
    assert not (set(INTENTS) & probe_intents), "intent overlap with probe!"

    # 抽大池再分层选：每地点约 21 态，地点内四桶轮取。
    # 池子长短游走对半掺：只抽长游走会缺空随身态（拿了东西就不再空，
    # 且去重后长游走的三元组更多），短游走补空随身态的供给。
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
    # 每地点内按（空/非空随身）×（白天/夜晚）四桶轮取，两维同时均衡。
    # （不分层会偏夜、偏非空：长游走的三元组更多，去重后池子天然偏向长游走。）
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

    records = []
    for i, (loc, t, carried) in enumerate(states):
        state = rebuild_state(loc, t, carried)
        p_a = PERSONAS[i % len(PERSONAS)]
        p_b = PERSONAS[(i + 2) % len(PERSONAS)]
        it_a = INTENTS[i % len(INTENTS)]
        it_b = INTENTS[(i + 5) % len(INTENTS)]
        assert p_a != p_b and it_a != it_b
        # 人格 A：6 问全做；人格 B：只做锚点题 Q1/Q5（逼读锚点）。
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

    locs = {r["state"]["loc"] for r in records}
    nights = sum(1 for r in records
                 if r["question"] == "问：现在何时？" and "夜晚" in r["answer"])
    days = sum(1 for r in records
               if r["question"] == "问：现在何时？" and "白天" in r["answer"])
    q6_empty = sum(1 for r in records
                   if r["question"] == "问：随身物品有什么？" and r["answer"] == "随身物品无。")
    q6_full = sum(1 for r in records
                  if r["question"] == "问：随身物品有什么？" and r["answer"] != "随身物品无。")
    nanhai = sum(1 for r in records if r["state"]["loc"] == "南海")
    report = (
        "# 全量自报数据报告\n\n"
        f"记录：{len(records)} 条 = {N_STATES} 态 × 8（人格 A 6 问 + 人格 B 的 Q1/Q5）\n"
        f"地点：{sorted(locs)}，每地点约 {N_STATES // 7} 态（含南海 {nanhai} 条）\n"
        f"昼夜：白天 {days} 条 / 夜晚 {nights} 条（Q3 计）\n"
        f"随身：空 {q6_empty} 条 / 非空 {q6_full} 条（Q6 计）\n"
        f"人格：{PERSONAS} 轮换，每态 2 人格不同；意图 16 句轮换，与探针零重合\n"
        f"留出：训练三元组与探针 5 态三元组零重合（已断言）\n"
        f"自验：全部 {len(records)} 条答案过探针评分；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
