#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自报 SFT 冒烟数据：锚点 + 摘要 + 自态问题 -> 标准答案（纯标准库生成）。

设计（只为验证“自报能不能被教会”，不是全量配方）：
  - 状态全部由工具真实走出来（随机游走，种子固定可复现），不用手工拼状态。
  - 关键性质（已验）：队伍永远跟随游人、观音永远在南海，所以一个状态
    完全由（地点，时间步，随身集合）决定；探针 5 态也落在这个三元组里。
  - 留出（holdout）：训练状态的三元组与探针 5 态的三元组一律不重合，
    意图也不用探针的 5 句；探针测的是泛化，不是背诵。
  - 记录格式复用 world_example 款式：text = prompt + "\\n" + answer，
    summary = prompt，于是损失只落在答案段 + EOS，提示段权重 0。
  - 答案按探针评分规则逐条自验：每条训练答案用 world_anchor_probe.score
    对本态打分，不过就硬报错——训练答案必须本身就是探针满分答案。
  - 词表安全：每条记录全部字符必须在西游记语料字符集内（即编码零 UNK；
    训练器里还会用真实分词器再验一次）。

问题与标准答案（与探针 Q1-Q6 一一对应）：
  Q1 我是谁。       -> 我是欧米伽。当前人格{人格模型名}。
  Q2 我在哪。       -> 你在{地点}。
  Q3 现在何时。     -> 现在是{白天/夜晚}。
  Q4 在场人物有谁。 -> 在场人物{在场名录}。
  Q5 我要干什么。   -> 我要{意图}。
  Q6 随身物品有什么。-> 随身物品无。/ 随身物品{随身名录}。

用法：python3 build_self_report_data.py
  写出 self_report_records.jsonl（50 态 × 6 问 = 300 条）与
  SELF_REPORT_DATA_REPORT.md。
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
from world_anchor_probe import build_cases, score as probe_score

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
OUT_JSONL = HERE / "self_report_records.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_REPORT.md"

N_STATES = 50
QUESTIONS = ["我是谁。", "我在哪。", "现在何时。",
             "在场人物有谁。", "我要干什么。", "随身物品有什么。"]
PERSONAS = ["omega", "紫薇", "悟空", "一一", "Dayan"]
# 探针意图一律不用（留出）。
INTENTS = ["去水帘洞看看", "找金箍棒", "去高老庄看看", "找九齿钉耙",
           "去流沙河看看", "找月牙铲", "去天宫看看", "回花果山",
           "在取经路休息", "找锦襕袈裟", "找紫金钵盂", "在花果山休息",
           "去取经路看看", "在高老庄休息", "在流沙河休息", "在天宫休息"]


def rebuild_state(loc, time_step, carried):
    """由（地点，时间步，随身）重建与真实游走一致的世界状态。

    依据：队伍永远跟随游人（初始同在取经路，每步跟随），观音永远南海，
    物品除随身外都在初始地点。故三元组决定全部状态。
    """
    st = ws.WorldState(loc)
    st.time_step = time_step
    st.char_loc = {name: (loc if info["party"] else "南海")
                   for name, info in ws.CHARACTERS.items()}
    st.item_loc = {name: (ws.CARRIED if name in carried else init)
                   for name, init in ws.ITEMS.items()}
    st.trace = []
    return st


def answers_for(state, me):
    """6 问标准答案（顺序与 QUESTIONS 对应）。"""
    present = state.present_characters()
    carried = state.carried_items()
    return [
        f"我是欧米伽。当前人格{me.persona_model()}。",
        f"你在{state.visitor_location}。",
        f"现在是{state.day_night()}。",
        f"在场人物{ws._join(present)}。",
        f"我要{me.intent}。",
        "随身物品无。" if not carried else f"随身物品{ws._join(carried)}。",
    ]


def random_walk(rng):
    """一次随机游走（0-6 步有效工具），返回（地点，时间步，随身 frozenset）。"""
    st = ws.WorldState()
    n_ops = rng.randint(0, 6)
    for _ in range(n_ops):
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
    # 探针留出：三元组 + 意图。
    probe_triples = set()
    probe_intents = set()
    for _sid, pstate, pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
        probe_intents.add(pself.intent)
    # 重建逻辑自验：探针 5 态用三元组重建，摘要必须一字不差。
    for _sid, pstate, _pself in build_cases():
        rebuilt = rebuild_state(pstate.visitor_location, pstate.time_step,
                                frozenset(pstate.carried_items()))
        assert rebuilt.summary() == pstate.summary(), (
            pstate.summary(), rebuilt.summary())
    print("rebuild check: probe 5 states summaries identical ✓")

    rng = random.Random(11)
    states = []
    seen = set()
    guard = 0
    while len(states) < N_STATES and guard < 20000:
        guard += 1
        triple = random_walk(rng)
        if triple in probe_triples or triple in seen:
            continue
        seen.add(triple)
        states.append(triple)
    assert len(states) == N_STATES, f"only collected {len(states)} states"
    assert not (set(INTENTS) & probe_intents), "intent overlap with probe!"

    records = []
    for i, (loc, t, carried) in enumerate(states):
        state = rebuild_state(loc, t, carried)
        me = SelfState(persona=PERSONAS[i % len(PERSONAS)],
                       intent=INTENTS[i % len(INTENTS)])
        prompt_base = me.anchor_lines() + "\n" + state.summary()
        for qi, q in enumerate(QUESTIONS):
            ans = answers_for(state, me)[qi]
            prompt = prompt_base + "\n" + q
            text = prompt + "\n" + ans
            # 词表安全：全部字符在语料字符集内。
            bad = [c for c in text if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in record {i} Q{qi + 1}"
            # 答案本身必须过探针评分（训练答案 = 满分答案）。
            ok, note = probe_score(state, me, qi, ans)
            assert ok, f"answer fails probe scoring: state {i} Q{qi + 1} {note}"
            records.append({
                "text": text, "summary": prompt,
                "state": {"loc": loc, "time_step": t,
                          "carried": sorted(carried)},
                "persona": me.persona, "intent": me.intent,
                "question": q, "answer": ans,
            })
    assert len(records) == N_STATES * 6
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    locs = sorted({r["state"]["loc"] for r in records})
    times = sorted({r["state"]["time_step"] for r in records})
    nights = sum(1 for r in records if "夜晚" in r["answer"] and r["question"] == "现在何时。")
    report = (
        "# 自报数据报告（冒烟用）\n\n"
        f"记录：{len(records)} 条 = {N_STATES} 态 × 6 问\n"
        f"地点覆盖：{locs}\n"
        f"时间步覆盖：{times}（夜晚态问答 {nights} 条）\n"
        f"人格：{PERSONAS} 各 {N_STATES // len(PERSONAS)} 态\n"
        f"意图：{len(INTENTS)} 句轮换，与探针 5 句零重合\n"
        f"留出：训练三元组与探针 5 态三元组零重合（已断言）\n"
        f"自验：全部 {len(records)} 条答案过探针评分；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
