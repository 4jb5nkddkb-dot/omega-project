#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT 悟空补丁数据：不用 DPO、改回 SFT 专治 DPO1 的悟空塌缩
（用户 2026-10-30 点“SFT 补丁治悟空”）。

基座 DPO1（step 73773）：南海已修好、问候 20/20，只剩悟空
身份在无意图下全塌到欧米伽（无意图 Q1 20/25，标记 Q1 4/5 仅
S2/悟空错）。DPO 两轮（v1＋v2 再平衡）都把南海翻到铭，
证明 DPO 更新方向本身伤脆弱的南海竞争；改回 SFT，失败模式
不同（缺护栏的轴回退，可用回放补），不再有被拒句抬生僻字
的动力。

补丁三教训（必须吸取）：补丁三回归集里没有 Q2，南海就破了。
本轮回归集必须把 Q2（尤其南海）摆足，不再留无护栏的轴。

构成（1200 条，与补丁二轮/三轮同规模）：
  A. 目标 Q1 400 条：
     A1 全人格 200 条：20 新状态 × 5 人格 × 2 形状（无/有意图），
        答案只认人格（悟空 40 条在内）；
     A2 悟空加码 200 条：100 新状态 × 悟空 × 2 形状。
  B. Q2 回归 300 条：
     B1 南海 150 条：15 新南海状态 × 5 人格 × 2 形状；
     B2 他地 150 条：6 地点 × 5 新状态 × 5 人格 × 无意图。
  C. 问候 150 条：15 新状态 × 5 人格 × 2 问法，无意图。
  D. 随身 Q6 150 条：15 新状态 × 5 人格 × 2 形状。
  E. 无意图意图 Q5 100 条：20 新状态 × 5 人格，无意图，
     答案我没有意图。（有意图 Q5 已是平台期，不列为目标。）
  F. 时间 Q3＋在场 Q4 100 条：10 新状态 × 5 人格 × 2 问，
     无意图。
  新状态一律排除探针三元组；有意图取自全量轮意图池（已留出
  探针意图，仍断言复核）；全部记录字符在语料字符集内。

用法：python3 build_sft_wukong_data.py
  写出 self_report_records_wukong.jsonl（1200 条）与
  SELF_REPORT_DATA_WUKONG_REPORT.md。
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_anchor import SelfState, PERSONAS
from build_self_report_data import rebuild_state, answers_for
from build_self_report_data_full import random_walk
from world_anchor_probe import build_cases

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
FULL_JSONL = HERE / "self_report_records_full.jsonl"
OUT_JSONL = HERE / "self_report_records_wukong.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_WUKONG_REPORT.md"

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
GREET_FORMS = (("问：你好？", "你好。"), ("问：你好啊？", "你好啊。"))


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

    intent_rng = random.Random(111)
    records = []

    def add(state, me, q, ans, source):
        triple = (state.visitor_location, state.time_step,
                  frozenset(state.carried_items()))
        prompt = me.anchor_lines() + "\n" + state.summary() + "\n" + q
        records.append({
            "text": prompt + ans, "summary": prompt,
            "answer_start": len(prompt),
            "state": {"loc": triple[0], "time_step": triple[1],
                      "carried": sorted(triple[2])},
            "persona": me.persona, "intent": me.intent,
            "question": q, "answer": ans, "source": source,
        })

    def both_shapes(state, persona, fn, source):
        me = SelfState(persona=persona, intent="")
        fn(state, me, source + "-nointent")
        me = SelfState(persona=persona,
                       intent=intent_rng.choice(intent_pool))
        fn(state, me, source + "-withintent")

    # A1. Q1 全人格 200 条。
    for triple in fresh_states(20, 101):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _q1(st, me, src):
                ans = answers_for(st, me)[0]
                assert ans == f"我是欧米伽。当前人格{me.persona_model()}。", ans
                add(st, me, "问：你是谁？", ans, src)
            both_shapes(state, persona, _q1, "sft-q1-all")
    # A2. Q1 悟空加码 200 条。
    for triple in fresh_states(100, 102):
        state = rebuild_state(*triple)
        def _q1w(st, me, src):
            ans = answers_for(st, me)[0]
            assert ans == "我是欧米伽。当前人格悟空。", ans
            add(st, me, "问：你是谁？", ans, src)
        both_shapes(state, "悟空", _q1w, "sft-q1-wukong")
    # B1. Q2 南海 150 条。
    for triple in fresh_states(15, 103, loc="南海"):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _q2n(st, me, src):
                ans = answers_for(st, me)[1]
                assert ans == "你在南海。", ans
                add(st, me, "问：我在哪？", ans, src)
            both_shapes(state, persona, _q2n, "sft-q2-nanhai")
    # B2. Q2 他地 150 条（无意图）。
    for loc in LOCATIONS:
        if loc == "南海":
            continue
        for triple in fresh_states(5, 104, loc=loc):
            state = rebuild_state(*triple)
            for persona in PERSONAS:
                me = SelfState(persona=persona, intent="")
                ans = answers_for(state, me)[1]
                assert ans == f"你在{loc}。", (ans, loc)
                add(state, me, "问：我在哪？", ans, "sft-q2-other-nointent")
    # C. 问候 150 条（无意图）。
    for triple in fresh_states(15, 105):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            for q, ans in GREET_FORMS:
                add(state, me, q, ans, "sft-greet-nointent")
    # D. Q6 随身 150 条。
    for triple in fresh_states(15, 106):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _q6(st, me, src):
                add(st, me, "问：随身物品有什么？",
                    answers_for(st, me)[5], src)
            both_shapes(state, persona, _q6, "sft-q6")
    # E. Q5 无意图 100 条。
    for triple in fresh_states(20, 107):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            add(state, me, "问：你要干什么？", "我没有意图。",
                "sft-q5-nointent")
    # F. Q3 时间＋Q4 在场 100 条（无意图）。
    for triple in fresh_states(10, 108):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            add(state, me, "问：现在何时？", answers_for(state, me)[2],
                "sft-q3-nointent")
            add(state, me, "问：在场人物有谁？", answers_for(state, me)[3],
                "sft-q4-nointent")
    assert len(records) == 1200, len(records)
    for r in records:
        bad = [c for c in r["text"] if c not in corpus_chars]
        assert not bad, f"non-vocab chars {bad} in {r['text'][:40]!r}"
    random.Random(109).shuffle(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in records)
    report = (
        "# SFT 悟空补丁数据报告\n\n"
        f"记录：{len(records)} 条，各来源：{dict(counts)}\n"
        f"目标 Q1 400 条：全人格 200（20 态 × 5 × 2 形状）＋ 悟空加码 200"
        f"（100 态 × 2 形状）\n"
        f"回归 Q2 300 条：南海 150（15 态 × 5 × 2）＋ 他地 150"
        f"（6 地 × 5 态 × 5 人格，无意图）\n"
        f"回归：问候 150 ＋ 随身 150 ＋ 无意图意图 100 ＋ 时间/在场 100\n"
        f"探针三元组已排除；全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
