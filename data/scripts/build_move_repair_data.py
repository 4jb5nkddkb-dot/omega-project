#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""移动修复补丁数据：修 SFT 悟空产品丢移动工具调用的问题
（用户 2026-10-30 点“再补一轮修移动”）。

病因（已实测）：悟空产品对循环移动提示（锚点＋摘要＋“去花果山
看看。”）不再发《移动》调用，改发西游记叙述（“涌万花果山中洞府
……”）；DPO1/补丁二轮同提示正常发调用。查看/拾取/对话仍正常，
只有移动翻了——移动提示与叙述竞争最激烈（花果山叙述在基座里
极强），Q1 重的 SFT 把纯文本路径推过了调用路径。

修法：把工具调用写进 self 记录（训练器 self 通道权重 0.5），
提示格式与循环完全一致（锚点行＋摘要＋用户话），答案只写调
用本身（《移动》X】），不写〔结果〕与叙述——结果由路由执行后
再生成，不教模型编造结果。内置工具记录（1763 条，0.1 通道）
不含锚点行，上一轮已证明保不住移动，故必须用锚点格式重教。

构成（1250 条）：
  工具调用 600 条（答案均为调用本身）：
    移动 300 条：去X看看 200（20 新状态 × 5 人格 × 2 形状）
      ＋ 到X去 100（10 新状态 × 5 × 2），目标随机取非当前地点
      （移动完全图，无邻接限制）；
    查看 100 条：看看周围 50（5 态 × 5 × 2，答案《查看》】空参数）
      ＋ 看看X 50（5 态 × 5 × 2，答案《查看》X】）；
    拾取 100 条：10 个有可见物品的新状态 × 5 × 2，答案《拾取》物】；
    对话 100 条：10 新状态 × 5 × 2，答案《对话》在场人物】。
  问答回归 650 条（保住悟空轮成果）：
    Q1 200 条（全人格 100：10 态 × 5 × 2；悟空加码 100：50 态 × 2）；
    Q2 200 条（南海 100：10 态 × 5 × 2；他地 100：20 非南海态 × 5
      无意图）；
    问候 100 条（10 态 × 5 × 2 问法，无意图）；
    随身 50 条（5 态 × 5 × 2）；无意图意图 50 条（10 态 × 5）；
    时间/在场 50 条（5 态 × 5 × 2 问）。
  新状态一律排除探针三元组；有意图取自全量轮意图池（已留出探针
  意图，仍断言复核）；字符检查：问答部分在语料字符集内，工具
  调用另含《》】（已验 tokenizer 可编、无 UNK）。

用法：python3 build_move_repair_data.py
  写出 self_report_records_move.jsonl（1250 条）与
  SELF_REPORT_DATA_MOVE_REPORT.md。
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
OUT_JSONL = HERE / "self_report_records_move.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_MOVE_REPORT.md"

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
GREET_FORMS = (("问：你好？", "你好。"), ("问：你好啊？", "你好啊。"))
BRACKETS = set("《》】")


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    allowed = corpus_chars | BRACKETS
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

    def fresh_states(n, seed, loc=None, need_item=False):
        rng = random.Random(seed)
        states, seen, guard = [], set(), 0
        while len(states) < n and guard < 60000:
            guard += 1
            triple = random_walk(rng, max_ops=8)
            if triple in probe_triples or triple in seen:
                continue
            if loc == "NOT_NANHAI":
                if triple[0] == "南海":
                    continue
            elif loc is not None and triple[0] != loc:
                continue
            st = rebuild_state(*triple)
            if need_item:
                visible = [it for it, l in st.item_loc.items()
                           if l == st.visitor_location]
                if not visible:
                    continue
            seen.add(triple)
            states.append(triple)
        assert len(states) == n, f"only {len(states)}/{n} for loc={loc}"
        return states

    intent_rng = random.Random(141)
    target_rng = random.Random(142)
    records = []

    def add(state, me, user_text, ans, source):
        triple = (state.visitor_location, state.time_step,
                  frozenset(state.carried_items()))
        prompt = me.anchor_lines() + "\n" + state.summary() + "\n" + user_text
        records.append({
            "text": prompt + ans, "summary": prompt,
            "answer_start": len(prompt),
            "state": {"loc": triple[0], "time_step": triple[1],
                      "carried": sorted(triple[2])},
            "persona": me.persona, "intent": me.intent,
            "question": user_text, "answer": ans, "source": source,
        })

    def both_shapes(state, persona, fn, source):
        me = SelfState(persona=persona, intent="")
        fn(state, me, source + "-nointent")
        me = SelfState(persona=persona,
                       intent=intent_rng.choice(intent_pool))
        fn(state, me, source + "-withintent")

    def move_target(state):
        choices = [l for l in LOCATIONS if l != state.visitor_location]
        return target_rng.choice(choices)

    # 移动：去X看看 200 条。
    for triple in fresh_states(20, 121):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _mv(st, me, src):
                tgt = move_target(st)
                add(st, me, f"去{tgt}看看。", f"《移动》{tgt}】", src)
            both_shapes(state, persona, _mv, "move-goto")
    # 移动：到X去 100 条。
    for triple in fresh_states(10, 122):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _mv2(st, me, src):
                tgt = move_target(st)
                add(st, me, f"到{tgt}去。", f"《移动》{tgt}】", src)
            both_shapes(state, persona, _mv2, "move-to")
    # 查看：看看周围 50 条（空参数）。
    for triple in fresh_states(5, 123):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _lk0(st, me, src):
                add(st, me, "看看周围。", "《查看》】", src)
            both_shapes(state, persona, _lk0, "look-here")
    # 查看：看看X 50 条。
    for triple in fresh_states(5, 124):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _lk1(st, me, src):
                tgt = target_rng.choice(LOCATIONS)
                add(st, me, f"看看{tgt}。", f"《查看》{tgt}】", src)
            both_shapes(state, persona, _lk1, "look-loc")
    # 拾取 100 条（仅有可见物品的状态）。
    for triple in fresh_states(10, 125, need_item=True):
        state = rebuild_state(*triple)
        visible = [it for it, l in state.item_loc.items()
                   if l == state.visitor_location]
        for persona in PERSONAS:
            def _tk(st, me, src):
                item = target_rng.choice(visible)
                add(st, me, f"拾起{item}。", f"《拾取》{item}】", src)
            both_shapes(state, persona, _tk, "take-item")
    # 对话 100 条。
    for triple in fresh_states(10, 126):
        state = rebuild_state(*triple)
        present = state.present_characters()
        for persona in PERSONAS:
            def _tl(st, me, src):
                ch = target_rng.choice(present)
                add(st, me, f"和{ch}说说话。", f"《对话》{ch}】", src)
            both_shapes(state, persona, _tl, "talk-char")
    # 问答回归：Q1 200 条。
    for triple in fresh_states(10, 127):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _q1(st, me, src):
                add(st, me, "问：你是谁？", answers_for(st, me)[0], src)
            both_shapes(state, persona, _q1, "reg-q1-all")
    for triple in fresh_states(50, 128):
        state = rebuild_state(*triple)
        def _q1w(st, me, src):
            add(st, me, "问：你是谁？", answers_for(st, me)[0], src)
        both_shapes(state, "悟空", _q1w, "reg-q1-wukong")
    # 问答回归：Q2 南海 100 条。
    for triple in fresh_states(10, 129, loc="南海"):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _q2n(st, me, src):
                add(st, me, "问：我在哪？", answers_for(st, me)[1], src)
            both_shapes(state, persona, _q2n, "reg-q2-nanhai")
    # 问答回归：Q2 他地 100 条（无意图）。
    for triple in fresh_states(20, 130, loc="NOT_NANHAI"):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            add(state, me, "问：我在哪？", answers_for(state, me)[1],
                "reg-q2-other-nointent")
    # 问答回归：问候 100 条（无意图）。
    for triple in fresh_states(10, 131):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            for q, ans in GREET_FORMS:
                add(state, me, q, ans, "reg-greet-nointent")
    # 问答回归：Q6 50 条。
    for triple in fresh_states(5, 132):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            def _q6(st, me, src):
                add(st, me, "问：随身物品有什么？",
                    answers_for(st, me)[5], src)
            both_shapes(state, persona, _q6, "reg-q6")
    # 问答回归：Q5 无意图 50 条。
    for triple in fresh_states(10, 133):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            add(state, me, "问：你要干什么？", "我没有意图。",
                "reg-q5-nointent")
    # 问答回归：Q3/Q4 50 条（无意图）。
    for triple in fresh_states(5, 134):
        state = rebuild_state(*triple)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            add(state, me, "问：现在何时？", answers_for(state, me)[2],
                "reg-q3-nointent")
            add(state, me, "问：在场人物有谁？", answers_for(state, me)[3],
                "reg-q4-nointent")
    assert len(records) == 1250, len(records)
    for r in records:
        bad = [c for c in r["text"] if c not in allowed]
        assert not bad, f"non-vocab chars {bad} in {r['text'][:40]!r}"
    random.Random(139).shuffle(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["source"] for r in records)
    report = (
        "# 移动修复补丁数据报告\n\n"
        f"记录：{len(records)} 条，各来源：{dict(counts)}\n"
        f"工具调用 600 条：移动 300（去看看 200＋到去 100）＋ 查看 100"
        f"＋ 拾取 100 ＋ 对话 100（答案均为调用本身）\n"
        f"问答回归 650 条：Q1 200（全人格 100＋悟空 100）＋ Q2 200"
        f"（南海 100＋他地 100）＋ 问候 100 ＋ 随身 50 ＋ 无意图意图 50"
        f"＋ 时间/在场 50\n"
        f"探针三元组已排除；字符在语料集＋《》】内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
