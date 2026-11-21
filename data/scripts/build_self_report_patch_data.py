#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补丁数据：只补无意图形状（补丁短跑用，基座为全量产品）。

已验根因：3000 条自报训练记录里无意图的 0 条——无意图锚点
（“我是欧米伽。当前人格欧米伽。”后无意图句）从没进过训练，
是分布外形状；有意图时身份全对，无意图时 omega 在部分态上
塌向大眼/紫薇。另：无意图时问意图，模型编造训练意图
（找金箍棒等）——没有目标却说有目标，须给诚实答案。

补丁构成（标记格式与之前一字不动）：
  1. 无意图 Q1：50 个工具真实走出的新状态 × 5 人格 = 250 条，
     答案 = “我是欧米伽。当前人格X。”（人格与意图无关，重点治 omega）。
  2. 无意图 Q5：同 50 态 × 5 人格 = 250 条，
     答案 = “我没有明确的意图。”（用户已点头的新 spec：没目标就直说）。
  3. 有意图锚定样本：从全量轮记录随机抽 500 条（种子固定），
     防止补丁把有意图问答带漂。
  合计 1000 条进训练器 self 池（补丁:锚定 = 1:1）。
  探针 5 态三元组继续留出；Q1 答案按探针评分自验；全部字符
  在语料字符集内（即“我没有明确的意图。”可编码，否则硬报错）。

用法：python3 build_self_report_patch_data.py
  写出 self_report_records_patch.jsonl（1000 条）与
  SELF_REPORT_DATA_PATCH_REPORT.md。
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
FULL_JSONL = HERE / "self_report_records_full.jsonl"
OUT_JSONL = HERE / "self_report_records_patch.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_PATCH_REPORT.md"

N_STATES = 50
# 注：原拟“我没有明确的意图。”含“确”字，西游记语料无此字（会变 UNK），
# 故改用同义且全词表可编码的“我没有意图。”。
NO_INTENT_Q5_ANSWER = "我没有意图。"


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    probe_triples = set()
    for _sid, pstate, _pself in build_cases():
        probe_triples.add((pstate.visitor_location, pstate.time_step,
                           frozenset(pstate.carried_items())))
    assert NO_INTENT_Q5_ANSWER
    assert all(c in corpus_chars for c in NO_INTENT_Q5_ANSWER), \
        "Q5 no-intent answer has non-vocab chars"

    rng = random.Random(31)
    states = []
    seen = set()
    guard = 0
    while len(states) < N_STATES and guard < 20000:
        guard += 1
        triple = random_walk(rng, max_ops=8)
        if triple in probe_triples or triple in seen:
            continue
        seen.add(triple)
        states.append(triple)
    assert len(states) == N_STATES, f"only collected {len(states)} states"

    records = []
    for (loc, t, carried) in states:
        state = rebuild_state(loc, t, carried)
        for persona in PERSONAS:
            me = SelfState(persona=persona, intent="")
            base = me.anchor_lines() + "\n" + state.summary()
            # Q1 无意图：答案只含主体 + 人格，与意图无关。
            q1 = "我是谁。"
            prompt = base + "\n" + "问：" + q1[:-1] + "？"
            ans = answers_for(state, me)[0]
            ok, note = probe_score(state, me, 0, ans)
            assert ok, f"Q1 no-intent answer fails scoring: {note}"
            records.append({
                "text": prompt + ans, "summary": prompt,
                "answer_start": len(prompt),
                "state": {"loc": loc, "time_step": t,
                          "carried": sorted(carried)},
                "persona": me.persona, "intent": "",
                "question": "问：我是谁？", "answer": ans,
                "source": "patch-q1-nointent",
            })
            # Q5 无意图：诚实答案（新 spec，用户已点头）。
            prompt5 = base + "\n" + "问：我要干什么？"
            text5 = prompt5 + NO_INTENT_Q5_ANSWER
            bad = [c for c in text5 if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in Q5 no-intent record"
            records.append({
                "text": text5, "summary": prompt5,
                "answer_start": len(prompt5),
                "state": {"loc": loc, "time_step": t,
                          "carried": sorted(carried)},
                "persona": me.persona, "intent": "",
                "question": "问：我要干什么？", "answer": NO_INTENT_Q5_ANSWER,
                "source": "patch-q5-nointent",
            })
    assert len(records) == N_STATES * len(PERSONAS) * 2, len(records)
    # 有意图锚定样本：全量轮记录随机抽 500 条（种子固定）。
    full_recs = [json.loads(l) for l in FULL_JSONL.open(encoding="utf-8")]
    anchor = random.Random(5).sample(full_recs, 500)
    for r in anchor:
        r = dict(r)
        r["source"] = "anchor-with-intent"
        records.append(r)
    assert len(records) == 1000, len(records)
    # 全量字符安全：补丁记录逐条验（锚定样本已在全量轮验过）。
    for r in records:
        if r["source"].startswith("patch"):
            bad = [c for c in r["text"] if c not in corpus_chars]
            assert not bad, f"non-vocab chars {bad} in {r['text'][:40]!r}"
    rng2 = random.Random(9)
    rng2.shuffle(records)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = (
        "# 补丁数据报告（补丁短跑用）\n\n"
        f"记录：{len(records)} 条 = 无意图 Q1 {N_STATES * 5} 条"
        f" + 无意图 Q5 {N_STATES * 5} 条"
        f" + 有意图锚定样本 500 条\n"
        f"状态：{N_STATES} 个工具真实走出的新状态，探针三元组留出（已断言）\n"
        f"人格：{list(PERSONAS)} 各 {N_STATES} 态（omega 覆盖 {N_STATES} 条 Q1）\n"
        f"Q5 无意图答案：{NO_INTENT_Q5_ANSWER}（新 spec，用户已点头）\n"
        f"自验：Q1 答案过探针评分；补丁记录全部字符在语料字符集内\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")


if __name__ == "__main__":
    main()
