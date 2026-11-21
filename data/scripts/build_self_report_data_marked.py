#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标记版自报数据：与冒烟 1 唯一变量 = 问答格式。

冒烟 1 的结论是未标记裸问题根本不被认作问题（训练态也只有 2/30）。
本轮只改一处：问题加协议已证明有效的问句标记——
  旧：锚点 + 摘要 + "我是谁。"            -> 答案另起一行
  新：锚点 + 摘要 + "问：我是谁？"         -> 答案紧跟其后（协议款式：
      协议记录就是 "问：…？" + 答案直接相连，无分隔符）
状态（50 态）、人格、意图、答案一字不动，全部沿用
self_report_records.jsonl；训练器种子与配比也不动，于是冒烟 2 与冒烟 1
的唯一差别就是这个标记——A/B 干净。

记录字段：text = prompt + answer；summary = prompt；
answer_start = len(prompt)（训练器按此切权重：提示段 0、答案段 + EOS 1）。
自验：全部字符在语料字符集内（即“问：？”皆可编码，否则硬报错）；
全部答案仍过未标记探针的评分规则（评分只看答案内容，与标记无关）。

用法：python3 build_self_report_data_marked.py
  写出 self_report_records_marked.jsonl（300 条）与
  SELF_REPORT_DATA_MARKED_REPORT.md。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/workspace/omega-chat"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_state as ws
from world_anchor import SelfState
from build_self_report_data import rebuild_state
from world_anchor_probe import score as probe_score

HERE = Path(__file__).resolve().parent
MIRROR = Path(os.path.expanduser("~/workspace/omega-chat"))
CORPUS = MIRROR / "Xiyouji.txt"
SRC_JSONL = HERE / "self_report_records.jsonl"
OUT_JSONL = HERE / "self_report_records_marked.jsonl"
OUT_REPORT = HERE / "SELF_REPORT_DATA_MARKED_REPORT.md"


def main():
    corpus_chars = set(CORPUS.read_text(encoding="utf-8"))
    src = [json.loads(l) for l in SRC_JSONL.open(encoding="utf-8")]
    assert len(src) == 300, len(src)
    records = []
    for r in src:
        st = rebuild_state(r["state"]["loc"], r["state"]["time_step"],
                           frozenset(r["state"]["carried"]))
        me = SelfState(persona=r["persona"], intent=r["intent"])
        # 锚点 + 摘要必须与冒烟 1 一致（只换问句包装）。
        old_prompt = r["summary"]
        q_old = r["question"]
        assert old_prompt.endswith("\n" + q_old), old_prompt[-20:]
        base = old_prompt[: -len(q_old)]
        q_core = q_old[:-1]  # 去掉句末“。”，如“我是谁”
        assert q_core, q_old
        prompt = base + "问：" + q_core + "？"
        assert prompt.startswith(me.anchor_lines() + "\n" + st.summary() + "\n")
        ans = r["answer"]
        text = prompt + ans
        bad = [c for c in text if c not in corpus_chars]
        assert not bad, f"non-vocab chars {bad} in {text[:40]!r}"
        qi = ["我是谁。", "我在哪。", "现在何时。",
              "在场人物有谁。", "我要干什么。", "随身物品有什么。"].index(q_old)
        ok, note = probe_score(st, me, qi, ans)
        assert ok, f"answer fails probe scoring: {note}"
        records.append({
            "text": text, "summary": prompt, "answer_start": len(prompt),
            "state": r["state"], "persona": r["persona"], "intent": r["intent"],
            "question": "问：" + q_core + "？", "answer": ans,
        })
    assert len(records) == 300
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = (
        "# 标记版自报数据报告（冒烟 2 用）\n\n"
        "记录：300 条 = 50 态 × 6 问，与冒烟 1 态、人格、意图、答案全同\n"
        "唯一改动：问句包装——旧“问题。”另起行答；新“问：问题？”答案紧跟\n"
        "自验：全部 300 条答案过探针评分；全部字符在语料字符集内\n"
        "（“问：？”可编码已由本断言验证，非假设）\n"
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSONL} ({len(records)} records)")
    print("sample prompt:", records[0]["summary"][-40:])
    print("sample text tail:", records[0]["text"][-30:])


if __name__ == "__main__":
    main()
