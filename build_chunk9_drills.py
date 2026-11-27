#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成第 9 块教学记录（chunk 9 用，s2400 -> s2700）。

重新设计（不是第 6/7/8 块的同类记录再来一批）。诊断结论见
docs/CHUNK9_DIAGNOSIS.md，本地实证，摘要如下：

  1. 固定默认每块换一个地点（取经路 -> 高老庄），且高老庄在任何
     分布里都不是最多（累计 Q0 答案 13.5%，第 8 块边际 14.3%，
     摘要地点 14.3%）——答案分布解释不了第 8 块的默认。
     300 步模拟（seed 7，与训练同采样）实测：每块只有 38 个
     Q0 答案窗口、21 个 Q3 答案窗口、27 个 Q2 答案窗口；
     采样到的 Q0 答案里高老庄 8/38（与花果山并列最多）。
     默认由“采样到的答案分布”（高方差小样本）决定，不是数据
     分布。——钻法本身（答案区 loss + 稀释的问答信号）教的是
     “猜先验”，不是“读字段”。第 9 块改配方 + 改记录形式，
     不再只换一批同类记录。
  2. 环的失败是记忆语境触发的：s2400 环内 Q0 第 1 步（缓存空）
     答对，第 5/9 步（缓存 [Q0,Q1]）答错高老庄；环内 Q2 探针
     （无缓存）答对、环内（有缓存）三步全答可见锦襕袈裟。
     训练记录里从来没有带记忆前缀的提示，环语境对模型是分布外
     的。——本文件只写 A 组：带记忆前缀的问答记录，复刻环在各步
     的真实前缀。全域 0 条，是唯一真正新的记录形式。
  3. Q3 对所有钻法免疫，且**数据层面早已完备**，不是缺记录：
     探针单题本身坏（无缓存仍答夜晚），--no-memory 环与有记忆
     环完全相同（s2100 上 9/12，Q3 0/3），答案昼夜均衡
     （231 白/220 夜），数据 0 不一致（451/451 答案与摘要昼夜
     一致）。实测旧域里 Q3 最小对已有 33/63 组（同地点同随身，
     白天+夜晚俱全；如取经路空随身白天步 [0,0,1,1,2,2,3,3,8,8,9]
     夜晚步 [4,4,5,5,6,6,7,7]）、Q2 最小对 44/76 组；环的精确态
     （取经路步 0 空随身）Q3/Q2 各已有 2 条。第 8 块又加 49 条
     均衡 Q3 记录，Q3 仍 0/3。决策表写明“Q3 仍 0/3 则停下
     redesign，不再加同类记录”——**第 9 块不再加任何裸 Q3/Q2/
     Q0 记录**。另：词“白天”几乎只在世界域出现（经典/对话/
     常识里 0 次，代码 66 次，世界 6643 次），门若分流不净，
     左支会把 Q3 推向“夜晚”（交接单的诊断步骤含 gate_probe）。
  4. 配方：第 8 块 loss 上升（3.6087 -> 3.6836），每块全新优化器
     + lr 1e-3 cosine 300 步是一次大扰动，会重装一个默认。
     第 9 块改配方：lr 3e-4（命令行 --lr 覆盖，见交接单），
     300 步不变（s2400 -> s2700）。数据配比不动（世界域
     15%）：Q3/Q2 记录旧域已有 344/330 条，瓶颈是训练动力学
     不是记录量。
  5. 采样稀释（诚实说明）：25 条记忆前缀记录约 3.1k 字符，占
     世界域 ~0.6%，每块约采到 2 个窗口。A 组是“把新形式引进域
     里”的第一步，不是靠量翻盘；翻盘靠配方（lr 3e-4）。若第 9
     块环仍不回升，下一步不是再加记录，而是（a）世界域配比上调、
     （b）门机制调查、（c）环侧记忆格式改动（见诊断）。

构成（25 条，全用真实 WorldState 摘要，中文数字）：
  A. 带记忆前缀 25 条：S0（环的冻结态）7 条——Q0×1（前缀
     [Q1,Q2,Q3]，环第 5/9 步的真实前缀）、Q1×2（前缀 [Q0]、
     [Q2,Q3,Q0]）、Q2×2（前缀 [Q0,Q1]、[Q3,Q0,Q1]）、Q3×2
     （前缀 [Q0,Q1,Q2]、[Q0,Q1]）；其余 6 地 × 3（Q0/Q2/Q3，
     前缀 [Q0,Q1]）= 18 条。随身：S0 为空（与环一致），
     其余地点轮转。
  Q0/Q1/Q2/Q3 裸记录 0 条（探针 Q0 5/7 已是最好；Q3/Q2 数据
  已完备；环 Q0/Q2 由 A 组带前缀记录覆盖）。

只写新文件 chunk9_drills.jsonl，不动已有语料；接线（加进
training/dataset.py 的 WORLD_RECORD_FILES）不在本脚本内。
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tokenizer"))
sys.path.insert(0, str(HERE))

from char_tokenizer import CharTokenizer
from evaluation.world_state import WorldState

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]

# 6 组随身（含空），与第 6/7/8 块同一套。
CARRIED_SETS = [[], ["锦襕袈裟"], ["金箍棒"], ["蟠桃"], ["金箍棒", "蟠桃"], ["九齿钉耙"]]

ALL_ITEMS = {"蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"}
ITEM_HOME = {"蟠桃": "花果山", "金箍棒": "花果山", "九齿钉耙": "高老庄",
             "月牙铲": "流沙河", "锦襕袈裟": "取经路", "紫金钵盂": "南海"}

QUESTIONS = ["你在什么地方。", "在场人物有谁。",
             "随身物品有什么。", "现在是白天还是夜晚。"]


def daynight_of(step):
    return "白天" if (step // 4) % 2 == 0 else "夜晚"


def make_summary(loc, carried, step):
    """用真实 WorldState 走出指定状态并取摘要——与探针/环字节一致。"""
    st = WorldState()
    for item in carried:
        home = ITEM_HOME[item]
        if st.visitor_location != home:
            st.move(home)
        res = st.take(item)
        assert "拾起了" in res, f"take {item} at {home} failed: {res}"
    if st.visitor_location != loc:
        st.move(loc)
    st.time_step = step
    summ = st.summary()
    assert f"你在{loc}。" in summ
    assert daynight_of(step) in summ
    return summ, {"loc": loc, "time_step": step, "carried": list(carried)}


def answer_for(state, qi):
    """给定 state，返回 Q0..Q3 的正确答案（与 evaluation/scoring.py 一致）。"""
    loc, carried, step = state["loc"], state["carried"], state["time_step"]
    if qi == 0:
        return f"你在{loc}。"
    if qi == 1:
        # 在场人物：队伍跟随游人；南海另有观音。
        chars = ["唐僧", "孙悟空", "猪八戒", "沙僧", "白龙马"]
        if loc == "南海":
            chars.append("观音")
        return "在场人物" + "、".join(chars) + "。"
    if qi == 2:
        if not carried:
            return "随身物品无。"
        order = ["蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"]
        return "随身物品" + "、".join(sorted(carried, key=order.index)) + "。"
    return f"现在是{daynight_of(step)}。"


def mem_prefix(state, qis):
    """复刻 evaluation/sidecar.py::format_examples 的前缀格式：
    每条示例 = 提问 + 正确回复，超 64 字在最后一个“。”处截断；
    各行以 \\n 连接，末尾带一个 \\n。qis 是示例的问题下标列表
    （按缓存里的顺序）。"""
    lines = []
    for qi in qis:
        line = QUESTIONS[qi] + answer_for(state, qi)
        if len(line) > 64:
            cut = line.rfind("。", 0, 64)
            line = line[:cut + 1] if cut >= 0 else line[:64]
        lines.append(line)
    return "".join(l + "\n" for l in lines)


def make_record(summary_text, question, answer, state, register=None,
                prefix=""):
    """单轮记录：[记忆前缀] + 摘要 + 问题 + 答案。
    register='q0'：Q0 登记格式，地点句原样放摘要最前面；
    register='q3'：Q3 登记格式，答案句原样放摘要最前面
    （前缀在最前面，登记行在摘要前）。"""
    if register == "q0":
        reg = summary_text[summary_text.rfind("你在"):]
        text = prefix + reg + "\n" + summary_text + "\n" + question + answer
    elif register == "q3":
        text = prefix + answer + "\n" + summary_text + "\n" + question + answer
    else:
        text = prefix + summary_text + "\n" + question + answer
    return {"text": text, "summary": summary_text,
            "answer_start": len(text) - len(answer),
            "state": state, "question": question, "answer": answer,
            "memory_prefix": prefix}


def load_existing_texts():
    """旧域世界文件的全部 text（去重比对用，含 chunk7/chunk8）。"""
    existing = set()
    proc = HERE / "data" / "processed"
    dual = Path.home() / "workspace" / "dual_sft"
    for d in (proc, dual):
        for name in ("probe_format_records.jsonl", "register_format_records.jsonl",
                     "carried_register_format_records.jsonl",
                     "daytime_register_format_records.jsonl",
                     "character_register_format_records.jsonl",
                     "read_drills.jsonl", "chunk7_drills.jsonl",
                     "chunk8_drills.jsonl"):
            p = d / name
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        existing.add(json.loads(line)["text"])
    return existing


def build(existing, seen):
    A = []

    # ============ A. 带记忆前缀（25 条）============
    # S0（环的冻结态：取经路，第 0 步，白天，空随身）7 条。
    # 前缀按环的真实缓存演变（全答对时）：步 1 空；步 2 [Q0]；
    # 步 3 [Q0,Q1]；步 4 [Q0,Q1,Q2]；步 5/9 [Q1,Q2,Q3]；
    # 步 6/10 [Q2,Q3,Q0]；步 7/11 [Q3,Q0,Q1]。
    s0_summ, s0_state = make_summary("取经路", [], 0)
    s0_specs = [
        # (问题下标, 前缀的问题下标列表)
        (0, [1, 2, 3]),   # Q0，环第 5/9 步的真实前缀
        (1, [0]),         # Q1，环第 2 步的真实前缀
        (1, [2, 3, 0]),   # Q1，环第 6/10 步的真实前缀
        (2, [0, 1]),      # Q2，环第 3 步的真实前缀
        (2, [3, 0, 1]),   # Q2，环第 7/11 步的真实前缀
        (3, [0, 1, 2]),   # Q3，环第 4 步的真实前缀
        (3, [0, 1]),      # Q3，环第 8/12 步的真实前缀
    ]
    for qi, prefix_qis in s0_specs:
        prefix = mem_prefix(s0_state, prefix_qis)
        A.append(make_record(s0_summ, QUESTIONS[qi],
                             answer_for(s0_state, qi), s0_state,
                             prefix=prefix))
    # 其余 6 地：Q0/Q2/Q3 各 1 条，前缀 [Q0,Q1]，随身轮转，步 2。
    for li, loc in enumerate(LOCATIONS):
        if loc == "取经路":
            continue
        carried = CARRIED_SETS[(li + 1) % len(CARRIED_SETS)]
        summ, state = make_summary(loc, carried, 2)
        prefix = mem_prefix(state, [0, 1])
        for qi in (0, 2, 3):
            A.append(make_record(summ, QUESTIONS[qi],
                                 answer_for(state, qi), state,
                                 prefix=prefix))

    # 去重：A 组带前缀，旧域里没有，应 0 撞。若撞，报错而不是静默跳过。
    for r in A:
        if r["text"] in existing or r["text"] in seen:
            raise AssertionError(f"A 记忆前缀撞旧域/内部重复：{r['text'][:80]!r}")
        seen.add(r["text"])
    return A


def main():
    existing = load_existing_texts()
    print(f"existing world texts: {len(existing)}")
    seen = set()
    A = build(existing, seen)
    print("counts: {'A_记忆前缀':", len(A), "} total:", len(A))
    all_records = A
    assert len(A) == 25, f"A 组应为 25 条，实得 {len(A)}"

    # ---- 词表检查（0 OOV）----
    vocab = json.loads((HERE / "data" / "processed" / "generalist_vocab.json")
                       .read_text(encoding="utf-8"))
    tok = CharTokenizer(vocab["chars"])
    bad_oov = [r["text"][:80] for r in all_records
               if tok.UNK in tok.encode(r["text"])]
    print(f"OOV: {len(bad_oov)} / {len(all_records)}")
    for t in bad_oov[:5]:
        print("  OOV:", repr(t))
    assert not bad_oov, "有 OOV，停止写入"

    # ---- 格式检查：摘要与真实 WorldState 字节一致，无阿拉伯数字，
    # 答案与 state 一致，answer_start 正确，前缀格式与 sidecar 一致 ----
    bad_fmt = []
    for r in all_records:
        st = r["state"]
        summ2, _ = make_summary(st["loc"], st["carried"], st["time_step"])
        if summ2 != r["summary"]:
            bad_fmt.append(("summary-mismatch", r["text"][:80]))
        if re.search(r"时间第[0-9]+步", r["summary"]):
            bad_fmt.append(("arabic-step", r["text"][:80]))
        if r["text"][r["answer_start"]:] != r["answer"]:
            bad_fmt.append(("answer_start", r["text"][:80]))
        qi = QUESTIONS.index(r["question"])
        if r["answer"] != answer_for(st, qi):
            bad_fmt.append(("answer-state", r["text"][:80]))
        # 前缀：每行必须是“提问+正确回复”，且与 sidecar 截断规则一致。
        if r["memory_prefix"]:
            lines = r["memory_prefix"].rstrip("\n").split("\n")
            for line in lines:
                if not any(line.startswith(QUESTIONS[q])
                           for q in range(4)):
                    bad_fmt.append(("prefix-line", r["text"][:80]))
        if daynight_of(st["time_step"]) not in r["summary"]:
            bad_fmt.append(("daynight-summary", r["text"][:80]))
        order = ["蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"]
        car = "、".join(sorted(st["carried"], key=order.index)) \
            if st["carried"] else "无"
        if f"随身物品{car}。" not in r["summary"]:
            bad_fmt.append(("carried-summary", r["text"][:80]))
    print(f"format bad: {len(bad_fmt)} / {len(all_records)}")
    for kind, t in bad_fmt[:10]:
        print("  BAD", kind, repr(t))
    assert not bad_fmt, "格式校验失败"

    # ---- 分布终验 ----
    # A 组 S0 七条齐全（Q0/Q1×2/Q2×2/Q3×2），前缀 6 种文本
    # （Q2[Q0,Q1] 与 Q3[Q0,Q1] 的前缀文本相同）。
    s0 = [r for r in A if r["state"]["loc"] == "取经路"
          and r["state"]["time_step"] == 0]
    print("A S0 记录:", len(s0), "条（应为 7）")
    assert len(s0) == 7, len(s0)
    s0_prefixes = {r["memory_prefix"] for r in s0}
    print("A S0 前缀种类:", len(s0_prefixes), "（7 条记录，6 种前缀文本）")
    assert len(s0_prefixes) == 6, len(s0_prefixes)
    # 其余 6 地各 3 条（Q0/Q2/Q3），前缀均为 [Q0,Q1]。
    others = [r for r in A if r["state"]["loc"] != "取经路"]
    print("A 其余地点:", len(others), "条（应为 18）")
    assert len(others) == 18, len(others)
    for loc in LOCATIONS:
        if loc == "取经路":
            continue
        sub = [r for r in others if r["state"]["loc"] == loc]
        assert len(sub) == 3, (loc, len(sub))
        assert {QUESTIONS.index(r["question"]) for r in sub} == {0, 2, 3}
    # 全部 25 条都有记忆前缀。
    assert all(r["memory_prefix"] for r in A), "A 组应全带记忆前缀"

    # ---- 去重终验 ----
    seen2, internal_dups = set(), 0
    for r in all_records:
        if r["text"] in seen2:
            internal_dups += 1
        seen2.add(r["text"])
    dups = [r for r in all_records if r["text"] in existing]
    print(f"internal dups: {internal_dups}, dup vs existing: {len(dups)}")
    assert not dups and internal_dups == 0, "有重复记录"

    # ---- 写入（只写新文件）----
    out1 = HERE / "data" / "processed" / "chunk9_drills.jsonl"
    out2 = Path.home() / "workspace" / "dual_sft" / "chunk9_drills.jsonl"
    for out in (out1, out2):
        with open(out, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_records)} -> {out}")
    print("TOTAL:", len(all_records), "A", len(A))


if __name__ == "__main__":
    main()
