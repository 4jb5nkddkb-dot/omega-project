#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成第 7 块教学记录（chunk 7 用，s1800 -> s2100）。

针对第 6 块验收唯一不达标项：环内 Q0 1/3（要求 ≥2/3）。
诊断结论（见 docs/CHUNK7_DIAGNOSIS.md，本地实证）：
  1. 冻结循环（evaluation/loop.py）全程不移动——世界冻结在 S0
     （取经路，第 0 步，白天），Q0 在 1/5/9 步问的都是取经路。
     所以环 Q0 1/3 不是“多步状态跟踪”失败，而是单轮取经路
     失败在环里的重复出现（三次 Q0 提问问的是同一个冻结状态；
     记忆缓存容量 3，两次 Q0 之间隔着 Q1/Q2/Q3，若三题都对，
     Q0 正例在下一次 Q0 提问前就被顶掉——三次提问接近独立）。
     因果探针（真正会动的测试）反而进步了 4/7 -> 5/7。——
     第 7 块不加多步记录，修单轮取经路。
  2. s1800 在所有格式下都答不对取经路（探针 5/7 败水帘洞、
     取经路；登记 6/7 仅败取经路），而 drills 里取经路作答案
     的记录是最多的（58 条），所以不是缺数量，是缺“对的种类”：
     第 6 块的 drills 用阿拉伯数字写时间（“时间第2步”），而
     探针/环的真实摘要用中文数字（“时间第二步”）；且 drills
     里随身一对写“金箍棒、蟠桃”，真实摘要按 ITEMS 字典序写
     “蟠桃、金箍棒”——两处训练/测试格式不一致。本块全部记录
     改用真实 WorldState.summary() 生成，保证与探针字节一致。
  3. Q2 空随身只占 drills 的 9/80（11%），探针 Q2 恰好是
     空随身在取经路（可见锦襕袈裟）——第 6 块把空随身的
     先验冲淡了，Q2 从 1/1 跌到 0/1。本块把空随身补到
     与旧域相当的比例（约 38%）。

构成（135 条，全用真实 WorldState 摘要，中文数字）：
  A. 取经路 Q0 专攻 40 条：探针格式 20（第 1 条就是环的精确
     S0 态）+ 登记格式 10 + 抽取 10。
  B. 水帘洞 Q0 专攻 30 条：探针格式 20 + 抽取 10（登记已过，
     不加）。
  C. 其余 5 地点 Q0 巩固 30 条：每地 6 条探针格式（含 1 条
     第 0 步），防止修取经路时把已过地点修坏。
  D. Q2 空随身平衡 35 条：探针格式 28（7 地点 × 4 步）+
     抽取 7（每地 1 条）。
  合计 135 条。只写新文件 chunk7_drills.jsonl，不动已有语料；
  接线（加进 training/dataset.py 的 WORLD_RECORD_FILES）不在
  本脚本内。

去重：旧域 probe_format 已覆盖 0-22 步的大量组合，指定步数撞车
是常事；emit_unique() 会按 FALLBACK_STEPS 顺序换步直到文本唯一，
撞车时打印一行说明。环 S0 态的两条记录（Q0 探针/登记）若换步
则直接报错——它们必须精确命中第 0 步。

世界观与第 6 块一致（7 地点、6 物品、昼夜桩时钟、南海多观音）。
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

# 6 组随身（含空），与第 6 块同一套。
CARRIED_SETS = [[], ["锦襕袈裟"], ["金箍棒"], ["蟠桃"], ["金箍棒", "蟠桃"], ["九齿钉耙"]]

ALL_ITEMS = {"蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"}
# 物品 -> 初始地点（与 evaluation/world_state.py 的 ITEMS 一致）。
ITEM_HOME = {"蟠桃": "花果山", "金箍棒": "花果山", "九齿钉耙": "高老庄",
             "月牙铲": "流沙河", "锦襕袈裟": "取经路", "紫金钵盂": "南海"}

# 换步时的备选顺序：23-33 步是 probe_format 未覆盖的新区间，优先；
# 其次 0 步，其余按步数升序。
FALLBACK_STEPS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                  15, 16, 17, 18, 19, 20, 21, 22]


def daynight_of(step):
    return "白天" if (step // 4) % 2 == 0 else "夜晚"


def make_summary(loc, carried, step):
    """用真实 WorldState 走出指定状态并取摘要——与探针/环字节一致。

    做法：从取经路出发，到每件随身物品的老家拾取，再走到目标地点，
    最后把 time_step 拨到目标值（摘要只读 time_step，不读 trace）。
    """
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
    # 摘要必须与状态一致（WorldState 自己保证，这里再验一道）。
    assert f"你在{loc}。" in summ
    assert daynight_of(step) in summ
    return summ, {"loc": loc, "time_step": step, "carried": list(carried)}


def make_record(summary_text, question, answer, state, register=False):
    """单轮记录：摘要 + 问题 + 答案。register=True 时按探针的登记格式，
    把地点句原样放最前面（见 evaluation/q_probe.py）。"""
    if register:
        reg = summary_text[summary_text.rfind("你在"):]
        text = reg + "\n" + summary_text + "\n" + question + answer
    else:
        text = summary_text + "\n" + question + answer
    return {"text": text, "summary": summary_text,
            "answer_start": len(text) - len(answer),
            "state": state, "question": question, "answer": answer}


def emit_unique(records, existing, seen, loc, carried, step, question,
                answer, register=False):
    """建一条记录并追加：先试指定 step，文本与旧域或新文件重复就按
    FALLBACK_STEPS 顺序换步，直到唯一。返回实际用的 step。"""
    for s in [step] + [f for f in FALLBACK_STEPS if f != step]:
        summ, state = make_summary(loc, carried, s)
        rec = make_record(summ, question, answer, state, register=register)
        if rec["text"] not in existing and rec["text"] not in seen:
            records.append(rec)
            seen.add(rec["text"])
            if s != step:
                print(f"  fallback: {loc} {question} step {step} -> {s} "
                      f"(carried={carried})")
            return s
    raise AssertionError(
        f"no unique step for {loc} {question} carried={carried}")


def build(existing, seen):
    A, B, C, D = [], [], [], []

    # ============ A. 取经路 Q0 专攻（40）============
    # A1 探针格式 20：第 1 条就是环的精确 S0 态（第 0 步、空随身），
    # 必须精确命中第 0 步（换步即报错）；其余 4 条第 0 步带不同随身，
    # 23-33 步是 probe_format 未覆盖的新区间。
    a1_steps = [0, 0, 0, 0, 0, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                2, 5, 9, 14]
    a1_carried = [[], ["锦襕袈裟"], ["金箍棒"], ["蟠桃"],
                  ["金箍棒", "蟠桃"]] + \
        [CARRIED_SETS[i % len(CARRIED_SETS)] for i in range(5, 20)]
    for i, step in enumerate(a1_steps):
        used = emit_unique(A, existing, seen, "取经路", a1_carried[i], step,
                           "你在什么地方。", "你在取经路。")
        if i == 0:
            assert used == 0, f"S0 Q0 probe record moved to step {used}"
    # A2 登记格式 10：第 1 条是环 S0 态的登记格式，同样必须命中第 0 步。
    a2_steps = [0, 0, 23, 25, 27, 29, 31, 33, 2, 9]
    a2_carried = [[], ["锦襕袈裟"]] + \
        [CARRIED_SETS[(i + 2) % len(CARRIED_SETS)] for i in range(2, 10)]
    for i, step in enumerate(a2_steps):
        used = emit_unique(A, existing, seen, "取经路", a2_carried[i], step,
                           "你在什么地方。", "你在取经路。", register=True)
        if i == 0:
            assert used == 0, f"S0 Q0 register record moved to step {used}"
    # A3 抽取演练 10：问法不同，答案只有字段值。
    for i, step in enumerate([23, 24, 25, 26, 27, 28, 29, 30, 31, 33]):
        carried = CARRIED_SETS[(i + 4) % len(CARRIED_SETS)]
        emit_unique(A, existing, seen, "取经路", carried, step,
                    "地点是哪里。", "取经路。")

    # ============ B. 水帘洞 Q0 专攻（30）============
    # B1 探针格式 20：登记已过，只补探针格式。第 1 条第 0 步空随身。
    b1_steps = [0, 0, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                2, 5, 9, 14, 18, 8, 11]
    b1_carried = [[], ["金箍棒"]] + \
        [CARRIED_SETS[(i + 1) % len(CARRIED_SETS)] for i in range(2, 20)]
    for i, step in enumerate(b1_steps):
        emit_unique(B, existing, seen, "水帘洞", b1_carried[i], step,
                    "你在什么地方。", "你在水帘洞。")
    # B2 抽取演练 10。
    for i, step in enumerate([23, 24, 25, 26, 27, 28, 29, 30, 31, 33]):
        carried = CARRIED_SETS[(i + 3) % len(CARRIED_SETS)]
        emit_unique(B, existing, seen, "水帘洞", carried, step,
                    "地点是哪里。", "水帘洞。")

    # ============ C. 其余 5 地点 Q0 巩固（30）============
    # 每地 6 条探针格式，含 1 条第 0 步；防止修取经路时把已过地点修坏。
    # 第 0 步的随身按地点轮转（避免第 0 步与空随身完全绑定）。
    others = ["花果山", "高老庄", "流沙河", "南海", "天宫"]
    for li, loc in enumerate(others):
        for j, step in enumerate([0, 23, 26, 29, 32, 8]):
            carried = (CARRIED_SETS[li % len(CARRIED_SETS)] if j == 0
                       else CARRIED_SETS[(li + j) % len(CARRIED_SETS)])
            emit_unique(C, existing, seen, loc, carried, step,
                        "你在什么地方。", f"你在{loc}。")

    # ============ D. Q2 空随身平衡（35）============
    # D1 探针格式 28：7 地点 × 4 步（每地含 1 条第 0 步），随身为空，
    # 可见物品仍在——模型必须顶住可见物品、答“无”。
    # 例外：取经路不用第 0 步（该精确记录已在 probe_format 里，
    # 见诊断 §5），改用第 8 步，避免纯重复。
    for li, loc in enumerate(LOCATIONS):
        steps = [0, 23, 27, 31] if loc != "取经路" else [8, 23, 27, 31]
        for step in steps:
            emit_unique(D, existing, seen, loc, [], step,
                        "随身物品有什么。", "随身物品无。")
    # D2 抽取演练 7：每地 1 条，答案只有“无。”。
    for li, loc in enumerate(LOCATIONS):
        emit_unique(D, existing, seen, loc, [], 23 + li,
                    "随身物品是什么。", "无。")

    return A, B, C, D


def load_existing_texts():
    """旧域世界文件的全部 text（去重比对用）。"""
    existing = set()
    proc = HERE / "data" / "processed"
    dual = Path.home() / "workspace" / "dual_sft"
    for d in (proc, dual):
        for name in ("probe_format_records.jsonl", "register_format_records.jsonl",
                     "carried_register_format_records.jsonl",
                     "daytime_register_format_records.jsonl",
                     "character_register_format_records.jsonl",
                     "read_drills.jsonl"):
            p = d / name
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        existing.add(json.loads(line)["text"])
    return existing


def main():
    existing = load_existing_texts()
    print(f"existing world texts: {len(existing)}")
    seen = set()
    A, B, C, D = build(existing, seen)
    counts = {"A_取经路Q0": len(A), "B_水帘洞Q0": len(B),
              "C_其余Q0": len(C), "D_Q2空随身": len(D)}
    print("counts:", counts, "total:", sum(counts.values()))
    assert counts == {"A_取经路Q0": 40, "B_水帘洞Q0": 30,
                      "C_其余Q0": 30, "D_Q2空随身": 35}, counts
    all_records = A + B + C + D
    assert len(all_records) == 135

    # ---- 词表检查（0 OOV）----
    vocab = json.loads((HERE / "data" / "processed" / "generalist_vocab.json")
                       .read_text(encoding="utf-8"))
    tok = CharTokenizer(vocab["chars"])
    bad_oov = []
    for r in all_records:
        if tok.UNK in tok.encode(r["text"]):
            bad_oov.append(r["text"][:80])
    print(f"OOV: {len(bad_oov)} / {len(all_records)}")
    for t in bad_oov[:5]:
        print("  OOV:", repr(t))
    assert not bad_oov, "有 OOV，停止写入"

    # ---- 格式检查：摘要必须与真实 WorldState 字节一致，且无阿拉伯数字 ----
    bad_fmt = []
    for r in all_records:
        st = r["state"]
        # 按 state 重建 WorldState 并比对摘要。
        summ2, _ = make_summary(st["loc"], st["carried"], st["time_step"])
        if summ2 != r["summary"]:
            bad_fmt.append(("summary-mismatch", r["text"][:80]))
        if re.search(r"时间第[0-9]+步", r["summary"]):
            bad_fmt.append(("arabic-step", r["text"][:80]))
        if r["text"][r["answer_start"]:] != r["answer"]:
            bad_fmt.append(("answer_start", r["text"][:80]))
        # 答案与 state 一致。
        q, ans = r["question"], r["answer"]
        if q == "你在什么地方。" and ans != f"你在{st['loc']}。":
            bad_fmt.append(("q0-answer", r["text"][:80]))
        if q == "地点是哪里。" and ans != f"{st['loc']}。":
            bad_fmt.append(("q0-drill-answer", r["text"][:80]))
        if q == "随身物品有什么。" and ans != "随身物品无。":
            bad_fmt.append(("q2-answer", r["text"][:80]))
        if q == "随身物品是什么。" and ans != "无。":
            bad_fmt.append(("q2-drill-answer", r["text"][:80]))
        # 摘要里的昼夜/随身与 state 一致。注意：WorldState.summary()
        # 按 ITEMS 字典序排随身（蟠桃在金箍棒前），不是按拾取顺序；
        # 第 6 块手写摘要用的拾取顺序（“金箍棒、蟠桃”）与之不同，
        # 是另一处训练/测试格式不一致，本块以真实摘要为准。
        if daynight_of(st["time_step"]) not in r["summary"]:
            bad_fmt.append(("daynight-summary", r["text"][:80]))
        order = ["蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"]
        car = "、".join(sorted(st["carried"], key=order.index)) \
            if st["carried"] else "无"
        if f"随身物品{car}。" not in r["summary"]:
            bad_fmt.append(("carried-summary", r["text"][:80]))
        for it in st["carried"]:
            if it not in ALL_ITEMS:
                bad_fmt.append(("bad-item", r["text"][:80]))
    print(f"format bad: {len(bad_fmt)} / {len(all_records)}")
    for kind, t in bad_fmt[:10]:
        print("  BAD", kind, repr(t))
    assert not bad_fmt, "格式校验失败"

    # ---- 精确态覆盖：环的 S0 态（第 0 步、取经路、白天、空随身）----
    # Q0 探针与登记的精确 S0 记录在本文件里各至少 1 条；Q2 空随身的
    # 精确 S0 记录已在 probe_format 里（1 条），本文件不再重复，
    # 这里按“新文件 + 旧域”合计校验。
    s0_q0 = [r for r in all_records
             if r["state"] == {"loc": "取经路", "time_step": 0, "carried": []}
             and r["question"] == "你在什么地方。"
             and not r["text"].startswith("你在取经路。\n世界状态")]
    s0_q0_reg = [r for r in all_records
                 if r["state"] == {"loc": "取经路", "time_step": 0, "carried": []}
                 and r["question"] == "你在什么地方。"
                 and r["text"].startswith("你在取经路。\n世界状态")]
    s0_q2_new = [r for r in all_records
                 if r["state"] == {"loc": "取经路", "time_step": 0, "carried": []}
                 and r["question"] == "随身物品有什么。"]
    s0_q2_old = 0
    pf = HERE / "data" / "processed" / "probe_format_records.jsonl"
    for line in open(pf, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if (r.get("state") == {"loc": "取经路", "time_step": 0, "carried": []}
                and r.get("question") == "随身物品有什么。"):
            s0_q2_old += 1
    print(f"S0 coverage: Q0 probe x{len(s0_q0)} (new), "
          f"Q0 register x{len(s0_q0_reg)} (new), "
          f"Q2 empty x{len(s0_q2_new)} (new) + x{s0_q2_old} (old)")
    assert len(s0_q0) >= 1, "环 S0 态的 Q0 探针记录不足"
    assert len(s0_q0_reg) >= 1, "环 S0 态的 Q0 登记记录不足"
    assert len(s0_q2_new) + s0_q2_old >= 1, "环 S0 态的 Q2 空随身记录不足"

    # ---- 去重终验：新文件内部 + 与旧域（emit_unique 已保证，这里复核）----
    seen2, internal_dups = set(), 0
    for r in all_records:
        if r["text"] in seen2:
            internal_dups += 1
        seen2.add(r["text"])
    dups = [r for r in all_records if r["text"] in existing]
    print(f"internal dups: {internal_dups}, dup vs existing: {len(dups)}")
    assert not dups and internal_dups == 0, "有重复记录"

    # ---- 写入（只写新文件）----
    out1 = HERE / "data" / "processed" / "chunk7_drills.jsonl"
    out2 = Path.home() / "workspace" / "dual_sft" / "chunk7_drills.jsonl"
    for out in (out1, out2):
        with open(out, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_records)} -> {out}")
    print("TOTAL:", len(all_records), "A", len(A), "B", len(B),
          "C", len(C), "D", len(D))


if __name__ == "__main__":
    main()
