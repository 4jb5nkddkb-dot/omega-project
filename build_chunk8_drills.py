#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成第 8 块教学记录（chunk 8 用，s2100 -> s2400）。

针对第 7 块的副作用：环 Q0 修好了（1/3 -> 3/3），但探针/登记/
因果 Q0 退化成固定默认“取经路”，环 Q3 归零（3/3 -> 0/3，全答
“夜晚”）。诊断结论（见 docs/CHUNK8_DIAGNOSIS.md，本地实证）：
  1. Q0 固定默认的根子在答案 token 分布：第 7 块新加的 Q0 答案
     token 里取经路占 40.1%（220/548 字），水帘洞 29.2%，其余
     五地各 5.5-6.6%；而第 7 块之前是均衡的（每地约 1/7）。
     新一块的答案分布决定行为走向——第 6 块 drills 均衡，
     s1800 的错误就是分散的；第 7 块 drills 偏取经路，s2100 的
     错误就收敛成固定默认。——第 8 块不再单点重压，Q0 答案
     严格 11 条/地（A 组），Q3 答案 25 白/24 夜（B 组）。
  2. Q3 归零在数据里找不到原因：Q3 答案 token 白 1236 / 夜 1176
     （均衡），第 7 块加了 0 条 Q3 记录；第 7 块摘要昼夜白 78 /
     夜 57（白天占多数）；地点×昼夜无伪相关。候选解释是读字段
     的本事被 Q0/Q2 的走捷径带坏了，和/或环里记忆缓存满了
     （s2100 的环 Q0/Q1/Q2 全对，到 Q3 时缓存恰好装满），和/或
     Q3 在第 0 步几乎没有直接训练支撑（全世界域只有 2 条，且
     都是精确 S0 态）。——第 8 块 B 组直接钻 Q3：探针/登记/
     抽取三种格式，7 地点 × 昼夜均衡，含每地第 0 步 Q3。
  3. 因果链第 9 步（@取经路，时间 4，夜晚）错答南海，是整批
     “默认取经路”里唯一的反向错答——固定默认不是绝对的。
     C 组把因果链 7 个（地点，时间）原样各钻一条，不赌机制。

构成（133 条，全用真实 WorldState 摘要，中文数字）：
  A. Q0 平衡 77 条：A1 探针格式 42（7 地 × 6，步 [0,2,5,9,13,18]，
     含每地第 0 步与第 2 步——环与探针各自的精确步数）；
     A2 登记格式 21（7 地 × 3，步 [0,2,13]）；A3 抽取 14
     （7 地 × 2，步 [2,14]）。答案 11 条/地，严格均衡。
  B. Q3 昼夜 49 条：B1 探针格式 28（7 地 × 4，步 [0,5,9,13]，
     每地 2 白/2 夜，含每地第 0 步 Q3）；B2 登记格式 14
     （7 地 × 2，步 [2,6]，每地 1 白/1 夜，答案前置——与
     daytime_register 同款）；B3 抽取 7（7 地 × 1，步 10/14
     轮换，4 白/3 夜）。答案 25 白/24 夜。
  C. 因果原态 7 条：因果链 7 个（地点，时间）原样各 1 条探针格式
     Q0：（取经路，0）、（水帘洞，1）、（高老庄，2）、
     （流沙河，3）、（取经路，4）、（南海，5）、（南海，5，
     换随身）。随身轮转，避免与旧域撞车。
  合计 133 条。Q2 0 条（探针 1/1、环 3/3，不动它；Q2/Q3 跷跷板
  警告不要在修 Q3 的同时喂 Q2）。Q1 0 条（环 3/3；探针 0/1 是
  老样子，不在本块范围；另发现 read_drills 的 Q1 问法
  “在场人物是谁。”与探针“在场人物有谁。”不一致，留给以后
  的语料清理）。只写新文件 chunk8_drills.jsonl，不动已有语料；
  接线（加进 training/dataset.py 的 WORLD_RECORD_FILES）不在
  本脚本内。

去重：emit_unique() 先在指定步数换随身（6 组随身轮流试），
都撞车才按 FALLBACK_STEPS 换步；23-33 步是第 7 块的区间，
本块换步优先用 34-44 步（probe_format 未覆盖的新区间），
其次 0-33 步。指定第 0 步或第 2 步的记录若被换步，构建器
直接报错——环与探针的精确步数必须命中。

step-0 的随身选择：第 7 块已在第 0 步放了取经路 Q0 探针 ×5
（随身 []、锦襕袈裟、金箍棒、蟠桃、金箍棒+蟠桃）与登记 ×2
（[]、锦襕袈裟），以及水帘洞探针 ×2（[]、金箍棒）；旧域
probe_format 里精确 S0 态（取经路，第 0 步，空随身）已有
Q2 与 Q3 探针格式各 1 条。emit_unique 的同歩换随身会绕开
这些已占组合；第 0 步记录若 6 组随身全撞车才换步（届时断言
会拦下）。
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

# 6 组随身（含空），与第 6/7 块同一套。
CARRIED_SETS = [[], ["锦襕袈裟"], ["金箍棒"], ["蟠桃"], ["金箍棒", "蟠桃"], ["九齿钉耙"]]

ALL_ITEMS = {"蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"}
# 物品 -> 初始地点（与 evaluation/world_state.py 的 ITEMS 一致）。
ITEM_HOME = {"蟠桃": "花果山", "金箍棒": "花果山", "九齿钉耙": "高老庄",
             "月牙铲": "流沙河", "锦襕袈裟": "取经路", "紫金钵盂": "南海"}

# 换步时的备选顺序：34-44 步是 probe_format 与 chunk7 都未覆盖的新区间，
# 优先；其次 0-33 步。
FALLBACK_STEPS = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                  28, 29, 30, 31, 32, 33]


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


def make_record(summary_text, question, answer, state, register=None):
    """单轮记录：摘要 + 问题 + 答案。
    register='q0'：Q0 登记格式，地点句原样放最前面（见
      evaluation/q_probe.py）。
    register='q3'：Q3 登记格式，答案句原样放最前面（与
      daytime_register_format_records.jsonl 同款）。
    register=None：探针格式。抽取格式也是 None（问法不同而已）。
    """
    if register == "q0":
        reg = summary_text[summary_text.rfind("你在"):]
        text = reg + "\n" + summary_text + "\n" + question + answer
    elif register == "q3":
        text = answer + "\n" + summary_text + "\n" + question + answer
    else:
        text = summary_text + "\n" + question + answer
    return {"text": text, "summary": summary_text,
            "answer_start": len(text) - len(answer),
            "state": state, "question": question, "answer": answer}


def emit_unique(records, existing, seen, loc, carried, step, question,
                answer, register=None):
    """建一条记录并追加。候选顺序：指定（步，随身）→ 同步换随身
    （6 组轮流试）→ 按 FALLBACK_STEPS 换步（随身不变）。返回实际
    用的 (step, carried)。指定第 0/2 步的记录若被换步，调用方断言拦下。
    """
    cands = [(step, carried)]
    cands += [(step, c) for c in CARRIED_SETS if c != carried]
    cands += [(s, carried) for s in FALLBACK_STEPS if s != step]
    for s, car in cands:
        summ, state = make_summary(loc, car, s)
        rec = make_record(summ, question, answer, state, register=register)
        if rec["text"] not in existing and rec["text"] not in seen:
            records.append(rec)
            seen.add(rec["text"])
            if (s, car) != (step, carried):
                print(f"  moved: {loc} {question} step {step} "
                      f"carried={carried} -> step {s} carried={car}")
            return s, car
    raise AssertionError(
        f"no unique (step, carried) for {loc} {question} carried={carried}")


def build(existing, seen):
    A1, A2, A3, B1, B2, B3, C = [], [], [], [], [], [], []

    # ============ A1. Q0 探针格式（42）：7 地 × 6 ============
    # 步 [0,2,5,9,13,18]：0 是环的精确步数，2 是探针的精确步数，
    # 其余 spread。随身按（地点 + 步序号）轮转。
    for li, loc in enumerate(LOCATIONS):
        for j, step in enumerate([0, 2, 5, 9, 13, 18]):
            carried = CARRIED_SETS[(li + j) % len(CARRIED_SETS)]
            used_step, _ = emit_unique(
                A1, existing, seen, loc, carried, step,
                "你在什么地方。", f"你在{loc}。")
            if step in (0, 2):
                assert used_step == step, (
                    f"A1 {loc} step {step} moved to {used_step}")

    # ============ A2. Q0 登记格式（21）：7 地 × 3 ============
    # 步 [0,2,13]：0 与 2 同上，13 是夜晚步（A1 的 13 也是夜晚，
    # 这里补一个登记格式的夜间样本）。
    for li, loc in enumerate(LOCATIONS):
        for j, step in enumerate([0, 2, 13]):
            carried = CARRIED_SETS[(li + j + 2) % len(CARRIED_SETS)]
            used_step, _ = emit_unique(
                A2, existing, seen, loc, carried, step,
                "你在什么地方。", f"你在{loc}。", register="q0")
            if step in (0, 2):
                assert used_step == step, (
                    f"A2 {loc} step {step} moved to {used_step}")

    # ============ A3. Q0 抽取演练（14）：7 地 × 2 ============
    # 步 [2,14]：2 白天、14 夜晚，各一。答案只有字段值。
    for li, loc in enumerate(LOCATIONS):
        for j, step in enumerate([2, 14]):
            carried = CARRIED_SETS[(li + j + 1) % len(CARRIED_SETS)]
            emit_unique(A3, existing, seen, loc, carried, step,
                        "地点是哪里。", f"{loc}。")

    # ============ B1. Q3 探针格式（28）：7 地 × 4 ============
    # 步 [0,5,9,13]：0 白、5 夜、9 白、13 夜，每地 2 白/2 夜。
    # 第 0 步 Q3：取经路空随身的精确 S0 记录已在旧域 probe_format
    # 里，emit_unique 的同歩换随身会绕开它（改带一件随身）。
    for li, loc in enumerate(LOCATIONS):
        for j, step in enumerate([0, 5, 9, 13]):
            carried = CARRIED_SETS[(li + j) % len(CARRIED_SETS)]
            ans = f"现在是{daynight_of(step)}。"
            used_step, _ = emit_unique(
                B1, existing, seen, loc, carried, step,
                "现在是白天还是夜晚。", ans)
            if step == 0:
                assert used_step == 0, (
                    f"B1 {loc} step 0 moved to {used_step}")

    # ============ B2. Q3 登记格式（14）：7 地 × 2 ============
    # 步 [2,6]：2 白、6 夜，每地 1 白/1 夜。答案前置（与
    # daytime_register 同款）。
    for li, loc in enumerate(LOCATIONS):
        for j, step in enumerate([2, 6]):
            carried = CARRIED_SETS[(li + j + 3) % len(CARRIED_SETS)]
            ans = f"现在是{daynight_of(step)}。"
            emit_unique(B2, existing, seen, loc, carried, step,
                        "现在是白天还是夜晚。", ans, register="q3")

    # ============ B3. Q3 抽取演练（7）：7 地 × 1 ============
    # 步 10/14 轮换：10 白、14 夜，4 白/3 夜。答案只有字段值
    # （与 read_drills 的“昼夜是白天还是夜晚。”同款）。
    for li, loc in enumerate(LOCATIONS):
        step = 10 if li % 2 == 0 else 14
        carried = CARRIED_SETS[(li + 1) % len(CARRIED_SETS)]
        ans = f"{daynight_of(step)}。"
        emit_unique(B3, existing, seen, loc, carried, step,
                    "昼夜是白天还是夜晚。", ans)

    # ============ C. 因果链原态（7）：7 个（地点，时间）各 1 条 ============
    # 因果探针的 7 次 Q0 提问原样：（取经路，0）、（水帘洞，1）、
    # （高老庄，2）、（流沙河，3）、（取经路，4）、（南海，5）、
    # （南海，5）。两个（南海，5）用不同随身区分；其余随身轮转。
    causal_states = [("取经路", 0), ("水帘洞", 1), ("高老庄", 2),
                     ("流沙河", 3), ("取经路", 4), ("南海", 5), ("南海", 5)]
    for i, (loc, step) in enumerate(causal_states):
        carried = CARRIED_SETS[(i + 2) % len(CARRIED_SETS)]
        emit_unique(C, existing, seen, loc, carried, step,
                    "你在什么地方。", f"你在{loc}。")

    return A1, A2, A3, B1, B2, B3, C


def load_existing_texts():
    """旧域世界文件的全部 text（去重比对用，含 chunk7）。"""
    existing = set()
    proc = HERE / "data" / "processed"
    dual = Path.home() / "workspace" / "dual_sft"
    for d in (proc, dual):
        for name in ("probe_format_records.jsonl", "register_format_records.jsonl",
                     "carried_register_format_records.jsonl",
                     "daytime_register_format_records.jsonl",
                     "character_register_format_records.jsonl",
                     "read_drills.jsonl", "chunk7_drills.jsonl"):
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
    A1, A2, A3, B1, B2, B3, C = build(existing, seen)
    counts = {"A1_Q0探针": len(A1), "A2_Q0登记": len(A2), "A3_Q0抽取": len(A3),
              "B1_Q3探针": len(B1), "B2_Q3登记": len(B2), "B3_Q3抽取": len(B3),
              "C_因果原态": len(C)}
    print("counts:", counts, "total:", sum(counts.values()))
    assert counts == {"A1_Q0探针": 42, "A2_Q0登记": 21, "A3_Q0抽取": 14,
                      "B1_Q3探针": 28, "B2_Q3登记": 14, "B3_Q3抽取": 7,
                      "C_因果原态": 7}, counts
    all_records = A1 + A2 + A3 + B1 + B2 + B3 + C
    assert len(all_records) == 133

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
        summ2, _ = make_summary(st["loc"], st["carried"], st["time_step"])
        if summ2 != r["summary"]:
            bad_fmt.append(("summary-mismatch", r["text"][:80]))
        if re.search(r"时间第[0-9]+步", r["summary"]):
            bad_fmt.append(("arabic-step", r["text"][:80]))
        if r["text"][r["answer_start"]:] != r["answer"]:
            bad_fmt.append(("answer_start", r["text"][:80]))
        q, ans = r["question"], r["answer"]
        if q == "你在什么地方。" and ans != f"你在{st['loc']}。":
            bad_fmt.append(("q0-answer", r["text"][:80]))
        if q == "地点是哪里。" and ans != f"{st['loc']}。":
            bad_fmt.append(("q0-drill-answer", r["text"][:80]))
        if q == "现在是白天还是夜晚。" and ans != f"现在是{daynight_of(st['time_step'])}。":
            bad_fmt.append(("q3-answer", r["text"][:80]))
        if q == "昼夜是白天还是夜晚。" and ans != f"{daynight_of(st['time_step'])}。":
            bad_fmt.append(("q3-drill-answer", r["text"][:80]))
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

    # ---- 答案分布终验：A 组 Q0 每地 11 条；加上 C 组（因果链本身
    # 问取经路、南海各两次）后 Q0 共 84 条，各地 11–13 条 ----
    q0_by_loc = {}
    for r in all_records:
        if r["question"] in ("你在什么地方。", "地点是哪里。"):
            m = re.search(r"(花果山|水帘洞|高老庄|流沙河|取经路|南海|天宫)",
                          r["answer"])
            q0_by_loc[m.group(1)] = q0_by_loc.get(m.group(1), 0) + 1
    print("Q0 answers by loc:", q0_by_loc)
    assert q0_by_loc == {"花果山": 11, "水帘洞": 12, "高老庄": 12,
                         "流沙河": 12, "取经路": 13, "南海": 13,
                         "天宫": 11}, q0_by_loc
    q3_dn = {"白天": 0, "夜晚": 0}
    for r in all_records:
        if r["question"] in ("现在是白天还是夜晚。", "昼夜是白天还是夜晚。"):
            q3_dn[daynight_of(r["state"]["time_step"])] += 1
    print("Q3 answers day/night:", q3_dn)
    assert q3_dn == {"白天": 25, "夜晚": 24}, q3_dn

    # ---- 精确步覆盖：每地第 0 步 Q0 探针/登记、Q3 探针各 1 条；
    # 每地第 2 步 Q0 探针各 1 条（探针的精确步数）----
    for loc in LOCATIONS:
        n0_probe = [r for r in all_records
                    if r["state"]["loc"] == loc and r["state"]["time_step"] == 0
                    and r["question"] == "你在什么地方。"
                    and not r["text"].startswith("你在" + loc + "。\n世界状态")]
        n0_reg = [r for r in all_records
                  if r["state"]["loc"] == loc and r["state"]["time_step"] == 0
                  and r["question"] == "你在什么地方。"
                  and r["text"].startswith("你在" + loc + "。\n世界状态")]
        n0_q3 = [r for r in all_records
                 if r["state"]["loc"] == loc and r["state"]["time_step"] == 0
                 and r["question"] == "现在是白天还是夜晚。"]
        n2_probe = [r for r in all_records
                    if r["state"]["loc"] == loc and r["state"]["time_step"] == 2
                    and r["question"] == "你在什么地方。"
                    and not r["text"].startswith("你在" + loc + "。\n世界状态")]
        assert len(n0_probe) >= 1, f"{loc} 缺第 0 步 Q0 探针"
        assert len(n0_reg) >= 1, f"{loc} 缺第 0 步 Q0 登记"
        assert len(n0_q3) >= 1, f"{loc} 缺第 0 步 Q3 探针"
        assert len(n2_probe) >= 1, f"{loc} 缺第 2 步 Q0 探针"
    print("step coverage: 每地第 0 步 Q0 探针/登记/Q3 探针 + 第 2 步 Q0 探针齐全")

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
    out1 = HERE / "data" / "processed" / "chunk8_drills.jsonl"
    out2 = Path.home() / "workspace" / "dual_sft" / "chunk8_drills.jsonl"
    for out in (out1, out2):
        with open(out, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_records)} -> {out}")
    print("TOTAL:", len(all_records),
          "A", len(A1) + len(A2) + len(A3),
          "B", len(B1) + len(B2) + len(B3), "C", len(C))


if __name__ == "__main__":
    main()
