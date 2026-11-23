#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Q0“先读后答”教学记录（chunk 6 用，s1500 -> s1800）。

方法见 llm-project/docs/Q0_TEACHING_DESIGN.md（先读后答：答案区 loss +
抽取演练 + 支架问答）；数据构成见 docs/CHUNK5_PLAN.md §2.2（365 条）。
仿 ~/workspace/dual_sft/build_minimal_pairs.py 的写法：同格式 jsonl，
含 text/summary/answer_start/state/question/answer；过词表检查，0 OOV。

只写新文件，不动现有语料：
  - ~/workspace/llm-project/data/processed/read_drills.jsonl
  - ~/workspace/dual_sft/read_drills.jsonl（同内容副本）
训练器要读到它，需把 read_drills.jsonl 加进 training/dataset.py 的
WORLD_RECORD_FILES，或把这 365 条并入 probe_format_records.jsonl——
那是接线步骤，不在本脚本内。

世界观（与现有语料一致）：
  7 地点：花果山/水帘洞/高老庄/流沙河/取经路/南海/天宫
  6 物品：蟠桃、金箍棒、九齿钉耙、月牙铲、锦襕袈裟、紫金钵盂
  昼夜：白天当 (step//4)%2==0，否则夜晚
  南海在场人物多一个观音（6 人），其余 5 人。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tokenizer"))

from char_tokenizer import CharTokenizer

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]

PRESENT = {
    "南海": "孙悟空、唐僧、猪八戒、沙僧、白龙马、观音",
    "花果山": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "水帘洞": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "高老庄": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "流沙河": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "取经路": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "天宫": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
}

VISIBLE = {
    "花果山": "蟠桃、金箍棒",
    "水帘洞": "无",
    "高老庄": "九齿钉耙",
    "流沙河": "月牙铲",
    "取经路": "锦襕袈裟",
    "南海": "紫金钵盂",
    "天宫": "无",
}

# 6 组随身（含空），供抽取演练 Q2 / 直接问答 Q2
CARRIED_SETS = [[], ["锦襕袈裟"], ["金箍棒"], ["蟠桃"], ["金箍棒", "蟠桃"], ["九齿钉耙"]]
# 6 组随身（全非空），供支架问答 Q2（空随身已是 3/3，不加支架）
CARRIED_NONEMPTY = [["锦襕袈裟"], ["金箍棒"], ["蟠桃"], ["金箍棒", "蟠桃"], ["九齿钉耙"], ["月牙铲"]]

ALL_ITEMS = {"蟠桃", "金箍棒", "九齿钉耙", "月牙铲", "锦襕袈裟", "紫金钵盂"}


def daynight_of(step):
    return "白天" if (step // 4) % 2 == 0 else "夜晚"


def summary(loc, carried, step):
    dn = daynight_of(step)
    car = "、".join(carried) if carried else "无"
    return (f"世界状态。在场人物{PRESENT[loc]}。可见物品{VISIBLE[loc]}。"
            f"随身物品{car}。时间第{step}步{dn}。你在{loc}。")


def fmt_list(items):
    return "、".join(items) if items else "无"


def make_record(summary_text, question, answer, state):
    text = summary_text + "\n" + question + answer
    return {"text": text, "summary": summary_text,
            "answer_start": len(text) - len(answer),
            "state": state, "question": question, "answer": answer}


def build():
    drills, scaff, direct = [], [], []

    # ============ 抽取演练 drills（155）============
    # Q0 70：10 模板 x 7 地点。问法与探针不同（“地点是哪里。”），答案只有字段值。
    q0_drill_steps = [2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    for b, step in enumerate(q0_drill_steps):
        carried = CARRIED_SETS[b % len(CARRIED_SETS)]
        for loc in LOCATIONS:
            summ = summary(loc, carried, step)
            drills.append(make_record(
                summ, "地点是哪里。", f"{loc}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q3 20：10 模板 x 2 昼夜。
    for b in range(10):
        loc = LOCATIONS[b % len(LOCATIONS)]
        carried = CARRIED_SETS[b % len(CARRIED_SETS)]
        day_step = 2 + (b % 3) * 8
        night_step = 6 + (b % 3) * 8
        for step in (day_step, night_step):
            dn = daynight_of(step)
            summ = summary(loc, carried, step)
            drills.append(make_record(
                summ, "昼夜是白天还是夜晚。", f"{dn}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q2 30：5 模板 x 6 随身组合（含空）。
    for b in range(5):
        loc = LOCATIONS[(b * 2) % len(LOCATIONS)]
        step = 3 + b
        for carried in CARRIED_SETS:
            summ = summary(loc, carried, step)
            drills.append(make_record(
                summ, "随身物品是什么。", f"{fmt_list(carried)}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q1 35：5 模板 x 7 地点，人物表随地点变（南海多观音）。
    q1_drill_steps = [2, 5, 9, 13, 17]
    for b, step in enumerate(q1_drill_steps):
        carried = CARRIED_SETS[b % len(CARRIED_SETS)]
        for loc in LOCATIONS:
            summ = summary(loc, carried, step)
            drills.append(make_record(
                summ, "在场人物是谁。", f"{PRESENT[loc]}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))

    # ============ 支架问答 scaffolded（120）============
    # Q0 70：10 模板 x 7 地点。答案 = 字段值 + 完整答案（先抄再答）。
    q0_scaff_steps = [4, 7, 11, 12, 15, 16, 19, 20, 21, 22]
    for b, step in enumerate(q0_scaff_steps):
        carried = CARRIED_SETS[(b + 3) % len(CARRIED_SETS)]
        for loc in LOCATIONS:
            summ = summary(loc, carried, step)
            scaff.append(make_record(
                summ, "你在什么地方。", f"{loc}。你在{loc}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q3 20：10 模板 x 2 昼夜。
    for b in range(10):
        loc = LOCATIONS[(b + 2) % len(LOCATIONS)]
        carried = CARRIED_SETS[(b + 1) % len(CARRIED_SETS)]
        day_step = 2 + (b % 3) * 8
        night_step = 6 + (b % 3) * 8
        for step in (day_step, night_step):
            dn = daynight_of(step)
            summ = summary(loc, carried, step)
            scaff.append(make_record(
                summ, "现在是白天还是夜晚。", f"{dn}。现在是{dn}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q2 30：5 模板 x 6 随身组合，全非空。
    for b in range(5):
        loc = LOCATIONS[(b * 3 + 1) % len(LOCATIONS)]
        step = 4 + b
        for carried in CARRIED_NONEMPTY:
            summ = summary(loc, carried, step)
            items = fmt_list(carried)
            scaff.append(make_record(
                summ, "随身物品有什么。", f"{items}。随身物品{items}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))

    # ============ 直接问答 direct（90，新模板）============
    # Q0 50：10 模板 x 5 地点（轮转），标准问答，不加支架。
    q0_direct_steps = [23, 24, 25, 26, 27, 28, 29, 30, 31, 33]
    for b, step in enumerate(q0_direct_steps):
        carried = CARRIED_SETS[(b + 5) % len(CARRIED_SETS)]
        for k in range(5):
            loc = LOCATIONS[(b + k) % len(LOCATIONS)]
            summ = summary(loc, carried, step)
            direct.append(make_record(
                summ, "你在什么地方。", f"你在{loc}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q3 20：10 模板 x 2 昼夜。
    for b in range(10):
        loc = LOCATIONS[(b + 4) % len(LOCATIONS)]
        carried = CARRIED_SETS[(b + 2) % len(CARRIED_SETS)]
        day_step = 2 + (b % 3) * 8
        night_step = 6 + (b % 3) * 8
        for step in (day_step, night_step):
            dn = daynight_of(step)
            summ = summary(loc, carried, step)
            direct.append(make_record(
                summ, "现在是白天还是夜晚。",
                f"现在是{dn}。",
                {"loc": loc, "time_step": step, "carried": list(carried)}))
    # Q2 20：20 条，随身轮转 6 组，地点轮转 7 个，步数 8..15。
    for i in range(20):
        carried = CARRIED_SETS[i % len(CARRIED_SETS)]
        loc = LOCATIONS[i % len(LOCATIONS)]
        step = 8 + (i % 8)
        summ = summary(loc, carried, step)
        direct.append(make_record(
            summ, "随身物品有什么。", f"随身物品{fmt_list(carried)}。",
            {"loc": loc, "time_step": step, "carried": list(carried)}))

    return drills, scaff, direct


def main():
    drills, scaff, direct = build()
    counts = {"drills": len(drills), "scaffolded": len(scaff), "direct": len(direct)}
    print("counts:", counts, "total:", sum(counts.values()))
    assert counts == {"drills": 155, "scaffolded": 120, "direct": 90}, counts
    all_records = drills + scaff + direct
    assert len(all_records) == 365

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

    # ---- 一致性校验：答案与状态摘要一致 ----
    bad_cons = []
    for r in all_records:
        st = r["state"]
        summ, ans, q = r["summary"], r["answer"], r["question"]
        # 地点：摘要尾句与 state.loc 一致
        if f"你在{st['loc']}。" not in summ:
            bad_cons.append(("loc-summary", r["text"][:80]))
        # 昼夜：摘要昼夜与 step 一致
        if daynight_of(st["time_step"]) not in summ:
            bad_cons.append(("daynight-summary", r["text"][:80]))
        # 随身：摘要随身与 state.carried 一致
        if f"随身物品{fmt_list(st['carried'])}。" not in summ:
            bad_cons.append(("carried-summary", r["text"][:80]))
        # 物品只在合法集合内
        for it in st["carried"]:
            if it not in ALL_ITEMS:
                bad_cons.append(("bad-item", r["text"][:80]))
        if st["loc"] not in LOCATIONS:
            bad_cons.append(("bad-loc", r["text"][:80]))
        # answer_start 指向答案开头
        if r["text"][r["answer_start"]:] != ans:
            bad_cons.append(("answer_start", r["text"][:80]))
        # 答案里的地点/昼夜/随身与 state 一致（按问题类型）
        if q == "你在什么地方。" and f"你在{st['loc']}。" not in ans:
            bad_cons.append(("q0-answer", r["text"][:80]))
        if q == "地点是哪里。" and ans != f"{st['loc']}。":
            bad_cons.append(("q0-drill-answer", r["text"][:80]))
        if q == "现在是白天还是夜晚。" and daynight_of(st["time_step"]) not in ans:
            bad_cons.append(("q3-answer", r["text"][:80]))
        if q == "昼夜是白天还是夜晚。" and ans != f"{daynight_of(st['time_step'])}。":
            bad_cons.append(("q3-drill-answer", r["text"][:80]))
        if q in ("随身物品有什么。", "随身物品是什么。"):
            if fmt_list(st["carried"]) not in ans:
                bad_cons.append(("q2-answer", r["text"][:80]))
        if q == "在场人物是谁。" and ans != f"{PRESENT[st['loc']]}。":
            bad_cons.append(("q1-drill-answer", r["text"][:80]))
    print(f"consistency bad: {len(bad_cons)} / {len(all_records)}")
    for kind, t in bad_cons[:10]:
        print("  BAD", kind, repr(t))
    assert not bad_cons, "一致性校验失败"

    # ---- 去重：与现有 probe_format_records.jsonl 的 text 不重复 ----
    existing = set()
    for path in (HERE / "data" / "processed" / "probe_format_records.jsonl",
                 Path.home() / "workspace" / "dual_sft" / "probe_format_records.jsonl"):
        if path.exists():
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["text"])
    dups = [r for r in all_records if r["text"] in existing]
    print(f"dup vs existing probe: {len(dups)} / {len(all_records)}")
    for r in dups[:5]:
        print("  DUP:", repr(r["text"][:80]))
    # 内部去重
    seen, internal_dups = set(), 0
    for r in all_records:
        if r["text"] in seen:
            internal_dups += 1
        seen.add(r["text"])
    print(f"internal dups: {internal_dups}")
    assert not dups and internal_dups == 0, "有重复记录"

    # ---- 写入（只写新文件）----
    out1 = HERE / "data" / "processed" / "read_drills.jsonl"
    out2 = Path.home() / "workspace" / "dual_sft" / "read_drills.jsonl"
    for out in (out1, out2):
        with open(out, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_records)} -> {out}")
    print("TOTAL:", len(all_records), "drills", len(drills),
          "scaffolded", len(scaff), "direct", len(direct))


if __name__ == "__main__":
    main()
