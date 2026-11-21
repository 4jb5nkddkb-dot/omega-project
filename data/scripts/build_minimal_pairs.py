#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Q0/Q2/Q3 最小对（minimal-pair）探针记录，针对性修“读可变字段并重复”。
原理：同一底本只改一个字段（地点/随身/昼夜），答案随之变——模型若输出固定答案，
7 个变体里 6 个错，梯度逼它读那个字段。其它字段在底本内保持不变，隔离出目标字段。
写入：probe_format_records.jsonl（探针格式）与各寄存器文件（寄存器格式）。
每条都过词表检查（0 OOV），与现有记录同格式（含 summary/answer_start/state/question/answer）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "omega_v010"))
sys.path.insert(0, str(Path.home() / "workspace" / "omega-chat"))

from dual_transformer_numpy import CharTokenizer
import generalist_pilot_train as gpt

LOCATIONS = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]
# 各地点在场人物（南海多观音，其余 5 人）
PRESENT = {
    "南海": "孙悟空、唐僧、猪八戒、沙僧、白龙马、观音",
    "花果山": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "水帘洞": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "高老庄": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "流沙河": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "取经路": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
    "天宫": "孙悟空、唐僧、猪八戒、沙僧、白龙马",
}
CARRIED_SETS = [[], ["锦襕袈裟"], ["金箍棒"], ["蟠桃"], ["金箍棒", "蟠桃"], ["九齿钉耙"]]

def summary(loc, carried, step, daynight):
    vis = {"花果山": "蟠桃、金箍棒", "水帘洞": "无", "高老庄": "九齿钉耙",
           "流沙河": "月牙铲", "取经路": "锦襕袈裟", "南海": "紫金钵盂", "天宫": "无"}[loc]
    car = "、".join(carried) if carried else "无"
    dn = "白天" if daynight == "day" else "夜晚"
    return (f"世界状态。在场人物{PRESENT[loc]}。可见物品{vis}。随身物品{car}。"
            f"时间第{step}步{dn}。你在{loc}。")

def fmt_list(items):
    return "、".join(items) if items else "无"

def make_record(text, summary_text, question, answer, state, register_line=None):
    if register_line is not None:
        text = register_line + "\n" + text
    return {"text": text, "summary": summary_text,
            "register_line": register_line,
            "answer_start": len(text) - len(answer),
            "state": state, "question": question, "answer": answer}

def main():
    vocab_meta = json.loads(gpt.VOCAB_JSON.read_text(encoding="utf-8"))
    tok = CharTokenizer(vocab_meta["chars"])
    probe_out = []      # probe_format_records.jsonl
    reg_loc_out = []    # register_format_records.jsonl (Q0 寄存器）
    reg_car_out = []    # carried_register_format_records.jsonl (Q2 寄存器）
    reg_day_out = []    # daytime_register_format_records.jsonl (Q3 寄存器）

    # ---- Q0 最小对：10 个底本 × 7 地点 ----
    # 底本：固定随身/时间，只改地点。随身与时间在底本间轮换，增加多样性。
    for b in range(10):
        carried = CARRIED_SETS[b % len(CARRIED_SETS)]
        step = 2 + b  # 2..11
        daynight = "day" if (step // 4) % 2 == 0 else "night"
        for loc in LOCATIONS:
            summ = summary(loc, carried, step, daynight)
            q = "你在什么地方。"
            a = f"你在{loc}。"
            text = summ + "\n" + q + a
            state = {"loc": loc, "time_step": step, "carried": carried}
            probe_out.append(make_record(text, summ, q, a, state))
            reg_loc_out.append(make_record(text, summ, q, a, state,
                                           register_line=f"你在{loc}。"))

    # ---- Q2 最小对：5 个底本 × 6 组随身 ----
    for b in range(5):
        loc = LOCATIONS[(b * 2) % len(LOCATIONS)]
        step = 3 + b
        daynight = "day" if (step // 4) % 2 == 0 else "night"
        for carried in CARRIED_SETS:
            summ = summary(loc, carried, step, daynight)
            q = "随身物品有什么。"
            a = f"随身物品{fmt_list(carried)}。"
            text = summ + "\n" + q + a
            state = {"loc": loc, "time_step": step, "carried": carried}
            probe_out.append(make_record(text, summ, q, a, state))
            reg_car_out.append(make_record(text, summ, q, a, state,
                                           register_line=f"随身物品{fmt_list(carried)}。"))

    # ---- Q3 最小对：10 个底本 × 白天/夜晚 ----
    for b in range(10):
        loc = LOCATIONS[b % len(LOCATIONS)]
        carried = CARRIED_SETS[b % len(CARRIED_SETS)]
        # 白天步：step 0-3, 8-11...；夜晚步：step 4-7, 12-15...
        for daynight, step in (("day", 2 + (b % 3) * 8), ("night", 6 + (b % 3) * 8)):
            summ = summary(loc, carried, step, daynight)
            q = "现在是白天还是夜晚。"
            a = "现在是白天。" if daynight == "day" else "现在是夜晚。"
            text = summ + "\n" + q + a
            state = {"loc": loc, "time_step": step, "carried": carried}
            probe_out.append(make_record(text, summ, q, a, state))
            reg_day_out.append(make_record(text, summ, q, a, state,
                                           register_line=a))

    # ---- 词表检查 + 写入 ----
    def check(records, name):
        bad = 0
        for r in records:
            ids = tok.encode(r["text"])
            if tok.UNK in ids:
                bad += 1
                print(f"  OOV in {name}: {r['text'][:60]!r}")
        print(f"{name}: {len(records)} 条，OOV {bad} 条")
        return bad

    total_bad = 0
    total_bad += check(probe_out, "probe")
    total_bad += check(reg_loc_out, "register_loc")
    total_bad += check(reg_car_out, "register_carried")
    total_bad += check(reg_day_out, "register_daytime")
    assert total_bad == 0, "有 OOV，停止写入"

    def append(path, records):
        with open(HERE / path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"appended {len(records)} -> {path}")

    append("probe_format_records.jsonl", probe_out)
    append("register_format_records.jsonl", reg_loc_out)
    append("carried_register_format_records.jsonl", reg_car_out)
    append("daytime_register_format_records.jsonl", reg_day_out)
    print(f"TOTAL new: probe {len(probe_out)} + reg_loc {len(reg_loc_out)} "
          f"+ reg_car {len(reg_car_out)} + reg_day {len(reg_day_out)}")

if __name__ == "__main__":
    main()
