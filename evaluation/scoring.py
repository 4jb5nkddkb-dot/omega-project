#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""世界自查评分：问题集 + 严格字符串评分（port 自 dual_sft/world_self_probe.py）。

这是 causal_probe / loop / q_probe 共用的判卷逻辑，与 NumPy 版
一字不差——改这里等于改探针测量的东西：
  Q0 你在什么地方。      Q1 在场人物有谁。
  Q2 随身物品有什么。    Q3 现在是白天还是夜晚。
评分只看生成续文，不看提示。Q0 的固定答案 bug（三次都答“你在花果山”）
和 Q2/Q3 跷跷板（一个 3/3 另一个就 0/3）都是这套评分先抓出来的。
"""
from __future__ import annotations

from .world_state import CHARACTERS, ITEMS, LOCATIONS

QUESTIONS = ["你在什么地方。", "在场人物有谁。",
             "随身物品有什么。", "现在是白天还是夜晚。"]

LOCATIONS_LIST = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]


def score(state, qi: int, cont: str):
    """返回 (pass, note)。state 是 evaluation.world_state.WorldState。"""
    if qi == 0:
        true = state.visitor_location
        others = [loc for loc in LOCATIONS if loc != true]
        if f"你在{true}" in cont and not any(f"你在{o}" in cont for o in others):
            return True, ""
        if true in cont and not any(o in cont for o in others):
            return True, ""
        missing = "缺真地点" if true not in cont else ""
        wrong = [o for o in others if f"你在{o}" in cont or o in cont]
        return False, f"{missing} 误提{wrong}".strip()
    if qi == 1:
        present = state.present_characters()
        absent = [c for c in CHARACTERS if c not in present]
        miss = [c for c in present if c not in cont]
        extra = [c for c in absent if c in cont]
        if not miss and not extra:
            return True, ""
        return False, f"缺{miss} 多{extra}"
    if qi == 2:
        carried = state.carried_items()
        if not carried:
            ok = ("无" in cont or "没有" in cont) and not any(i in cont for i in ITEMS)
            return ok, "" if ok else "空随身未答对"
        miss = [i for i in carried if i not in cont]
        extra = [i for i in ITEMS if i not in carried and i in cont]
        if not miss and not extra:
            return True, ""
        return False, f"缺{miss} 多{extra}"
    true = state.day_night()
    other = "夜晚" if true == "白天" else "白天"
    if true in cont and other not in cont:
        return True, ""
    return False, f"缺{true}或误提{other}"


def anchor_prompt(state, persona_model: str = "欧米伽") -> str:
    """到达叙述用的锚点提示（port 自 dual_sft/world_anchor.py 的最小集：
    人格 omega、无意图时的锚点行 + 状态摘要）。产品侧的 world_anchor.py
    仍在旧仓库，这里只复制探针需要的两行格式。"""
    return f"我是欧米伽。当前人格{persona_model}。\n" + state.summary()
