#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""世界格式训练数据构造器 v2（第二轮）：真实《西游记》段落 + world_prompt 同款状态摘要。

相对上一轮构造器的唯一口径变化：Tier 1 扩容——每段落每个 distinct 锚点
地点各成一组记录、每地点窗口上限 2 -> 4；Tier 2、硬规则（实体一致、
编码无损、探针零污染）与格式（world_state.world_prompt 逐字同款）不变。
输出 world_records_v2.jsonl。

一致性规则（写清，验收时逐条复核）：
  1. 地点分两档，均不编造矛盾状态：
     - Tier 1（地点锚定）：段落含 world_state.LOCATIONS 锚点地点（含下述
       别名映射），摘要地点取段落中首个出现的锚点地点。
     - Tier 2（默认取经路）：段落无锚点地点但含锚点人物，且段落中没有任何
       锚点人物“在/到/回/往/去某地”的陈述（某地为锚点地点或已知他地表），
       摘要地点取状态机自身的默认地点取经路。缺席不等于矛盾；此档宁可
       多跳过（宁严勿松），统计里单列。
  2. 人物：摘要在场人物 = 师徒队伍（跟随规则，与探针摘要同款：全队在场）
     减去 chunk 中明示在别处者；观音只在地点为南海且无矛盾时在场。
     不强求摘要人物全出现在 chunk 中（与探针摘要的全队跟随行为一致）。
  3. 物品：可见物品 = 该地初始物品 + 段落中出现的物品（改记在摘要地点），
     chunk 中明示在别处的物品除外；随身物品一律无。
  4. 时间：time_step 在 0..19 均匀取（覆盖白天/夜晚、中文数字读法），与
     段落无关（段落不含时间步陈述，故不构成矛盾）。
  5. 格式：摘要与拼接一律调用 world_state.WorldState.summary() 与
     world_prompt() 生成，不手写模板——与探针逐字同款。
  6. 无损：每条记录须满足 tok.decode(tok.encode(text)) == text 且编码中无
     UNK，否则跳过（corpus 里的全角逗号等词表外字符会被静默丢弃，绝不
     让这种记录进训练）。
  7. 续写必须是真实原文：chunk 取段落内句子起点的真实切片（不拼模板句），
     长度 >= 30 字；chunk 须含至少一个锚点人物全名；Tier 1 优先取地点锚定
     窗口（chunk 内含摘要地点，统计单列 loc_in_chunk），其次人物锚定窗口；
     总文本（摘要 + 换行 + chunk）<= 126 字，以便训练窗口 128 内容纳 BOS/EOS。
  8. 均衡：Tier 1 每地点上限 600 条；Tier 2 上限 1200 条；每段落最多 2 个
     不重叠窗口；洗牌种子 7。

别名映射（别名所指即该地，西游记语境下无歧义）：
  普陀/落伽/珞珈 -> 南海；天庭/凌霄 -> 天宫；流沙 -> 流沙河；
  高家庄 -> 高老庄。
已知他地表（Tier 2 排除用，段落明示人物在这些地方即跳过）：
  火焰山 女儿国 盘丝洞 通天河 五庄观 车迟国 宝象国 乌鸡国 朱紫国 狮驼岭
  雷音寺 灵山 长安 五行山 五指山 东海 龙宫 地府 西天 白虎岭 平顶山
  莲花洞 金兜洞 黑风山 黄风岭 麒麟山 无底洞 枯松涧 火云洞 号山 陷空山
  比丘国 灭法国 凤仙郡 玉华州 金平府 天竺国

输出：~/workspace/dual_sft/world_records_v2.jsonl，每行一记录
  {text, summary, chunk, visitor_location, present, visible, time_step,
   day_night, para_idx, tier, loc_in_chunk}
同时打印统计。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MIRROR = Path("/home/hatch/workspace/omega-chat")
sys.path.insert(0, str(MIRROR))
sys.path.insert(0, str(HERE))

from dual_masked_loss import load_dual_module
from world_state import (CARRIED, CHARACTERS, ITEMS, LOCATIONS, PARTY,
                         WorldState, world_prompt)

BASE_CKPT = MIRROR / "dual_d128_s69000_params.npz"
CORPUS = MIRROR / "Xiyouji.txt"
OUT = HERE / "world_records_v2.jsonl"
MAX_TEXT = 126
MIN_CHUNK = 30
PER_LOCATION_CAP = 600
TIER2_CAP = 1200
MAX_WINDOWS_PER_PARA = 2
# v2 改动（Tier 1 扩容，v1 其余规则一字不动）：
#  - 每段落不再只取首个地点：段落中每个 distinct 锚点地点各成一组记录
#    （摘要地点 = 该地点），多地点段落（65 段）因此多产记录；
#  - 每地点窗口上限 2 -> 4（先地点锚定、后人物锚定补足），模拟测得
#    Tier 1 170 -> 388 条（loc_in_chunk 79 -> 143）。
#  Tier 2 逻辑、硬规则（实体一致、无损、探针零污染）与 v1 完全相同。
MAX_WINDOWS_PER_LOCATION = 4
# 探针提示（world_probe_world.py CASES 原样复制，污染检查用）：
PROBE_PROMPTS = ("孙悟空回到花果山，", "唐僧师徒走在取经路上，",
                 "猪八戒想起高老庄，", "沙僧想起流沙河，",
                 "孙悟空大闹天宫后，", "唐僧收了孙悟空为徒，",
                 "白龙马驮着唐僧走在取经路上，", "观音菩萨坐在南海，")
VERBS = ("在", "到", "回", "往", "去")
TERMINATORS = ("。", "！", "？", "；", "\n")

ALIAS = {"普陀": "南海", "落伽": "南海", "珞珈": "南海",
         "天庭": "天宫", "凌霄": "天宫", "流沙": "流沙河", "高家庄": "高老庄"}
OTHER_PLACES = ("火焰山", "女儿国", "盘丝洞", "通天河", "五庄观", "车迟国",
                "宝象国", "乌鸡国", "朱紫国", "狮驼岭", "雷音寺", "灵山",
                "长安", "五行山", "五指山", "东海", "龙宫", "地府", "西天",
                "白虎岭", "平顶山", "莲花洞", "金兜洞", "黑风山", "黄风岭",
                "麒麟山", "无底洞", "枯松涧", "火云洞", "号山", "陷空山",
                "比丘国", "灭法国", "凤仙郡", "玉华州", "金平府", "天竺国")


def first_location(para: str):
    """返回 (canonical, pos)：段落中首个出现的锚点地点（含别名）。"""
    best, best_i = None, None
    for loc in LOCATIONS:
        i = para.find(loc)
        if i >= 0 and (best_i is None or i < best_i):
            best, best_i = loc, i
    for alias, canon in ALIAS.items():
        i = para.find(alias)
        if i >= 0 and (best_i is None or i < best_i):
            best, best_i = canon, i
    return best, best_i


def contradiction_map(chunk: str, visitor: str):
    """返回 {人物: 矛盾地点} 与 {物品: 矛盾地点}：chunk 明示在别处者。"""
    char_bad, item_bad = {}, {}
    for name in CHARACTERS:
        for v in VERBS:
            for loc in LOCATIONS:
                if loc != visitor and f"{name}{v}{loc}" in chunk:
                    char_bad[name] = loc
    for item in ITEMS:
        for v in VERBS:
            for loc in LOCATIONS:
                if loc != visitor and f"{item}{v}{loc}" in chunk:
                    item_bad[item] = loc
    return char_bad, item_bad


def tier2_blocked(para: str):
    """Tier 2 排除：段落明示锚点人物在取经路以外的地点（含已知他地）。"""
    for name in CHARACTERS:
        for v in VERBS:
            for loc in list(LOCATIONS) + list(OTHER_PLACES):
                if loc != "取经路" and f"{name}{v}{loc}" in para:
                    return True
    return False


def build_state(visitor: str, para_items, chunk: str, step: int):
    """按 chunk（模型实际所见文本）算矛盾，构造与 chunk 一致的状态。"""
    char_bad, item_bad = contradiction_map(chunk, visitor)
    st = WorldState(visitor_location=visitor)
    for name in PARTY:
        st.char_loc[name] = char_bad.get(name, visitor)
    if "观音" in char_bad:
        st.char_loc["观音"] = char_bad["观音"]
    else:
        st.char_loc["观音"] = "南海"
    for item in ITEMS:
        if item in para_items and item not in item_bad:
            st.item_loc[item] = visitor
        elif item in item_bad:
            st.item_loc[item] = item_bad[item]
    st.time_step = step
    return st


def sentence_start(para: str, pos: int):
    """pos 所在句子的起点（上一个句末标点之后，或段落开头）。"""
    cut = 0
    for t in TERMINATORS:
        i = para.rfind(t, 0, pos)
        if i >= 0:
            cut = max(cut, i + 1)
    return cut


def windows_for(para: str, visitor: str, tier: int):
    """返回至多 2 个不重叠窗口起点：Tier 1 先地点锚定、后人物锚定；
    Tier 2 只人物锚定。"""
    starts = []
    if tier == 1:
        needles = [s for s in list(LOCATIONS) + list(ALIAS)
                   if (s in para and (s in LOCATIONS and s == visitor
                                      or ALIAS.get(s) == visitor))]
        # 地点出现位置（按在段落中的顺序）
        occ = []
        for s in needles:
            i = para.find(s)
            while i >= 0:
                occ.append(i)
                i = para.find(s, i + 1)
        for pos in sorted(occ):
            st0 = sentence_start(para, pos)
            if all(abs(st0 - s) > 10 for s in starts):
                starts.append(st0)
            if len(starts) >= MAX_WINDOWS_PER_PARA:
                return starts
    # 人物锚定窗口（Tier 1 的补充、Tier 2 的全部）
    occ = []
    for name in CHARACTERS:
        i = para.find(name)
        while i >= 0:
            occ.append(i)
            i = para.find(name, i + 1)
    for pos in sorted(occ):
        st0 = sentence_start(para, pos)
        if all(abs(st0 - s) > 10 for s in starts):
            starts.append(st0)
        if len(starts) >= MAX_WINDOWS_PER_PARA:
            break
    return starts


def all_locations(para: str):
    """段落中所有 distinct 锚点地点（标准名），按首次出现顺序。"""
    found = {}
    for loc in LOCATIONS:
        i = para.find(loc)
        if i >= 0:
            found[loc] = i
    for alias, canon in ALIAS.items():
        i = para.find(alias)
        if i >= 0 and canon not in found:
            found[canon] = i
    return [loc for loc, _ in sorted(found.items(), key=lambda kv: kv[1])]


def windows_for_location(para: str, visitor: str):
    """v2 Tier 1：该地点的窗口至多 4 个，先地点锚定、后人物锚定补足。"""
    starts = []
    needles = [s for s in list(LOCATIONS) + list(ALIAS)
               if (s in para and (s in LOCATIONS and s == visitor
                                  or ALIAS.get(s) == visitor))]
    occ = []
    for s in needles:
        i = para.find(s)
        while i >= 0:
            occ.append(i)
            i = para.find(s, i + 1)
    for pos in sorted(occ):
        st0 = sentence_start(para, pos)
        if all(abs(st0 - s) > 10 for s in starts):
            starts.append(st0)
        if len(starts) >= MAX_WINDOWS_PER_LOCATION:
            return starts
    occ = []
    for name in CHARACTERS:
        i = para.find(name)
        while i >= 0:
            occ.append(i)
            i = para.find(name, i + 1)
    for pos in sorted(occ):
        st0 = sentence_start(para, pos)
        if all(abs(st0 - s) > 10 for s in starts):
            starts.append(st0)
        if len(starts) >= MAX_WINDOWS_PER_LOCATION:
            break
    return starts


def main():
    dside = load_dual_module()
    _, ck = dside.DualTransformerNP.load_checkpoint(BASE_CKPT)
    tok = dside.tokenizer_from_meta(ck["meta"])
    text = CORPUS.read_text(encoding="utf-8").replace("\r\n", "\n")
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    rng = np.random.default_rng(7)
    stats = Counter(total=len(paras))
    tier1, tier2 = [], []
    for idx, para in enumerate(paras):
        para_chars = [c for c in CHARACTERS if c in para]
        if not para_chars:
            stats["no_character"] += 1
            continue
        locs = all_locations(para)
        tasks = []
        if not locs:
            if tier2_blocked(para):
                stats["tier2_blocked"] += 1
                continue
            for start in windows_for(para, "取经路", 2):
                tasks.append(("取经路", 2, start))
        else:
            # v2：每个 distinct 地点各成一组 Tier 1 任务（不再只取首个地点）。
            for visitor in locs:
                for start in windows_for_location(para, visitor):
                    tasks.append((visitor, 1, start))
        para_items = [it for it in ITEMS if it in para]
        for visitor, tier, start in tasks:
            step = int(rng.integers(0, 20))
            st = build_state(visitor, para_items, para, step)
            budget = MAX_TEXT - len(st.summary()) - 1
            if budget < MIN_CHUNK:
                stats["summary_too_long"] += 1
                continue
            chunk = para[start:start + budget]
            stable = False
            for _ in range(10):
                st = build_state(visitor, para_items, chunk, step)
                new_budget = MAX_TEXT - len(st.summary()) - 1
                if new_budget >= len(chunk):
                    stable = True
                    break
                chunk = chunk[:new_budget]
                if len(chunk) < MIN_CHUNK:
                    break
            if not stable or len(chunk) < MIN_CHUNK:
                stats["chunk_too_short"] += 1
                continue
            if not any(c in chunk for c in CHARACTERS):
                stats["chunk_no_character"] += 1
                continue
            char_bad, _ = contradiction_map(chunk, visitor)
            present = st.present_characters()
            if any(c in char_bad for c in present):
                stats["contradiction_after_rebuild"] += 1
                continue
            summary = st.summary()
            record_text = world_prompt(st, chunk)
            ids = tok.encode(record_text)
            if any(i == tok.UNK for i in ids) or tok.decode(ids) != record_text:
                stats["lossy"] += 1
                continue
            rec = {
                "text": record_text, "summary": summary, "chunk": chunk,
                "visitor_location": visitor, "present": present,
                "visible": st.present_items(), "time_step": step,
                "day_night": st.day_night(), "para_idx": idx, "tier": tier,
                "loc_in_chunk": (visitor in chunk),
            }
            (tier1 if tier == 1 else tier2).append(rec)
            stats["kept"] += 1
    # 均衡：Tier 1 每地点上限；Tier 2 总量上限；洗牌。
    by_loc = {}
    for r in tier1:
        by_loc.setdefault(r["visitor_location"], []).append(r)
    final1 = []
    for loc, rs in by_loc.items():
        order = rng.permutation(len(rs))
        final1.extend([rs[i] for i in order[:PER_LOCATION_CAP]])
    order = rng.permutation(len(tier2))
    final2 = [tier2[i] for i in order[:TIER2_CAP]]
    final = final1 + final2
    order = rng.permutation(len(final))
    final = [final[i] for i in order]
    # 探针零污染（硬规则）：8 条提示全文、去尾标点版、前 6 字均不得出现在 chunk。
    cores = set()
    for p in PROBE_PROMPTS:
        cores.add(p)
        cores.add(p.rstrip("，。！？；"))
        cores.add(p[:6])
    hits = sum(1 for r in final for c in cores if c in r["chunk"])
    print(f"探针污染检查：{hits} 命中（要求 0）")
    assert hits == 0, f"探针污染 {hits} 条！"
    with OUT.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"段落总数 {stats['total']}，通过筛选 {stats['kept']}，均衡后 {len(final)}")
    print("淘汰：", {k: v for k, v in stats.items()
                   if k not in ("total", "kept")})
    print(f"Tier 1 {len(final1)} 条（各地点：",
          dict(Counter(r["visitor_location"] for r in final1)), ")；",
          f"Tier 2 {len(final2)} 条")
    print("Tier 1 chunk 内含摘要地点：",
          sum(1 for r in final1 if r["loc_in_chunk"]), "/", len(final1))
    cl = np.array([len(r["chunk"]) for r in final])
    print(f"chunk 长度：均值 {cl.mean():.1f}，中位 {np.median(cl):.0f}，"
          f"最小 {cl.min()}，最大 {cl.max()}")
    print("昼夜：", dict(Counter(r["day_night"] for r in final)))
    print("时间步范围：", min(r["time_step"] for r in final),
          "->", max(r["time_step"] for r in final))
    print(f"写出 {OUT}（{len(final)} 条）")
    for r in final[:2]:
        print("--- 样本 ---")
        print(r["text"])


if __name__ == "__main__":
    main()
