#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成世界工具调用 SFT 数据补丁（world9 用）：修规范状态缺口。

背景（world7/world8 两次栽在同一陷阱的根因，2026-10-27 已验）：
v7 生成器的 validate() 上限是 128 token（含 BOS/EOS），而 3 个失败组合
的真实规范样本长 129/131/130 token，于是规范样本在校验步被丢弃；
校验后补齐只按（工具，目标，措辞）查覆盖、不看摘要，从全部 setup 里
找了个能塞进 128 的轻装版本顶替——覆盖全绿，规范状态却被悄悄换掉。
模型于是只学到（措辞 × 地点 × 状态形状）三元记忆，扫掠的规范状态
（新鲜状态 + 最少移动，时间第一步、随身无物）对新措辞是分布外的。

本轮（v8 生成器）改两处，缺一不可：
  1. 上限 128 → 136（覆盖最长规范样本 131 并留余量；训练器
     tool_example 全文编码、pad_batch 按批内最长补齐，本来就不截断，
     所以上限只卡生成器，不卡训练）。
  2. 规范状态断言：每个（工具 × 目标 × 措辞）必须有一条规范状态样本
     在最终数据内，覆盖键带摘要（工具，目标，措辞，摘要）逐条核对；
     缺失则直接从规范 setup 建样并校验，校验不过就硬报错——不许用
     非规范 setup 顶替，不许静默通过。

其余与 v7 完全一致：枚举状态 × 工具 × 用户措辞，状态由 setup 动作
序列真实走到，工具结果真实执行取串，叙述句与执行后状态一致；失败类
样本保留少量；0 UNK、0 坏调用、0 重复。

规范 setup 定义（扫掠路径的字面定义：新鲜状态 + 最少移动）：
  查看-current：取经路直接看，其他地点先走一次去该地；
  查看-other：取经路看他处，看取经路则先去花果山；
  移动：从取经路出发，移回取经路则先去花果山；
  拾取：新鲜 + 最少移动到物品初始地点，随身无物；
  对话：新鲜（队伍成员在取经路），观音则先去南海。

用法：python3 build_world_tool_data_v8.py
  写出 world_tool_records_v9.jsonl 与 WORLD_TOOL_DATA_V9_REPORT.md。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/omega-chat")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import world_state as ws
from agent_runtime import Protocol

HERE = Path(__file__).resolve().parent
OUT_JSONL = HERE / "world_tool_records_v9.jsonl"
OUT_REPORT = HERE / "WORLD_TOOL_DATA_V9_REPORT.md"
MAX_LEN = 136  # v9：上限 128 → 136（摘要改地点压后，长度不变），规范样本最长 131，必须能通过校验
P = Protocol()

# -- 用户措辞池（{dest}/{loc}/{item}/{name} 为槽位）-------------------------
MOVE_USERS = ["去{dest}看看。", "到{dest}去。", "我们去{dest}。",
              "去{dest}。", "去{dest}走走。", "往{dest}去。",
              "前往{dest}。", "去{dest}瞧瞧。", "奔{dest}去。"]
LOOK_CUR_USERS = ["看看。", "看看周围。", "看看这里。", "周围有什么。",
                  "看看周围有什么。", "瞧瞧。", "四处看看。", "看一看。",
                  "打量一下。", "环顾四周。", "这里怎么样。",
                  "这是什么地方。", "瞅瞅。", "望一望。"]
LOOK_OTHER_USERS = ["看看{loc}。", "查看{loc}。", "瞧瞧{loc}。",
                    "望一望{loc}。"]
TAKE_USERS = ["拾起{item}。", "把{item}拾起来。", "拿{item}。",
              "把这里的{item}拾起来。", "拿起{item}。", "捡起{item}。",
              "把{item}带上。", "收起{item}。"]
TAKE_FAIL_USERS = ["拾起{item}。"]
TALK_USERS = ["和{name}说说话。", "跟{name}聊聊。", "问问{name}。",
              "和{name}说话。", "找{name}聊聊。", "与{name}交谈。",
              "拜见{name}。"]
TALK_FAIL_USERS = ["和{name}说说话。"]

MOVE_ROT = 4   # 每 setup×目的地取几个移动措辞（轮换）
TALK_ROT = 3   # 每 setup×人物取几个对话措辞（轮换）

SETUPS = [
    [],
    [("移动", "花果山")],
    [("移动", "水帘洞")],
    [("移动", "高老庄")],
    [("移动", "流沙河")],
    [("移动", "南海")],
    [("移动", "天宫")],
    [("移动", "花果山"), ("移动", "水帘洞")],
    [("移动", "高老庄"), ("移动", "流沙河")],
    [("拾取", "锦襕袈裟"), ("移动", "花果山")],
    [("拾取", "锦襕袈裟"), ("移动", "花果山"), ("拾取", "蟠桃")],
    [("移动", "南海"), ("拾取", "紫金钵盂")],
    [("移动", "花果山"), ("移动", "南海")],
    [("移动", "南海"), ("移动", "取经路")],
    [("拾取", "锦襕袈裟"), ("移动", "南海")],
    [("移动", "天宫"), ("移动", "花果山")],
    # 轻装回访：花果山只剩一件可见物品，摘要与结果更短，让长措辞的
    # look-current 样本能塞进 128 token（花果山满载时长措辞会超 1–6 token）。
    [("移动", "花果山"), ("拾取", "蟠桃"), ("移动", "水帘洞"),
     ("移动", "花果山")],
    [("移动", "花果山"), ("拾取", "蟠桃"), ("拾取", "金箍棒"),
     ("移动", "水帘洞"), ("移动", "花果山")],
]


def state_after(setup):
    st = ws.WorldState()
    tools = st.tools()
    for name, arg in setup:
        tools[name](arg)
    return st


def narrate(tool, arg, result, st_before, st_after):
    if tool == "移动":
        if result.startswith("你来到"):
            return f"你来到{arg}。{ws.LOCATIONS[arg]}"
        return result
    if tool == "查看":
        loc = arg.strip() or st_before.visitor_location
        if arg.strip() and arg.strip() != st_before.visitor_location:
            return f"{loc}。{ws.LOCATIONS[loc]}"
        return f"你在{loc}。"
    if tool == "拾取":
        if result.startswith("你拾起了"):
            return f"你拾起了{arg}。"
        return result
    if tool == "对话":
        return result
    raise ValueError(tool)


def make_sample(tool, arg, setup, summary, user, result, narr):
    return {"tool": tool, "arg": arg, "setup": [list(s) for s in setup],
            "summary": summary, "user": user, "result": result, "narr": narr}


def gen_for_setup(setup, move_ph, look_cur_ph, look_other_ph, take_ph,
                  talk_ph):
    """一个 setup 下的基础样本（轮换已应用）。"""
    base = state_after(setup)
    out = []
    si = SETUPS.index(setup)
    # 移动：每目的地轮换 MOVE_ROT 个措辞
    for di, dest in enumerate(ws.LOCATIONS):
        for k in range(MOVE_ROT):
            u = move_ph[(si + di + k) % len(move_ph)].format(dest=dest)
            st = state_after(setup)
            summary = st.summary()
            result = st.tools()["移动"](dest)
            out.append(make_sample("移动", dest, setup, summary, u, result,
                                   narrate("移动", dest, result, base, st)))
    # 查看当前：全部措辞
    for u in look_cur_ph:
        st = state_after(setup)
        summary = st.summary()
        result = st.tools()["查看"]("")
        out.append(make_sample("查看", "", setup, summary, u, result,
                               narrate("查看", "", result, base, st)))
    # 查看他处：全部模板 × 其余地点
    for loc in ws.LOCATIONS:
        if loc == base.visitor_location:
            continue
        for tmpl in look_other_ph:
            st = state_after(setup)
            summary = st.summary()
            user = tmpl.format(loc=loc)
            result = st.tools()["查看"](loc)
            out.append(make_sample("查看", loc, setup, summary, user, result,
                                   narrate("查看", loc, result, base, st)))
    # 拾取：在场全部措辞 + 不在场/随身失败类 1 个措辞
    present = base.present_items()
    absent = [i for i in ws.ITEMS if i not in present
              and base.item_loc[i] != ws.CARRIED]
    carried = base.carried_items()
    for item in present:
        for u in take_ph:
            st = state_after(setup)
            summary = st.summary()
            user = u.format(item=item)
            result = st.tools()["拾取"](item)
            out.append(make_sample("拾取", item, setup, summary, user, result,
                                   narrate("拾取", item, result, base, st)))
    for item in absent + carried:
        st = state_after(setup)
        summary = st.summary()
        user = TAKE_FAIL_USERS[0].format(item=item)
        result = st.tools()["拾取"](item)
        out.append(make_sample("拾取", item, setup, summary, user, result,
                               narrate("拾取", item, result, base, st)))
    # 对话：在场轮换 + 不在场失败类 1 个措辞
    present_chars = base.present_characters()
    absent_chars = [c for c in ws.CHARACTERS if c not in present_chars]
    for ci, name in enumerate(present_chars):
        for k in range(TALK_ROT):
            u = talk_ph[(si + ci + k) % len(talk_ph)].format(name=name)
            st = state_after(setup)
            summary = st.summary()
            result = st.tools()["对话"](name)
            out.append(make_sample("对话", name, setup, summary, u, result,
                                   narrate("对话", name, result, base, st)))
    for name in absent_chars:
        st = state_after(setup)
        summary = st.summary()
        user = TALK_FAIL_USERS[0].format(name=name)
        result = st.tools()["对话"](name)
        out.append(make_sample("对话", name, setup, summary, user, result,
                               narrate("对话", name, result, base, st)))
    return out


def find_setup(pred):
    """找一个满足 pred(base_state) 的 setup；没有则返回 None。"""
    for setup in SETUPS:
        if pred(state_after(setup)):
            return setup
    return None


def coverage_of(samples):
    """覆盖集合：(tool, target, user)；look-current 的 target 取摘要地点。"""
    cov = set()
    for s in samples:
        if s["tool"] == "查看" and s["arg"] == "":
            # v9：地点压到摘要末尾，不再是“世界状态。你在X。”开头
            m = re.search(r"你在([^。]+)。", s["summary"])
            target = m.group(1) if m else "?"
        else:
            target = s["arg"]
        cov.add((s["tool"], target, s["user"]))
    return cov


def build():
    import dual_transformer_numpy as dtn
    # v9：只借词表做措辞池校验（旧线 ckpt 已物理删除，改用全能线最新产品；
    # 词表同为冻结 4622，校验等价）
    _, ck = dtn.DualTransformerNP.load_checkpoint(
        "/home/hatch/workspace/dual_sft/out/generalist_s59700.npz")
    tok = dtn.tokenizer_from_meta(ck["meta"])

    def vocab_ok(s):
        return all(i != tok.UNK for i in tok.encode(s, add_bos=False))

    # -- 措辞池词表校验 ------------------------------------------------------
    dropped_ph = {}
    def filt(pool):
        kept, drop = [], []
        for u in pool:
            probe = u.format(dest="花果山", loc="花果山", item="蟠桃", name="孙悟空")
            (kept if vocab_ok(probe) else drop).append(u)
        return kept, drop
    look_cur_v, d = filt(LOOK_CUR_USERS)
    dropped_ph["查看当前"] = d
    look_other_v, d = filt(LOOK_OTHER_USERS)
    dropped_ph["查看他处"] = d
    move_v, d = filt(MOVE_USERS)
    dropped_ph["移动"] = d
    take_v, d = filt(TAKE_USERS)
    dropped_ph["拾取"] = d
    talk_v, d = filt(TALK_USERS)
    dropped_ph["对话"] = d
    assert len(look_cur_v) >= 8, f"查看可用措辞不足 8：{look_cur_v}"
    print(f"措辞池（校验后）：查看当前 {len(look_cur_v)}、查看他处 "
          f"{len(look_other_v)}、移动 {len(move_v)}、拾取 "
          f"{len(take_v)}、对话 {len(talk_v)}")
    print(f"因 UNK 丢弃的措辞：{dropped_ph}")

    samples = []
    for setup in SETUPS:
        samples.extend(gen_for_setup(setup, move_v, look_cur_v, look_other_v,
                                     take_v, talk_v))
    print(f"基础生成：{len(samples)} 条")

    # -- 覆盖核对 + 定向补齐 --------------------------------------------------
    def need_look_cur():
        cov = coverage_of(samples)
        miss = []
        for loc in ws.LOCATIONS:
            for u in look_cur_v:
                if ("查看", loc, u) not in cov:
                    miss.append(("查看", loc, u))
        return miss

    def need_look_other():
        cov = coverage_of(samples)
        miss = []
        for loc in ws.LOCATIONS:
            for tmpl in look_other_v:
                u = tmpl.format(loc=loc)
                if ("查看", loc, u) not in cov:
                    miss.append(("查看", loc, u))
        return miss

    def need_move():
        cov = coverage_of(samples)
        miss = []
        for dest in ws.LOCATIONS:
            for tmpl in move_v:
                u = tmpl.format(dest=dest)
                if ("移动", dest, u) not in cov:
                    miss.append(("移动", dest, u))
        return miss

    def need_take():
        cov = coverage_of(samples)
        miss = []
        for item in ws.ITEMS:
            for tmpl in take_v:
                u = tmpl.format(item=item)
                if ("拾取", item, u) not in cov:
                    miss.append(("拾取", item, u))
        return miss

    def need_talk():
        cov = coverage_of(samples)
        miss = []
        for name in ws.CHARACTERS:
            for tmpl in talk_v:
                u = tmpl.format(name=name)
                if ("对话", name, u) not in cov:
                    miss.append(("对话", name, u))
        return miss

    topped = 0
    for _ in range(3):  # 至多 3 轮补齐（每轮后重算）
        missing = []
        for tool, target, user in (need_look_cur() + need_look_other()
                                   + need_move() + need_take() + need_talk()):
            missing.append((tool, target, user))
        if not missing:
            break
        for tool, target, user in missing:
            if tool == "查看" and user in look_cur_v:
                setup = find_setup(lambda st: st.visitor_location == target)
            elif tool == "查看":
                setup = find_setup(lambda st: st.visitor_location != target)
            elif tool == "移动":
                setup = find_setup(lambda st: st.visitor_location != target)
            elif tool == "拾取":
                setup = find_setup(lambda st: target in st.present_items())
            elif tool == "对话":
                setup = find_setup(lambda st: target in st.present_characters())
            else:
                setup = None
            if setup is None:
                continue
            base = state_after(setup)
            st = state_after(setup)
            summary = st.summary()
            if tool == "查看" and user in look_cur_v:
                result = st.tools()["查看"]("")
                samples.append(make_sample("查看", "", setup, summary, user,
                                           result,
                                           narrate("查看", "", result, base, st)))
            elif tool == "查看":
                result = st.tools()["查看"](target)
                samples.append(make_sample("查看", target, setup, summary, user,
                                           result,
                                           narrate("查看", target, result, base, st)))
            elif tool == "移动":
                result = st.tools()["移动"](target)
                samples.append(make_sample("移动", target, setup, summary, user,
                                           result,
                                           narrate("移动", target, result, base, st)))
            elif tool == "拾取":
                result = st.tools()["拾取"](target)
                samples.append(make_sample("拾取", target, setup, summary, user,
                                           result,
                                           narrate("拾取", target, result, base, st)))
            elif tool == "对话":
                result = st.tools()["对话"](target)
                samples.append(make_sample("对话", target, setup, summary, user,
                                           result,
                                           narrate("对话", target, result, base, st)))
            topped += 1
        print(f"补齐一轮：新增 {topped} 条（累计）")
    missing = (need_look_cur() + need_look_other() + need_move()
               + need_take() + need_talk())
    assert not missing, f"覆盖仍有缺失（校验前）：{missing[:10]}"
    print(f"覆盖核对（校验前）：全部通过（补齐 {topped} 条）")

    # -- 组装、去重、校验（与第六轮同口径）-------------------------------------
    from agent_runtime import ToolRouter
    router = ToolRouter(ws.WorldState().tools(), tok)
    kept, per_tool, dropped = [], Counter(), Counter()
    seen = set()

    def validate(s):
        """样本 -> kept 记录；不合格返回 None（原因计入 dropped）。"""
        call = f"{P.call_open}{s['tool']}{P.name_end}{s['arg']}{P.call_close}"
        wrapped = f"{P.result_open}{s['result']}{P.result_close}"
        text = s["summary"] + "\n" + s["user"] + call + wrapped + s["narr"]
        if text in seen:
            dropped["dup"] += 1
            return None
        seen.add(text)
        ids = tok.encode(text, add_bos=True, add_eos=True)
        if any(i == tok.UNK for i in ids):
            dropped["unk"] += 1
            return None
        if len(ids) > MAX_LEN:
            dropped[f"toolong_{s['tool']}"] += 1
            return None
        call_start = len(s["summary"]) + 1 + len(s["user"])
        call_end = call_start + len(call)
        narr_start = call_end + len(wrapped)
        if router.dispatch(text[call_start:call_end]) is None:
            dropped["badcall"] += 1
            return None
        return {"text": text, "call_start": call_start,
                "call_end": call_end, "narr_start": narr_start,
                "tool": s["tool"], "arg": s["arg"],
                "setup": s["setup"], "user": s["user"]}

    for s in samples:
        r = validate(s)
        if r is not None:
            kept.append(r)
            per_tool[r["tool"]] += 1

    def kept_cov():
        return coverage_of([{"tool": r["tool"], "arg": r["arg"],
                             "summary": r["text"].split("\n")[0],
                             "user": r["user"]} for r in kept])

    def build_one(tool, target, user, setup):
        """从指定 setup 为缺失组合定向建一条样本（不校验）。"""
        base = state_after(setup)
        st = state_after(setup)
        summary = st.summary()
        if tool == "查看" and user in look_cur_v:
            result = st.tools()["查看"]("")
            return make_sample("查看", "", setup, summary, user, result,
                               narrate("查看", "", result, base, st))
        if tool == "查看":
            result = st.tools()["查看"](target)
            return make_sample("查看", target, setup, summary, user, result,
                               narrate("查看", target, result, base, st))
        if tool == "移动":
            result = st.tools()["移动"](target)
            return make_sample("移动", target, setup, summary, user, result,
                               narrate("移动", target, result, base, st))
        if tool == "拾取":
            result = st.tools()["拾取"](target)
            return make_sample("拾取", target, setup, summary, user, result,
                               narrate("拾取", target, result, base, st))
        if tool == "对话":
            result = st.tools()["对话"](target)
            return make_sample("对话", target, setup, summary, user, result,
                               narrate("对话", target, result, base, st))
        return None

    def missing_on_kept():
        cov = kept_cov()
        miss = []
        for loc in ws.LOCATIONS:
            for u in look_cur_v:
                if ("查看", loc, u) not in cov:
                    miss.append(("查看", loc, u))
        for loc in ws.LOCATIONS:
            for tmpl in look_other_v:
                if ("查看", loc, tmpl.format(loc=loc)) not in cov:
                    miss.append(("查看", loc, tmpl.format(loc=loc)))
        for dest in ws.LOCATIONS:
            for tmpl in move_v:
                if ("移动", dest, tmpl.format(dest=dest)) not in cov:
                    miss.append(("移动", dest, tmpl.format(dest=dest)))
        for item in ws.ITEMS:
            for tmpl in take_v:
                if ("拾取", item, tmpl.format(item=item)) not in cov:
                    miss.append(("拾取", item, tmpl.format(item=item)))
        for name in ws.CHARACTERS:
            for tmpl in talk_v:
                if ("对话", name, tmpl.format(name=name)) not in cov:
                    miss.append(("对话", name, tmpl.format(name=name)))
        return miss

    # 校验后复核：长度过滤可能吃掉个别组合（v7 是 128 上限吃掉规范样本），
    # 逐个从全部 setup 里找能通过校验的版本补齐；补不上直接报错。
    # 注意：本轮补齐仍不区分摘要，真正的规范状态保障在后面的规范断言。
    for _ in range(3):
        miss = missing_on_kept()
        if not miss:
            break
        added = 0
        for tool, target, user in miss:
            for setup in SETUPS:
                if tool == "查看" and user in look_cur_v:
                    if state_after(setup).visitor_location != target:
                        continue
                elif tool == "查看":
                    if state_after(setup).visitor_location == target:
                        continue
                elif tool == "移动":
                    if state_after(setup).visitor_location == target:
                        continue
                elif tool == "拾取":
                    if target not in state_after(setup).present_items():
                        continue
                elif tool == "对话":
                    if target not in state_after(setup).present_characters():
                        continue
                r = validate(build_one(tool, target, user, setup))
                if r is not None:
                    kept.append(r)
                    per_tool[r["tool"]] += 1
                    added += 1
                    break
        print(f"校验后补齐：本轮新增 {added} 条，剩余缺失待复核")
    miss = missing_on_kept()
    assert not miss, f"覆盖仍有缺失（校验后）：{miss[:10]}"
    print("覆盖核对（校验后）：全部通过 ✓")

    # -- 规范状态断言（v8 新增，world7/world8 陷阱的针对性修复）---------
    # 覆盖键带摘要：每个（工具 × 目标 × 措辞）必须有一条规范状态样本。
    # 缺失则直接从规范 setup 建样并校验；校验不过就硬报错，绝不允许用
    # 非规范 setup 顶替（v7 的补齐逻辑就是在这里悄悄换掉规范状态的）。
    def canonical_setup(tool, target, is_cur):
        """规范 setup：新鲜状态 + 最少移动（扫掠路径的字面定义）。"""
        if tool == "查看" and is_cur:
            return [] if target == "取经路" else [("移动", target)]
        if tool == "查看":
            return [] if target != "取经路" else [("移动", "花果山")]
        if tool == "移动":
            return [] if target != "取经路" else [("移动", "花果山")]
        if tool == "拾取":
            home = ws.ITEMS[target]
            return [] if home == "取经路" else [("移动", home)]
        if tool == "对话":
            return [] if ws.CHARACTERS[target]["party"] else [("移动", "南海")]
        raise ValueError(tool)

    def kept_has(tool, arg, user, summary):
        for r in kept:
            if (r["tool"] == tool and r["arg"] == arg and r["user"] == user
                    and r["text"].split("\n")[0] == summary):
                return True
        return False

    canon_grid = []
    for loc in ws.LOCATIONS:
        for u in look_cur_v:
            canon_grid.append(("查看", "", loc, u, True))
    for loc in ws.LOCATIONS:
        for tmpl in look_other_v:
            canon_grid.append(("查看", loc, loc, tmpl.format(loc=loc), False))
    for dest in ws.LOCATIONS:
        for tmpl in move_v:
            canon_grid.append(("移动", dest, dest, tmpl.format(dest=dest), False))
    for item in ws.ITEMS:
        for tmpl in take_v:
            canon_grid.append(("拾取", item, item, tmpl.format(item=item), False))
    for name in ws.CHARACTERS:
        for tmpl in talk_v:
            canon_grid.append(("对话", name, name, tmpl.format(name=name), False))

    canon_present = canon_added = 0
    canon_fail = []
    for tool, arg, target, user, is_cur in canon_grid:
        setup = canonical_setup(tool, target, is_cur)
        summary = state_after(setup).summary()
        if kept_has(tool, arg, user, summary):
            canon_present += 1
            continue
        r = validate(build_one(tool, target, user, setup))
        if r is None:
            canon_fail.append((tool, target, user, setup, summary))
            continue
        kept.append(r)
        per_tool[r["tool"]] += 1
        canon_added += 1
    assert not canon_fail, (
        "规范状态样本未通过校验（不许用非规范 setup 顶替，须先修上限或"
        f"措辞）：{canon_fail[:5]}")
    print(f"规范状态断言：网格 {len(canon_grid)} 项，已有 {canon_present} 条，"
          f"本阶段补齐 {canon_added} 条，缺失 0 ✓")

    # 此前失败的 3 个组合必须在最终数据内（look-current，规范状态版）
    for loc, user in [("花果山", "周围有什么。"), ("花果山", "看看周围有什么。"),
                      ("南海", "看看周围有什么。")]:
        setup = canonical_setup("查看", loc, True)
        summary = state_after(setup).summary()
        assert kept_has("查看", "", user, summary), \
            f"此前失败组合规范样本缺失：{loc} {user}"
        n_tok = len(tok.encode(
            summary + "\n" + user + "《查看》】" + "〔"
            + state_after(setup).tools()["查看"]("") + "〕" + f"你在{loc}。",
            add_bos=True, add_eos=True))
        print(f"此前失败组合规范样本 ✓：{loc}+{user}（{n_tok} token，"
              f"上限 {MAX_LEN}）")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    r0 = kept[0]
    w = [0.0] * len(r0["text"])
    for j in range(r0["call_start"], r0["call_end"]):
        w[j] = 1.0
    for j in range(r0["narr_start"], len(r0["text"])):
        w[j] = 1.0
    assert sum(w[:r0["call_start"]]) == 0.0
    assert sum(w[r0["call_end"]:r0["narr_start"]]) == 0.0
    assert sum(w[r0["call_start"]:r0["call_end"]]) > 0.0
    assert sum(w[r0["narr_start"]:]) > 0.0
    maxlen = max(len(tok.encode(r["text"], add_bos=True, add_eos=True))
                 for r in kept)

    cov = coverage_of([{"tool": r["tool"], "arg": r["arg"],
                        "summary": r["text"].split("\n")[0],
                        "user": r["user"]} for r in kept])
    cov_lines = []
    for label, tool, targets, pool in [
            ("查看-current", "查看", ws.LOCATIONS, look_cur_v),
            ("查看-other", "查看", ws.LOCATIONS, look_other_v),
            ("移动", "移动", ws.LOCATIONS, move_v),
            ("拾取", "拾取", ws.ITEMS, take_v),
            ("对话", "对话", ws.CHARACTERS, talk_v)]:
        n_ok = sum(1 for t in targets for u in pool
                   if (tool, t, u.format(dest=t, loc=t, item=t, name=t)) in cov)
        cov_lines.append(f"  {label}：{n_ok}/{len(targets) * len(pool)} 组合")
    lines = ["# 世界工具调用数据补丁报告（world9，v8 生成器）",
             "",
             f"总样本：{len(kept)} 条（去重后；第七轮 1658 条）",
             f"分工具：{dict(per_tool)}",
             f"丢弃：{dict(dropped)}",
             f"最长样本：{maxlen} token（含 BOS/EOS，上限 {MAX_LEN}）",
             f"规范状态断言：网格 {len(canon_grid)} 项，已有 {canon_present} 条，"
             f"本阶段补齐 {canon_added} 条，缺失 0 ✓（带摘要的覆盖键，"
             f"不许非规范 setup 顶替）",
             f"权重自查：首条样本摘要/用户/结果区 0、调用+叙述区 1 ✓",
             f"调用合法性：全部样本 dispatch 非 None ✓（生成时即校验）",
             "",
             "## 措辞池（词表校验后）",
             f"查看当前：{look_cur_v}",
             f"查看他处：{look_other_v}",
             f"移动：{move_v}",
             f"拾取：{take_v}",
             f"对话：{talk_v}",
             f"因 UNK 丢弃：{dropped_ph}",
             "",
             "## 覆盖核对（成功类；失败类另计，保留少量）",
             *cov_lines,
             "此前失败组合（look-current，规范状态版）：花果山+周围有什么 ✓、"
             "花果山+看看周围有什么 ✓、南海+看看周围有什么 ✓（129/131/130 "
             f"token，均在 {MAX_LEN} 上限内）",
             "",
             "## 叙述句说明",
             "成功类移动/查看用地点描述合成短句，对话类复用人物固定台词，",
             "失败类复述工具结果原文；无手编状态事实。",
             f"输出：{OUT_JSONL.name}",
             ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return kept


if __name__ == "__main__":
    build()
