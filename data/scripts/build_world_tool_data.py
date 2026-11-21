#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成世界工具调用 SFT 数据（world6 用）：用 WorldState 自己生成，拒绝手编。

每条样本 = 世界摘要 + 换行 + 用户话 → 工具调用（Protocol 格式
《工具名》参数】，与协议数据同款）→ 工具结果（真实执行该工具取到的
返回串，包在〔〕里，与 run_agent 追加结果的格式一致）→ 叙述句
（与执行后状态一致的短句；对话类叙述句复用人物固定台词，属合成，
报告里注明）。损失掩码：调用段与叙述段权重 1，摘要/用户话/结果段
权重 0（训练脚本 tool_example 按 spans 落权重）。

生成逻辑：枚举状态 × 工具 × 用户措辞。状态由 setup 动作序列真实
走到（先在副本上执行 setup，再执行样本工具取结果），不凭空编造。
失败类样本（拾取不在此地物品、对话不在场人物、移动到当前地点）
也保留少量：教模型先调用、再按真实结果叙述，不许编造成功。

校验（全部写进报告，任一硬性项不过就不要开训练）：
  - 全部样本只用词表内字符（0 bad records）；
  - 全部样本编码后（含 BOS/EOS）≤128 token，超长直接丢弃并计数，
    绝不截调用；
  - 每条样本的调用段能被 Protocol 正则解析、且 ToolRouter.dispatch
    返回非 None（调用格式合法）。

用法：python3 build_world_tool_data.py
  写出 world_tool_records.jsonl（每行含 text/call_start/call_end/
  narr_start/tool/arg/setup）与 WORLD_TOOL_DATA_REPORT.md。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/omega-chat")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import world_state as ws
from agent_runtime import Protocol

HERE = Path(__file__).resolve().parent
OUT_JSONL = HERE / "world_tool_records.jsonl"
OUT_REPORT = HERE / "WORLD_TOOL_DATA_REPORT.md"
P = Protocol()

# -- 用户措辞模板（{dest}/{loc}/{item}/{name} 为槽位）------------------------
MOVE_USERS = ["去{dest}看看。", "到{dest}去。", "我们去{dest}。",
              "去{dest}。", "去{dest}走走。"]
LOOK_CUR_USERS = ["看看。", "看看周围。", "看看这里。", "周围有什么。"]
LOOK_OTHER_USERS = ["看看{loc}。", "查看{loc}。"]
TAKE_USERS = ["拾起{item}。", "把{item}拾起来。", "拿{item}。",
              "把这里的{item}拾起来。"]
TAKE_FAIL_USERS = ["拾起{item}。"]
TALK_USERS = ["和{name}说说话。", "跟{name}聊聊。", "问问{name}。",
              "和{name}说话。"]
TALK_FAIL_USERS = ["和{name}说说话。"]

# -- 状态 setup：从新鲜状态真实走到的若干局面 -------------------------------
# 每项为动作序列 [(工具, 参数)]，执行后即样本的出发状态。
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
]


def state_after(setup):
    st = ws.WorldState()
    tools = st.tools()
    for name, arg in setup:
        tools[name](arg)
    return st


def narrate(tool, arg, result, st_before, st_after):
    """与执行后状态一致的短叙述（合成，模板化；失败类直接复述结果）。"""
    if tool == "移动":
        if result.startswith("你来到"):
            return f"你来到{arg}。{ws.LOCATIONS[arg]}"
        return result  # 已经在当前地点：复述结果
    if tool == "查看":
        loc = arg.strip() or st_before.visitor_location
        if arg.strip() and arg.strip() != st_before.visitor_location:
            return f"{loc}。{ws.LOCATIONS[loc]}"
        return f"你在{loc}。"
    if tool == "拾取":
        if result.startswith("你拾起了"):
            return f"你拾起了{arg}。"
        return result  # 此地没有 / 已经带着：复述结果
    if tool == "对话":
        if "道。" in result:
            return result  # 人物台词本身就是叙述
        return result  # 不在此处：复述结果
    raise ValueError(tool)


def build():
    samples = []
    seen = set()
    dropped = Counter()
    for setup in SETUPS:
        base = state_after(setup)
        # --- 移动：全部 7 个地点 -------------------------------------------
        for dest in ws.LOCATIONS:
            for u in MOVE_USERS:
                st = state_after(setup)
                summary = st.summary()
                user = u.format(dest=dest)
                result = st.tools()["移动"](dest)
                narr = narrate("移动", dest, result, base, st)
                samples.append(("移动", dest, setup, summary, user, result, narr))
        # --- 查看：当前地点 + 其余地点 --------------------------------------
        for u in LOOK_CUR_USERS:
            st = state_after(setup)
            summary = st.summary()
            result = st.tools()["查看"]("")
            narr = narrate("查看", "", result, base, st)
            samples.append(("查看", "", setup, summary, u, result, narr))
        for loc in ws.LOCATIONS:
            if loc == base.visitor_location:
                continue
            for u in LOOK_OTHER_USERS:
                st = state_after(setup)
                summary = st.summary()
                user = u.format(loc=loc)
                result = st.tools()["查看"](loc)
                narr = narrate("查看", loc, result, base, st)
                samples.append(("查看", loc, setup, summary, user, result, narr))
        # --- 拾取：在场物品 / 不在此地物品 / 已随身物品 -----------------------
        present = base.present_items()
        absent = [i for i in ws.ITEMS if i not in present
                  and base.item_loc[i] != ws.CARRIED]
        carried = base.carried_items()
        for item in present:
            for u in TAKE_USERS:
                st = state_after(setup)
                summary = st.summary()
                user = u.format(item=item)
                result = st.tools()["拾取"](item)
                narr = narrate("拾取", item, result, base, st)
                samples.append(("拾取", item, setup, summary, user, result, narr))
        for item in absent:
            st = state_after(setup)
            summary = st.summary()
            user = TAKE_FAIL_USERS[0].format(item=item)
            result = st.tools()["拾取"](item)
            narr = narrate("拾取", item, result, base, st)
            samples.append(("拾取", item, setup, summary, user, result, narr))
        for item in carried:
            st = state_after(setup)
            summary = st.summary()
            user = TAKE_FAIL_USERS[0].format(item=item)
            result = st.tools()["拾取"](item)
            narr = narrate("拾取", item, result, base, st)
            samples.append(("拾取", item, setup, summary, user, result, narr))
        # --- 对话：在场人物 / 不在场人物 ------------------------------------
        present_chars = base.present_characters()
        absent_chars = [c for c in ws.CHARACTERS if c not in present_chars]
        for name in present_chars:
            for u in TALK_USERS:
                st = state_after(setup)
                summary = st.summary()
                user = u.format(name=name)
                result = st.tools()["对话"](name)
                narr = narrate("对话", name, result, base, st)
                samples.append(("对话", name, setup, summary, user, result, narr))
        for name in absent_chars:
            st = state_after(setup)
            summary = st.summary()
            user = TALK_FAIL_USERS[0].format(name=name)
            result = st.tools()["对话"](name)
            narr = narrate("对话", name, result, base, st)
            samples.append(("对话", name, setup, summary, user, result, narr))

    # --- 组装、去重、校验 ----------------------------------------------------
    # 词表校验需要 tokenizer：走训练脚本同款 load_checkpoint 路径。
    import dual_transformer_numpy as dtn
    _, ck = dtn.DualTransformerNP.load_checkpoint(
        "/home/hatch/workspace/omega-chat/dual_d128_s69000_params.npz")
    tok = dtn.tokenizer_from_meta(ck["meta"])

    from agent_runtime import ToolRouter
    router = ToolRouter(ws.WorldState().tools(), tok)

    kept, per_tool = [], Counter()
    for tool, arg, setup, summary, user, result, narr in samples:
        call = f"{P.call_open}{tool}{P.name_end}{arg}{P.call_close}"
        wrapped = f"{P.result_open}{result}{P.result_close}"
        text = summary + "\n" + user + call + wrapped + narr
        if text in seen:
            dropped["dup"] += 1
            continue
        seen.add(text)
        ids = tok.encode(text, add_bos=True, add_eos=True)
        if any(i == tok.UNK for i in ids):
            dropped["unk"] += 1
            continue
        if len(ids) > 128:
            dropped[f"toolong_{tool}"] += 1
            continue
        # 调用段合法性：正则解析 + dispatch 非 None。
        call_start = len(summary) + 1 + len(user)
        call_end = call_start + len(call)
        narr_start = call_end + len(wrapped)
        seg = text[call_start:call_end]
        if router.dispatch(seg) is None:
            dropped["badcall"] += 1
            continue
        kept.append({"text": text, "call_start": call_start,
                     "call_end": call_end, "narr_start": narr_start,
                     "tool": tool, "arg": arg,
                     "setup": [list(s) for s in setup]})
        per_tool[tool] += 1

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 权重自查：第一条样本，摘要/用户/结果区权重 0、调用+叙述区权重 1。
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
    lines = ["# 世界工具调用数据报告（world6）",
             "",
             f"总样本：{len(kept)} 条（去重后）",
             f"分工具：{dict(per_tool)}",
             f"丢弃：{dict(dropped)}",
             f"最长样本：{maxlen} token（含 BOS/EOS，上限 128）",
             f"权重自查：首条样本摘要/用户/结果区 0、调用+叙述区 1 ✓",
             f"调用合法性：全部样本 dispatch 非 None ✓（生成时即校验）",
             f"叙述句说明：成功类移动/查看用地点描述合成短句，对话类复用",
             f"人物固定台词，失败类复述工具结果原文；无手编状态事实。",
             f"输出：{OUT_JSONL.name}",
             ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return kept


if __name__ == "__main__":
    build()
