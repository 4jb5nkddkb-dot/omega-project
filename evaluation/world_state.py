#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 【评估专用副本】本文件是 ~/workspace/omega-chat/world_state.py 的副本，
# 仅供 evaluation/ 下的探针（causal_probe / q_probe / loop）使用。
# 探针的打分语义（摘要格式、昼夜时钟、在场/随身真值）由这台状态机定义，
# 改格式会静默改变探针测量的东西，所以探针自带一份冻结副本，而不是
# import 产品代码。Omega 产品文件（world_*.py、governance、kernel、
# omega_v010 GUI）仍留在旧仓库，不进入本模型布局；如旧文件修了状态机，
# 须同步更新此副本并在 evaluation/README.md 记一笔。
"""最小可用世界状态机：西游记小世界的沙盒状态 + 工具 + 上下文携带 + 轻量校验。

纯标准库，不需要 NumPy。状态只存在内存对象里，不碰真实记忆、不碰文件、
不调用模型。对话回应是固定模板，不是模型生成。

设计（写清的三条规则）：
  1. 移动跟随规则：移动只改变游人的位置；师徒队伍（唐僧、孙悟空、猪八戒、
     沙僧、白龙马）中凡与游人在同一地点者，随游人一同到达新地点；不在同
     地点者不动；非队伍人物（观音）永远不动。
  2. 时间规则：每次有实际效果的世界工具（查看成功、实际换地点、拾取成功、
     对话成功）使时间前进一步；未知目标、无变化的移动不计步。昼夜是桩
     时钟：每四步换一次昼夜，不是仿真。
  3. 校验规则：只记录、不改写。叙述中出现与状态矛盾的“人物在某地”（人物
     的状态地点不是该地）或不在场人物发言（“某道”而某不在游人地点），记
     一条标记；模型输出原样返回。

上下文携带格式（world_prompt）：状态摘要在前、提示在后，换行分隔。
  世界状态。你在花果山。在场人物孙悟空、猪八戒。可见物品蟠桃。
  随身物品无。时间第三步夜晚。
  （摘要只用词表内字符：不用全角逗号、不用阿拉伯数字、不用全角冒号
  问号——dual 词表没有这些字符，编码时会静默丢弃。）

工具协议：与 agent_runtime.ToolRouter 一致，中文工具名，调用形式
《查看》】、《移动》花果山】、《拾取》蟠桃】、《对话》孙悟空】。
工具输出只用词表内字符，编码后可无损还原（见 world_state_test）。

治理说明：世界工具只改 WorldState 对象内的沙盒状态。默认世界循环跑在
governance=None 下。治理开启时，调用方须在策略里显式列入查看、移动、
拾取、对话（例如低风险、无需确认），否则 ToolGate 按白名单外拒绝；
不要改 omega_governance.py 的默认策略。
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# 世界内容：地点、人物、物品（最小集，覆盖一致性探测的 8 例锚点）
# ----------------------------------------------------------------------------

LOCATIONS = {
    "花果山": "猴王故地。水帘洞天。",
    "水帘洞": "花果山中洞府。瀑布遮门。",
    "高老庄": "庄户人家。八戒旧居。",
    "流沙河": "河水汹涌。沙僧旧地。",
    "取经路": "西行大道。师徒同行。",
    "南海": "观音道场。普陀落伽。",
    "天宫": "玉皇居所。云雾缭绕。",
}

# 队伍：移动时跟随游人。观音不在队伍里，永远不动。
PARTY = ("唐僧", "孙悟空", "猪八戒", "沙僧", "白龙马")

CHARACTERS = {
    "孙悟空": {"party": True, "line": "俺老孙在此。有何见教。"},
    "唐僧": {"party": True, "line": "善哉。前路还远。莫要停步。"},
    "猪八戒": {"party": True, "line": "俺老猪饿了。先找些吃的。"},
    "沙僧": {"party": True, "line": "师父放心。行李有我看着。"},
    "白龙马": {"party": True, "line": "驮你前行。不敢停步。"},
    "观音": {"party": False, "line": "阿弥陀佛。善哉善哉。"},
}

# 物品名 -> 初始地点（“随身”表示已被游人携带）。
ITEMS = {
    "蟠桃": "花果山",
    "金箍棒": "花果山",
    "九齿钉耙": "高老庄",
    "月牙铲": "流沙河",
    "锦襕袈裟": "取经路",
    "紫金钵盂": "南海",
}

CARRIED = "随身"

_DIGITS = "零一二三四五六七八九"


def int_to_zh(n: int, _leading: bool = True) -> str:
    """整数 -> 中文数字（0 到 9999 以上用万进位），摘要里不用阿拉伯数字。

    _leading 只管“十”要不要补“一”：顶层 10 读“十”，但百、千、万、
    零后面的十读“一十”（一百一十、一千零一十），这是标准读法。
    """
    if n < 0:
        raise ValueError("time step must be non-negative")
    if n < 10:
        return _DIGITS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if tens == 1:
            head = "十" if _leading else "一十"
        else:
            head = _DIGITS[tens] + "十"
        return head + (_DIGITS[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = _DIGITS[hundreds] + "百"
        if rest == 0:
            return head
        if rest < 10:
            return head + "零" + _DIGITS[rest]
        return head + int_to_zh(rest, _leading=False)
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        head = _DIGITS[thousands] + "千"
        if rest == 0:
            return head
        if rest < 100:
            return head + "零" + int_to_zh(rest, _leading=False)
        return head + int_to_zh(rest, _leading=False)
    high, rest = divmod(n, 10000)
    head = int_to_zh(high) + "万"
    if rest == 0:
        return head
    if rest < 1000:
        return head + "零" + int_to_zh(rest, _leading=False)
    return head + int_to_zh(rest, _leading=False)


def _join(names) -> str:
    return "、".join(names) if names else "无"


class WorldState:
    """西游记小世界的最小状态：地点、人物位置、物品位置、时间步。"""

    def __init__(self, visitor_location: str = "取经路"):
        if visitor_location not in LOCATIONS:
            raise ValueError(f"unknown location {visitor_location!r}")
        self.visitor_location = visitor_location
        # 初始：师徒队伍与游人同在取经路，观音在南海。
        self.char_loc = {name: ("取经路" if info["party"] else "南海")
                         for name, info in CHARACTERS.items()}
        self.item_loc = dict(ITEMS)
        self.time_step = 0
        self.trace = []  # (工具， 参数， 结果， 执行后时间步），供探测报告用。

    # -- 时间（桩时钟：每四步换昼夜）-------------------------------------------
    def day_night(self) -> str:
        return "白天" if (self.time_step // 4) % 2 == 0 else "夜晚"

    def _tick(self) -> None:
        self.time_step += 1

    def _record(self, tool: str, arg: str, result: str) -> str:
        self.trace.append((tool, arg, result, self.time_step))
        return result

    # -- 查询 -----------------------------------------------------------------
    def present_characters(self, location: str = None):
        loc = self.visitor_location if location is None else location
        return [name for name in CHARACTERS if self.char_loc[name] == loc]

    def present_items(self, location: str = None):
        loc = self.visitor_location if location is None else location
        return [name for name in ITEMS if self.item_loc[name] == loc]

    def carried_items(self):
        return [name for name in ITEMS if self.item_loc[name] == CARRIED]

    def summary(self) -> str:
        """状态摘要：上下文携带的固定格式（只用词表内字符）。
        地点压后（v9 修法 A）：让“你在X。”紧贴后面的问题，问答距离
        从约 53 字缩到约 9 字，64 字窗口能同时盖住线索与答案。"""
        return (f"世界状态。在场人物{_join(self.present_characters())}。"
                f"可见物品{_join(self.present_items())}。"
                f"随身物品{_join(self.carried_items())}。"
                f"时间第{int_to_zh(self.time_step)}步{self.day_night()}。"
                f"你在{self.visitor_location}。")

    # -- 工具（ToolRouter 可直接挂的函数，参数为字符串）-------------------------
    def look(self, arg: str) -> str:
        loc = arg.strip() or self.visitor_location
        if loc not in LOCATIONS:
            return self._record("查看", arg, f"没有{arg.strip()}这个地方。")
        prefix = "你在" if loc == self.visitor_location else ""
        self._tick()
        return self._record(
            "查看", arg,
            f"{prefix}{loc}。{LOCATIONS[loc]}"
            f"在场人物{_join(self.present_characters(loc))}。"
            f"可见物品{_join(self.present_items(loc))}。")

    def move(self, arg: str) -> str:
        dest = arg.strip()
        if dest not in LOCATIONS:
            return self._record("移动", arg, f"没有{dest}这个地方。")
        if dest == self.visitor_location:
            return self._record("移动", arg, f"你已经在{dest}了。")
        old = self.visitor_location
        followers = [name for name in PARTY if self.char_loc[name] == old]
        self.visitor_location = dest
        for name in followers:
            self.char_loc[name] = dest
        self._tick()
        tail = f"{_join(followers)}随你同行。" if followers else ""
        return self._record("移动", arg, f"你来到{dest}。{tail}")

    def take(self, arg: str) -> str:
        item = arg.strip()
        if item not in ITEMS:
            return self._record("拾取", arg, f"没有{item}这样东西。")
        if self.item_loc[item] == CARRIED:
            return self._record("拾取", arg, f"你已经带着{item}了。")
        if self.item_loc[item] != self.visitor_location:
            return self._record("拾取", arg, f"此地没有{item}。")
        self.item_loc[item] = CARRIED
        self._tick()
        return self._record("拾取", arg, f"你拾起了{item}。")

    def talk(self, arg: str) -> str:
        name = arg.strip()
        if name not in CHARACTERS:
            return self._record("对话", arg, f"没有{name}这个人物。")
        if self.char_loc[name] != self.visitor_location:
            return self._record("对话", arg, f"{name}不在此处。")
        self._tick()
        return self._record("对话", arg, f"{name}道。{CHARACTERS[name]['line']}")

    def tools(self) -> dict:
        """ToolRouter 可直接挂的工具表（中文工具名）。"""
        return {"查看": self.look, "移动": self.move,
                "拾取": self.take, "对话": self.talk}

    # -- 轻量校验（只记录，不改写输出）------------------------------------------
    def validate(self, text: str):
        """叙述与状态矛盾时返回标记列表；无矛盾返回空列表。

        两条字符串规则（近似，只抓显式矛盾）：
          矛盾地点：叙述写“某在某地”，而某的状态地点不是该地。
          不在场发言：叙述写“某道”，而某不在游人地点。
        """
        flags = []
        for name in CHARACTERS:
            loc = self.char_loc[name]
            for other in LOCATIONS:
                if other != loc and f"{name}在{other}" in text:
                    flags.append(f"人物地点矛盾：{name}在{other}，状态为{loc}")
            if (self.char_loc[name] != self.visitor_location
                    and f"{name}道" in text):
                flags.append(f"不在场人物发言：{name}道，"
                             f"{name}在{self.char_loc[name]}不在{self.visitor_location}")
        return flags


def world_prompt(state: WorldState, prompt: str) -> str:
    """状态摘要在前、提示在后（换行分隔），即每回合给模型的上下文。"""
    return state.summary() + "\n" + prompt
