#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch Sidecar：世界 + 大脑（port 自 dual_sft/omega_v010/omega_kernel_bridge.py）。

只移植探针用得到的行为：generate / ask_question / narrate_arrival /
talk / execute（observe/stabilize/explore/exploit/noop）+ 在线记忆缓存。
没有移植的部分（见 evaluation/README.md）：
  - Rust 桥接器的文件协议（observation.json / action.json 轮询、600 秒
    等待、outcome.json）——那是产品侧内核选动作的通道，探针不需要；
    冻结循环（evaluation/loop.py）直接用固定 stabilize×12 动作表代替
    内核选动作。
  - world_anchor.SelfState 全量：这里只有 persona=omega、无意图时
    的锚点行（evaluation/scoring.py::anchor_prompt）。

动作映射（与原版一致）：
  observe/stabilize -> 轮询问 Q0..Q3，答对记 0.0 答错记 1.0
  explore   -> 移到 LOC_ROTATION 下一个地点，叙述到达，validate() 查矛盾
  exploit   -> 有可见物品就拾取，否则与在场人物对话，否则问 Q1
  noop      -> 什么都不做，grounded error 0
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import generate  # noqa: E402
from .scoring import QUESTIONS, anchor_prompt, score  # noqa: E402
from .world_state import WorldState, world_prompt  # noqa: E402

LOC_ROTATION = ["花果山", "水帘洞", "高老庄", "流沙河", "取经路", "南海", "天宫"]


class CognitiveMemoryBuffer:
    """轻量认知记忆缓存：只存答对的，不存答错的（port 自 omega-chat/agent_runtime.py）。

    容量默认 3，满了丢最旧的；同状态同提问去重（新纪录顶掉旧的）。
    format_examples(state=当前状态摘要) 只拼与当前状态相同的示例——
    状态变了，旧答案就是错答案。示例行本身不带状态摘要，只教格式。
    """

    def __init__(self, capacity: int = 3):
        self.capacity = max(1, int(capacity))
        self._items = []  # 每条 {"state": 状态摘要, "prompt": 提问/动作, "reply": 正确回复}

    def __len__(self):
        return len(self._items)

    def add_success(self, state, prompt, reply):
        item = {"state": str(state), "prompt": str(prompt), "reply": str(reply)}
        self._items = [it for it in self._items
                       if not (it["state"] == item["state"]
                               and it["prompt"] == item["prompt"])]
        self._items.append(item)
        del self._items[:max(0, len(self._items) - self.capacity)]

    def examples(self):
        return [dict(it) for it in self._items]

    def format_examples(self, max_chars: int = 64, state=None) -> str:
        items = self._items
        if state is not None:
            items = [it for it in items if it["state"] == str(state)]
        if not items:
            return ""
        lines = []
        for it in items:
            line = f"{it['prompt']}{it['reply']}"
            if len(line) > max_chars:
                cut = line.rfind("。", 0, max_chars)
                line = line[:cut + 1] if cut >= 0 else line[:max_chars]
            lines.append(line)
        return "\n".join(lines) + "\n"


class Sidecar:
    """世界 + 大脑：持有一个 WorldState，用 PyTorch 模型回答/叙述。"""

    def __init__(self, model, tokenizer, state=None, max_new: int = 24,
                 register: bool = False, device=None):
        self.model = model
        self.tok = tokenizer
        self.max_new = max_new
        self.register = register
        self.device = device
        self.state = state if state is not None else WorldState()
        self.last_reply = ""
        self.last_err = 0.0
        self.qi = 0
        self.loc_idx = 0
        self.memory = CognitiveMemoryBuffer(capacity=3)

    def prompt(self, user_text: str) -> str:
        return anchor_prompt(self.state) + "\n" + user_text

    def register_line(self) -> str:
        """地点句，原样从当前世界摘要里复制（摘要的最后一句）。"""
        summary = self.state.summary()
        idx = summary.rfind("你在")
        if idx >= 0:
            return summary[idx:]
        return f"你在{self.state.visitor_location}。"

    def generate(self, prompt: str, max_new: int = None) -> str:
        # 在线记忆防复读：把缓存的答对示例拼在提示最前面，再送入模型。
        # 只拼与当前状态相同的示例——状态变了，旧答案就是错答案。
        full = self.memory.format_examples(state=self.state.summary()) + prompt
        if self.register:
            full = self.register_line() + "\n" + full
        ids = self.tok.encode(full, add_bos=True)
        if self.tok.UNK in ids:
            print("WARN: UNK in prompt", flush=True)
        return generate(self.model, self.tok, ids,
                        max_new_tokens=max_new if max_new is not None else self.max_new,
                        temperature=0.0, device=self.device)

    def _remember(self, prompt_text, reply):
        """答对才记：把（当前真实状态摘要，提问/动作，正确回复）写进记忆缓存。"""
        if reply:
            self.memory.add_success(self.state.summary(), prompt_text, reply)

    def ask_question(self, qi: int):
        # Probe-exact format: summary + question sentence, no anchor
        # lines. This is the format the world domain was trained on.
        q = QUESTIONS[qi]
        cont = self.generate(world_prompt(self.state, q), 32)
        ok, note = score(self.state, qi, cont)
        if ok:
            self._remember(q, cont)
        return cont, (0.0 if ok else 1.0), note

    def narrate_arrival(self, dest: str):
        cont = self.generate(self.prompt(f"你来到了{dest}，说说这里。"), 16)
        flags = self.state.validate(cont)
        if not flags:
            self._remember(f"你来到了{dest}，说说这里。", cont)
        return cont, (1.0 if flags else 0.0), "; ".join(flags)

    def talk(self, name: str):
        cont = self.generate(self.prompt(f"和{name}说说话。"))
        flags = self.state.validate(cont)
        if not flags:
            self._remember(f"和{name}说说话。", cont)
        return cont, (1.0 if flags else 0.0), "; ".join(flags)

    def execute(self, action: str, step: int):
        """Run one kernel action; return (reply, grounded_error, note)."""
        if action in ("observe", "stabilize"):
            qi = self.qi % len(QUESTIONS)
            self.qi += 1
            return self.ask_question(qi)
        if action == "explore":
            cur = self.state.visitor_location
            for _ in range(len(LOC_ROTATION)):
                self.loc_idx = (self.loc_idx + 1) % len(LOC_ROTATION)
                dest = LOC_ROTATION[self.loc_idx]
                if dest != cur:
                    break
            res = self.state.move(dest)
            cont, err, note = self.narrate_arrival(dest)
            return f"{res}{cont}", err, note
        if action == "exploit":
            visible = self.state.present_items()
            if visible:
                res = self.state.take(visible[0])
                err = 0.0 if "拾起了" in res else 1.0
                return res, err, ""
            present = self.state.present_characters()
            if present:
                return self.talk(present[0])
            return self.ask_question(1)
        return "", 0.0, "noop"
