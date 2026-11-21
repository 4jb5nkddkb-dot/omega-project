#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q0/Q1/Q2/Q3 探针集（PyTorch 版）：模型读不读状态摘要。

Port of dual_sft/diag_q0_s600.py。用真实 WorldState 构造每个地点的
状态摘要（与 loop/causal probe 完全一致的格式），问：
  - Q0：7 个地点 × 两种格式（probe 格式：摘要 + 问题；register 格式：
    地点句放最前面再跟摘要 + 问题）；
  - Q1/Q2/Q3：只在取经路（step 2，白天）做对照。
每题用 evaluation/scoring.py::score() 判 PASS/FAIL 并汇总。
Q0 固定答案（三次都答“你在花果山”）与 Q2/Q3 跷跷板就是这套探针
先抓出来的（见 evaluation/README.md）。

用法：
  python -m evaluation.q_probe --checkpoint checkpoints/generalist-v5_s600.pt
  python -m evaluation.q_probe --checkpoint <ckpt.pt> --log diag_q0.log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation._common import load_model_and_tokenizer  # noqa: E402
from evaluation.scoring import LOCATIONS_LIST, QUESTIONS, score  # noqa: E402
from evaluation.world_state import WorldState, world_prompt  # noqa: E402
from inference.generate import generate  # noqa: E402


def make_state(loc: str, step: int = 2) -> WorldState:
    """用工具真实走出来的状态（不用手工拼状态，保证可达）。"""
    st = WorldState()
    if loc != st.visitor_location:
        st.tools()["移动"](loc)
    st.time_step = step
    return st


def main():
    ap = argparse.ArgumentParser(description="Q0/Q1/Q2/Q3 探针集（probe + register 格式）")
    ap.add_argument("--checkpoint", required=True, help="已转换的 .pt 检查点路径")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new", type=int, default=16,
                    help="每题生成的新 token 数（原版 16）")
    ap.add_argument("--log", default="diag_q0.log", help="写出日志的文件路径")
    args = ap.parse_args()

    try:
        model, tok, step, config, extra = load_model_and_tokenizer(
            args.checkpoint, args.device)
    except Exception as e:
        # 无检查点 / 检查点损坏 / model 包未就绪：报清楚的错，不抛堆栈。
        print(f"ERROR: 无法加载检查点：{e}", file=sys.stderr)
        sys.exit(2)
    print(f"model loaded: {args.checkpoint} step={step} "
          f"merge={config.get('merge_method')} d={config.get('d_model')}",
          flush=True)

    log = open(args.log, "w", encoding="utf-8")
    results = []  # (tag, qi, ok, note, reply)

    def run(tag, state, qi, prompt):
        ids = tok.encode(prompt, add_bos=True)
        if tok.UNK in ids:
            print(f"WARN: UNK in prompt {tag}", flush=True)
        reply = generate(model, tok, ids, max_new_tokens=args.max_new,
                         temperature=0.0, device=args.device)
        ok, note = score(state, qi, reply)
        results.append((tag, qi, ok, note, reply))
        print(f"--- {tag} {'PASS' if ok else 'FAIL'} {note}", flush=True)
        print(f"PROMPT: {prompt!r}", flush=True)
        print(f"REPLY : {reply!r}", flush=True)
        log.write(f"--- {tag} {'PASS' if ok else 'FAIL'} {note}\n"
                  f"PROMPT: {prompt!r}\nREPLY : {reply!r}\n")
        log.flush()
        return reply

    for loc in LOCATIONS_LIST:
        st = make_state(loc)
        summ = st.summary()
        # Q0 probe 格式（无寄存器、无 anchor）
        run(f"Q0 probe {loc}", st, 0, world_prompt(st, QUESTIONS[0]))
        # Q0 寄存器格式（地点句放最前面）
        reg = summ[summ.rfind("你在"):]
        run(f"Q0 register {loc}", st, 0,
            reg + "\n" + summ + "\n" + QUESTIONS[0])

    # Q1/Q2/Q3 对照（只在取经路，step 2 白天）
    st = make_state("取经路")
    summ = st.summary()
    for qi in (1, 2, 3):
        run(f"Q{qi} probe {st.visitor_location}", st, qi,
            world_prompt(st, QUESTIONS[qi]))

    log.close()
    total = sum(1 for r in results if r[2])
    print(f"TOTAL: {total}/{len(results)}", flush=True)
    for qi in range(4):
        sub = [r for r in results if r[1] == qi]
        print(f"  Q{qi} {QUESTIONS[qi]} "
              f"{sum(1 for r in sub if r[2])}/{len(sub)}", flush=True)
    for fmt in ("probe", "register"):
        sub = [r for r in results if fmt in r[0] and r[1] == 0]
        print(f"  Q0 {fmt}: {sum(1 for r in sub if r[2])}/{len(sub)}",
              flush=True)
    print(f"done -> {args.log}", flush=True)


if __name__ == "__main__":
    main()
