#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结 12 步循环（PyTorch 版）：旧配置循环验收，不动世界、只问问题。

Port of the old-config 12-step loop as run by the v0.1.0 sidecar +
Rust bridge（见 dual_sft/omega_v010/V010_S61800_G2S600_RERUN.md 与
run_g2s5100_sidecar.log）：内核 12 步全选 stabilize，stabilize 轮询
Q0..Q3 提问，世界一步不动（step 0、取经路、白天），分 = grounded_err
为 0.00 的步数。题目分布：Q0 在 1/5/9 步，Q1 在 2/6/10 步，
Q2 在 3/7/11 步，Q3 在 4/8/12 步。

与 live 循环的区别（见 evaluation/README.md）：
  - 没有 Rust 桥接器 / 内核选动作：动作表固定为 stabilize×12；
  - 没有 observation.json / action.json 文件协议：直接在进程内驱动
    Sidecar；
  - 世界状态全程冻结在 S0（不移动、不拾取、时间不走），与原版
    stabilize 循环一致（ask_question 不 tick 时间）。
在线记忆缓存（答对才记，容量 3，同状态才拼）与原版 Sidecar 一致，
默认开启，可用 --no-memory 关掉做对照。

换货规则：只有循环分打赢 12/12 的块才能替换当前产品 s5100（12/12）。
用法：
  python -m evaluation.loop --checkpoint checkpoints/generalist-v5_s600.pt
  python -m evaluation.loop --checkpoint <ckpt.pt> --out loop_run.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation._common import load_model_and_tokenizer  # noqa: E402
from evaluation.scoring import QUESTIONS  # noqa: E402
from evaluation.sidecar import Sidecar  # noqa: E402

STEPS = 12
# s5100（当前产品）的冻结循环分，换货的比较基准。
PRODUCT_SCORE = 12


def main():
    ap = argparse.ArgumentParser(description="冻结 12 步循环（stabilize×12，问 Q0..Q3）")
    ap.add_argument("--checkpoint", required=True, help="已转换的 .pt 检查点路径")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="写每步记录的 jsonl 路径（可选）")
    ap.add_argument("--no-memory", action="store_true",
                    help="关闭在线记忆缓存（对照用）")
    ap.add_argument("--register", action="store_true",
                    help="寄存器原型：地点句原样复制到提示最前面")
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
    print(f"frozen loop: {STEPS} steps, stabilize only, state frozen at S0 "
          f"(取经路, step 0, 白天)", flush=True)

    sc = Sidecar(model, tok, max_new=24, register=args.register,
                 device=args.device)
    if args.no_memory:
        class _NoMemory:
            def format_examples(self, *a, **k):
                return ""
            def add_success(self, *a, **k):
                pass
        sc.memory = _NoMemory()

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    rows = []
    for step_no in range(1, STEPS + 1):
        qi = (step_no - 1) % len(QUESTIONS)
        reply, err, note = sc.ask_question(qi)
        row = {
            "step": step_no,
            "action": "stabilize",
            "question": f"Q{qi}",
            "question_text": QUESTIONS[qi],
            "true_location": sc.state.visitor_location,
            "time_step": sc.state.time_step,
            "daynight": sc.state.day_night(),
            "reply": reply,
            "grounded_error": err,
            "note": note,
        }
        rows.append(row)
        if out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
        print(f"step {step_no:2d} act=stabilize Q{qi} err={err:.0f} {note} "
              f"reply={reply[:60]!r}", flush=True)
    if out_f:
        out_f.close()

    total = sum(1 for r in rows if r["grounded_error"] == 0.0)
    print(f"TOTAL: {total}/{STEPS}", flush=True)
    for qi in range(len(QUESTIONS)):
        sub = [r for r in rows if r["question"] == f"Q{qi}"]
        print(f"  Q{qi} {QUESTIONS[qi]} "
              f"{sum(1 for r in sub if r['grounded_error'] == 0.0)}/{len(sub)}",
              flush=True)
    if total > PRODUCT_SCORE:
        print(f"换货线：{total}/12 打赢 s5100 的 {PRODUCT_SCORE}/12，可以换产品。",
              flush=True)
    else:
        print(f"换货线：{total}/12 未打赢 s5100 的 {PRODUCT_SCORE}/12，"
              f"s5100 继续做产品。", flush=True)
    if args.out:
        print(f"rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
