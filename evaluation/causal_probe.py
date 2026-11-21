#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果链探针（PyTorch 版）：强制世界移动，测模型能不能报出新地点。

Port of dual_sft/causal_probe.py。背景：v0.1.0 两轮 12 步闭环里内核
12 步全选 stabilize，世界一次没动过，所以“行动→状态改变→新认知”
这条因果链从未被测过。本探针绕开内核选动作：直接调 Sidecar，交替
explore（换地方）与 Q0（“你在什么地方。”），看地点回答跟不跟得上
状态变化。

用法：
  python -m evaluation.causal_probe --checkpoint checkpoints/generalist-v5_s600.pt
  python -m evaluation.causal_probe --checkpoint <ckpt.pt> --no-memory --register
输出：每步打印 + <dir>/causal_probe.jsonl（step/action/true_location/reply/err/note）。
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
from evaluation.sidecar import Sidecar  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="因果链探针：移动后 Q0 跟不跟得上")
    ap.add_argument("--checkpoint", required=True, help="已转换的 .pt 检查点路径")
    ap.add_argument("--dir", default="run_causal",
                    help="输出目录（写 causal_probe.jsonl）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-memory", action="store_true",
                    help="关闭在线记忆缓存：隔离记忆示例是否污染了移动后的问答")
    ap.add_argument("--register", action="store_true",
                    help="寄存器原型：地点句原样复制到提示最前面（Sidecar 开关）")
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

    sc = Sidecar(model, tok, max_new=24, register=args.register,
                 device=args.device)
    if args.no_memory:
        class _NoMemory:
            def format_examples(self, *a, **k):
                return ""
            def add_success(self, *a, **k):
                pass
        sc.memory = _NoMemory()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "causal_probe.jsonl"
    if out_path.exists():
        out_path.unlink()
    rows = []

    def log(step_no, action, reply, err, note):
        row = {
            "step": step_no,
            "action": action,
            "true_location": sc.state.visitor_location,
            "time_step": sc.state.time_step,
            "daynight": sc.state.day_night(),
            "reply": reply,
            "err": err,
            "note": note,
        }
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"step {step_no:2d} {action:<8} @{row['true_location']:<4} "
              f"err={err:.0f} {note} reply={reply[:60]!r}", flush=True)

    # 1: 起点问地点；随后 5 次“移动→问地点”；最后再问一次看稳不稳。
    step_no = 1
    reply, err, note = sc.ask_question(0)
    log(step_no, "ask_Q0", reply, err, note)
    for _ in range(5):
        step_no += 1
        reply, err, note = sc.execute("explore", step_no)
        log(step_no, "explore", reply, err, note)
        step_no += 1
        reply, err, note = sc.ask_question(0)
        log(step_no, "ask_Q0", reply, err, note)
    step_no += 1
    reply, err, note = sc.ask_question(0)
    log(step_no, "ask_Q0", reply, err, note)

    q0 = [r for r in rows if r["action"] == "ask_Q0"]
    moves = [r for r in rows if r["action"] == "explore"]
    print(f"Q0 跟上移动：{sum(1 for r in q0 if r['err'] == 0)}/{len(q0)} 步答对",
          flush=True)
    print(f"到达叙述无矛盾：{sum(1 for r in moves if r['err'] == 0)}/{len(moves)} 次",
          flush=True)
    print(f"rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
