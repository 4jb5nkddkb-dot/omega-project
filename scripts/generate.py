#!/usr/bin/env python3
"""生成入口：检查点 + 提示 -> 续写文本。

薄 CLI：只做加载与输出，采样逻辑在 inference/generate.py
（Agent C 已落地：generate / generate_with_gate / DecodeController，
签名见 inference/generate.py）。

注意：generate() 只返回**新增 token 解码出的续写**（不回显提示），
generate_with_gate() 额外返回提示的门值 g（每序列左支占比）。
模型加载依赖 model/checkpoint.py + model/transformer.py（Agent A，
写本文时尚未落地：model/ 只有 norm/attention/block/feedforward，
无 transformer.py / checkpoint.py / __init__.py），故本脚本目前
import 即失败——待 Agent A 落地后直接可用，不用改本文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description="PyTorch 生成入口（generalist）")
    ap.add_argument("--checkpoint", required=True, help=".pt 检查点路径")
    ap.add_argument("--prompt", default="",
                    help="提示文本（空则从 BOS 开始自由生成）")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = 贪心")
    ap.add_argument("--top-k", type=int, default=0, help="0 = 不用 top-k")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--device", default=None,
                    help="cpu/cuda；缺省用模型所在设备")
    ap.add_argument("--show-gate", action="store_true",
                    help="门控模型时同时打印提示的门值 g（generate_with_gate）")
    ap.add_argument("--vocab", default="data/processed/generalist_vocab.json")
    args = ap.parse_args(argv)

    from model.checkpoint import load_checkpoint  # Agent A，未落地
    from model.transformer import DualTransformer  # Agent A，未落地
    from tokenizer import CharTokenizer  # 已就绪

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    tok = CharTokenizer(vocab["chars"])
    ckpt = load_checkpoint(args.checkpoint)
    model = DualTransformer(**{k: v for k, v in ckpt["config"].items()
                               if k in ("d_model", "n_blocks", "n_heads",
                                        "n_kv_heads", "merge_method",
                                        "vocab_size", "max_seq",
                                        "tie_embeddings", "use_rms",
                                        "pre_norm", "pos_mode", "ff_mode",
                                        "init_style", "n_stem_blocks",
                                        "final_norm")})
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.show_gate:
        from inference.generate import generate_with_gate  # Agent C，已落地
        text, gate = generate_with_gate(
            model, tok, args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            device=args.device)
        print(text)
        print(f"[门值 g（左支占比）：{gate:.4f}]")
    else:
        from inference.generate import generate  # Agent C，已落地
        text = generate(model, tok, args.prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, device=args.device)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
