#!/usr/bin/env python3
"""训练入口：configs/*.yaml + 命令行 -> training.trainer.train。

薄 CLI：参数合并、建模型、建分层采样 batch_fn，再调
training.trainer.train(model, batch_fn=..., ...)（Agent B，
签名见 training/trainer.py 头注释）。训练循环、断点、优化器
都在 trainer 里，不在这里。

NumPy 训练器（generalist_pilot_train.py）的调用约定尽量保留：
--config / --steps / --resume / --init / --out / --batch / --seq /
--lr / --save-every / --device / --amp。

续跑硬规则（沿用 NumPy 时代，见 configs/train_default.yaml）：
- --resume 接 _last.pt 时**省略 --steps**，由 trainer 按
  schedule_total_steps - 已跑步数自动算剩余步数；显式传 --steps
  会跑满给定步数（不按 schedule_total 封顶）。
- --resume 时 trainer 会校验模型形状与断点 config 一致：
  d1024 续跑必须显式传全套形状参数
  （--d_model 1024 --n_blocks 4 --n_heads 16 --n_kv_heads 8），
  否则死在 config-mismatch 上（与 NumPy 时代同一教训）。

配置是扁平 key: value 的 YAML（见 configs/*.yaml 头注释），这里先用
PyYAML 读，缺 PyYAML 时退回内置简易读取器（只支持扁平标量）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_KWARGS = ("d_model", "n_blocks", "n_heads", "n_kv_heads", "merge_method",
                "vocab_size", "max_seq", "tie_embeddings", "dropout",
                "use_rms", "pre_norm", "pos_mode", "ff_mode", "init_style",
                "n_stem_blocks", "final_norm")


def _flat_parse(text):
    """极简扁平 YAML 读取器：只支持 `key: value` 行，忽略注释与空行。"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip().strip('"').strip("'")
        if val in ("true", "True"):
            val = True
        elif val in ("false", "False"):
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        out[key.strip()] = val
    return out


def load_config(path):
    """读一份扁平 YAML 配置，返回 dict。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        data = yaml.safe_load(text)
        return dict(data or {})
    except ImportError:
        return _flat_parse(text)


def merge_configs(*paths):
    """多个配置按顺序合并（后者覆盖前者）。"""
    merged = {}
    for p in paths:
        if p:
            merged.update(load_config(p))
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description="PyTorch 训练入口（generalist）")
    ap.add_argument("--config", default="configs/train_default.yaml",
                    help="训练配置（可叠加：--model-config 另指模型配置）")
    ap.add_argument("--model-config", default=None,
                    help="模型配置，如 configs/model_gated.yaml")
    ap.add_argument("--steps", type=int, default=None,
                    help="本块步数；--resume 时省略，由 trainer 自动算剩余步数")
    ap.add_argument("--resume", default=None,
                    help="续跑的检查点（_last.pt；形状参数须显式传全）")
    ap.add_argument("--init", default=None,
                    help="从已有 .pt 检查点初始化权重（非续跑：不恢复优化器/步数）")
    ap.add_argument("--out", default=None,
                    help="检查点输出路径，如 checkpoints/generalist3_d1024_s<N>.pt"
                         "（<N> 由 model/checkpoint.save_checkpoint 替换为步数）")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_blocks", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="cpu/cuda；缺省由 trainer 自动选（有 CUDA 用 CUDA）")
    ap.add_argument("--amp", action="store_true",
                    help="CUDA 上开 AMP（CPU 下无效）")
    args = ap.parse_args(argv)

    train_cfg = load_config(args.config)
    model_cfg_path = args.model_config or train_cfg.get("model_config")
    cfg = merge_configs(model_cfg_path, args.config)
    # 命令行覆盖配置（只覆盖给出的项）
    for flag, key in (("batch", "batch_size"), ("seq", "seq_len"),
                      ("lr", "lr"), ("save_every", "save_every"),
                      ("d_model", "d_model"), ("n_blocks", "n_blocks"),
                      ("n_heads", "n_heads"), ("n_kv_heads", "n_kv_heads"),
                      ("seed", "seed")):
        if getattr(args, flag) is not None:
            cfg[key] = getattr(args, flag)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.resume and args.steps is not None:
        print("警告：--resume 时显式传了 --steps，trainer 会跑满给定步数，"
              "可能超过 schedule_total（见本文件头注释的续跑规则）。",
              file=sys.stderr)

    out = args.out
    if out is None:
        pattern = str(cfg.get("checkpoint_pattern",
                              "generalist3_d1024_s<N>.pt"))
        if pattern.endswith(".npz"):  # 旧 NumPy 配置里的模式，换后缀
            pattern = pattern[:-4] + ".pt"
        out = str(Path(cfg.get("checkpoint_dir", "checkpoints")) / pattern)

    print(f"训练配置：recipe={cfg.get('recipe')} "
          f"d={cfg.get('d_model')} blocks={cfg.get('n_blocks')} "
          f"heads={cfg.get('n_heads')} kv={cfg.get('n_kv_heads')} "
          f"merge={cfg.get('merge_method')} steps={cfg.get('steps')} "
          f"batch={cfg.get('batch_size')} seq={cfg.get('seq_len')} "
          f"device={args.device or 'auto'} amp={args.amp} "
          f"resume={args.resume or '-'} out={out}")

    # 模型（Agent A：model.transformer.DualTransformer，CONTRACT.md API）
    from model.transformer import DualTransformer
    model_kwargs = {k: cfg[k] for k in MODEL_KWARGS if k in cfg}
    model = DualTransformer(**model_kwargs)

    # 分词器 + 分层采样 batch_fn（training/dataset.py，Agent B 已移植）
    from tokenizer import CharTokenizer
    from training.dataset import domain_texts, make_batch_fn
    vocab = json.loads((PROJECT_ROOT / "data" / "processed"
                        / "generalist_vocab.json").read_text(encoding="utf-8"))
    tok = CharTokenizer(vocab["chars"])
    ids, answer_masks = domain_texts(tok, return_answer_masks=True)  # 默认读新布局 data/；过渡期可改传 live 路径
    batch_fn = make_batch_fn(ids, seq_len=int(cfg.get("seq_len", 128)),
                             batch_size=int(cfg.get("batch_size", 8)),
                             seed=int(cfg.get("seed", 7)),
                             answer_masks=answer_masks)

    extra_meta = {
        "recipe": cfg.get("recipe"),
        "ratios": {name: w for name, w in
                   (("经典", cfg.get("ratio_classics")),
                    ("代码", cfg.get("ratio_code")),
                    ("对话", cfg.get("ratio_dialogue")),
                    ("常识", cfg.get("ratio_knowledge")),
                    ("推理", cfg.get("ratio_reasoning")),
                    ("世界", cfg.get("ratio_world"))) if w is not None},
        "seq_len": cfg.get("seq_len"),
        "merge": cfg.get("merge_method"),
        "tied": cfg.get("tie_embeddings"),
        "dropout": cfg.get("dropout"),
        "label_smoothing": cfg.get("label_smoothing"),
        "gate_loss_weight": cfg.get("gate_loss_weight"),
        "gate": "left=经典/代码/对话/常识/推理, right=世界",
        "tokenizer": tok.to_meta(),
    }

    from training.trainer import train  # Agent B
    result = train(
        model, batch_fn=batch_fn,
        steps=cfg.get("steps"),
        total_steps=None,
        batch_size=int(cfg.get("batch_size", 8)),
        seq_len=int(cfg.get("seq_len", 128)),
        lr=float(cfg.get("lr", 1e-3)),
        schedule=str(cfg.get("schedule", "cosine")),
        warmup_steps=int(cfg.get("warmup_steps", 10)),
        min_lr_ratio=float(cfg.get("min_lr_ratio", 0.1)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        label_smoothing=float(cfg.get("label_smoothing", 0.1)),
        gate_loss_weight=float(cfg.get("gate_loss_weight", 1.0)),
        seed=int(cfg.get("seed", 7)),
        device=args.device, amp=args.amp,
        save_every=int(cfg.get("save_every", 10)),
        checkpoint_path=out,
        resume_from=args.resume, init_from=args.init,
        extra_meta=extra_meta,
        gated=(cfg.get("merge_method") == "gated"),
    )
    print(f"完成：{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
