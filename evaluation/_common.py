#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针共用：从 .pt 检查点加载 PyTorch 模型 + 字符分词器。

检查点格式见 CONTRACT.md（model/checkpoint.py）：.pt dict 含
"model"（state_dict，键与 NumPy 参数名 1:1）、"config"、"step"、"extra"。
只 load 一次（read_checkpoint_meta 的教训：续跑绝不把检查点读三遍）。

词表：staging data/processed/generalist_vocab.json（4619 字 + BOS/EOS/UNK
= 4622，与 configs/model_gated.yaml 一致）。找不到 staging 词表时
回退到 ~/workspace/dual_sft/generalist_vocab.json（只读）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DualTransformer.__init__ 接受的键（CONTRACT.md）。检查点 config 里
# 多出来的键（如 NumPy 的 scaled_init）在这里过滤掉，不往构造器里传。
_MODEL_KWARGS = (
    "d_model", "n_blocks", "n_heads", "n_kv_heads", "merge_method",
    "vocab_size", "max_seq", "tie_embeddings", "dropout", "use_rms",
    "pre_norm", "pos_mode", "ff_mode", "init_style", "n_stem_blocks",
    "final_norm",
)


def vocab_path() -> Path:
    """词表文件：优先 staging，缺失时回退 live 仓库（只读）。"""
    staged = PROJECT_ROOT / "data" / "processed" / "generalist_vocab.json"
    if staged.exists():
        return staged
    live = Path.home() / "workspace" / "dual_sft" / "generalist_vocab.json"
    return live


def load_tokenizer(path=None):
    """CharTokenizer（纯 Python/NumPy，无 torch 依赖）。"""
    from tokenizer import CharTokenizer
    p = Path(path) if path else vocab_path()
    meta = json.loads(p.read_text(encoding="utf-8"))
    return CharTokenizer(meta["chars"])


def load_model(checkpoint: str, device: str = "cpu"):
    """加载 .pt 检查点，返回 (model, step, config, extra)。

    model/ 由 Agent A 提供；如果还没就绪，这里报明确的错而不是
    ImportError 堆栈——探针脚本在无模型时应优雅退出。
    """
    try:
        from model import DualTransformer
        from model.checkpoint import load_checkpoint
    except ImportError as e:
        raise RuntimeError(
            "model/ 包尚未就绪（Agent A 并行建设中）："
            "需要 model.DualTransformer 与 model.checkpoint.load_checkpoint。"
            f"原始错误：{e}"
        ) from e
    ckpt = load_checkpoint(checkpoint)
    config = dict(ckpt.get("config") or {})
    kwargs = {k: config[k] for k in _MODEL_KWARGS if k in config}
    model = DualTransformer(**kwargs)
    state = ckpt.get("model")
    if state is None:
        raise ValueError(f"{checkpoint}: 检查点里没有 'model' state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"{checkpoint}: state_dict 与模型不匹配 "
            f"missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.eval()
    return model, int(ckpt.get("step", 0)), config, dict(ckpt.get("extra") or {})


def load_model_and_tokenizer(checkpoint: str, device: str = "cpu",
                             vocab=None):
    """(model, tokenizer, step, config, extra) 一次性加载。"""
    model, step, config, extra = load_model(checkpoint, device)
    tok = load_tokenizer(vocab)
    return model, tok, step, config, extra


def probe_records_path(name: str = "probe_format_records.jsonl") -> Path:
    """探针格式训练记录：优先 staging data/processed，回退 live 仓库。"""
    staged = PROJECT_ROOT / "data" / "processed" / name
    if staged.exists():
        return staged
    return Path.home() / "workspace" / "dual_sft" / name
