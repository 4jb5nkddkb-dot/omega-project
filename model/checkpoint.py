"""检查点读写（.pt 格式）。

格式（见 CONTRACT.md，与 NumPy `.npz` 的 `p_/m_/v_/meta_json`
一一对应，供 `scripts/convert_npz_to_pt.py` 做 1:1 转换）：

    {
      "model": state_dict,            # 键名与 NumPy 参数名完全一致
      "optimizer": {"m": {name: tensor}, "v": {name: tensor}, "t": int}
                     | None,         # NumPy 式命名：m/v = Adam exp_avg/exp_avg_sq
      "step": int,
      "config": dict,                # DualTransformer.config()
      "extra": dict,
      "rng_state": ...,              # 原样存放（一般为 torch.get_rng_state()）
    }

恢复时的教训（2026-11-21，NumPy chunk 3 OOM）：续跑只加载检查点
一次。本模块的 `read_checkpoint_meta` 用 `map_location="meta"`
只读 pickle 头与元数据、不读 1 GB 的参数数据，供续跑前恢复
flags 与步数；`load_checkpoint` 完整读一次后调用方应立即 `del`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _optimizer_state_to_np_style(model: nn.Module, optimizer) -> dict | None:
    """把 torch AdamW 的 state 转成 NumPy 式 {"m", "v", "t"}。

    optimizer 也可以直接是已经整理好的 {"m": ..., "v": ..., "t": ...}
    dict（转换脚本等场景），此时原样返回。
    """
    if optimizer is None:
        return None
    if isinstance(optimizer, dict):
        return {"m": optimizer["m"], "v": optimizer["v"], "t": int(optimizer["t"])}
    state = optimizer.state_dict()
    param_names = [name for name, _ in model.named_parameters()]
    m, v, t = {}, {}, 0
    for idx, group_state in state["state"].items():
        # torch 的 state 以参数在模型中的顺序编号（param_groups 顺序）。
        name = param_names[int(idx)]
        if "exp_avg" in group_state:
            m[name] = group_state["exp_avg"].detach().cpu().clone()
        if "exp_avg_sq" in group_state:
            v[name] = group_state["exp_avg_sq"].detach().cpu().clone()
        step = group_state.get("step")
        if step is not None:
            t = max(t, int(torch.as_tensor(step).item()))
    return {"m": m, "v": v, "t": t}


def save_checkpoint(model: nn.Module, optimizer, path, step: int,
                    extra: dict | None = None, rng_state: Any = None) -> Path:
    """保存检查点（.pt）。optimizer 为 torch AdamW 或 None
    或已整理好的 {"m", "v", "t"} dict。返回写出的 Path。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": _optimizer_state_to_np_style(model, optimizer),
        "step": int(step),
        "config": model.config() if hasattr(model, "config") else {},
        "extra": dict(extra) if extra else {},
        "rng_state": rng_state,
    }
    torch.save(payload, target)
    return target


def load_checkpoint(path) -> dict:
    """完整读回一个 .pt 检查点 dict（torch.load，map_location="cpu"）。

    调用方恢复模型与优化器后应立即 `del` 返回的 dict（只加载一次，
    不要像 NumPy 旧续跑那样读三遍）。
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def read_checkpoint_meta(path) -> dict:
    """只读检查点的小元数据（config/step/extra），不加载参数数组。

    用 `map_location="meta"` 实现：只解析 pickle 头与张量元信息，
    1 GB 的参数数据不读入内存。返回 {"config", "step", "extra",
    "has_optimizer", "format_version"}。
    """
    payload = torch.load(path, map_location="meta", weights_only=False)
    return {
        "config": payload.get("config", {}),
        "step": int(payload.get("step", 0)),
        "extra": payload.get("extra", {}),
        "has_optimizer": payload.get("optimizer") is not None,
        "format_version": 1,
    }


def last_checkpoint_path(path) -> Path:
    """最佳检查点路径对应的 latest-state 路径：`<stem>_last.pt`
   （与 NumPy `last_checkpoint_path` 的 `<stem>_last.npz` 同规则）。"""
    p = Path(path)
    return p.with_name(f"{p.stem}_last{p.suffix}")
