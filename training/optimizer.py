"""优化器与学习率调度（PyTorch 版，从单体 dual_transformer_numpy.py 抽出）。

- build_optimizer：torch.optim.AdamW，两组参数——ndim>=2 的参数
  （矩阵、embedding）wd=weight_decay，ndim<2 的参数（bias、norm 权重）
  wd=0.0；与 NumPy AdamWNP 的规则一致（`wd if params[k].ndim >= 2 else 0.0`）。
- lr_at：与 NumPy 版逐行同一纯函数（线性 warmup + cosine 衰减到
  min_lr_ratio）；训练器每步手动给 param group 设 lr（不用 LambdaLR——
  手动设置精确且断点续跑安全，调度状态只由步数决定）。
- clip_grad_norm：torch.nn.utils.clip_grad_norm_ 的薄包装，返回裁剪前
  的全局 L2 范数（与 NumPy clip_grad_norm 同一语义）。
"""
import math

import torch

__all__ = ["build_optimizer", "lr_at", "clip_grad_norm"]


def build_optimizer(model, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
    """AdamW，两组参数：ndim>=2 用 wd=weight_decay，ndim<2 用 wd=0.0。

    model：nn.Module（或任何带 named_parameters() 的对象）。
    只收 requires_grad=True 的参数。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": float(weight_decay),
                       "betas": betas, "eps": eps})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0,
                       "betas": betas, "eps": eps})
    if not groups:
        raise ValueError("build_optimizer: model has no trainable parameters")
    return torch.optim.AdamW(groups)


def clip_grad_norm(parameters, max_norm: float) -> float:
    """按全局 L2 范数裁剪梯度（in place），返回裁剪前的范数。"""
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))


def lr_at(step: int, base_lr: float, schedule: str = "constant", warmup_steps: int = 0,
          total_steps: int = 1, min_lr_ratio: float = 0.1) -> float:
    """Learning rate for optimizer update `step` (1-indexed).

    Pure function of the step index, so resuming from a checkpoint needs no scheduler state.
    Warmup (linear, from 0 to base_lr) applies to both schedules.
    与单体 lr_at 逐行一致（2026-11-21 核对过）。
    """
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
