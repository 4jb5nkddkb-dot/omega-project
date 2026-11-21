"""归一化层：RMSNorm（与 NumPy 单体 `_norm_fwd` 的 use_rms=True 分支逐式对应）。

单体公式（`dual_transformer_numpy.py`，LN_EPS = 1e-5）：

    r  = 1 / sqrt(mean(x^2, axis=-1) + eps)
    xn = x * r
    y  = xn * weight

没有偏置项（use_rms=True 时单体也不建 norm bias）。`use_rms=False` 的
LayerNorm 分支不由这里实现——block 里直接用 `nn.LayerNorm`，其
weight/bias 键名（`norm1.weight` / `norm1.bias`）与单体一致。
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 与单体 LN_EPS 相同。
LN_EPS = 1e-5


class RMSNorm(nn.Module):
    """RMSNorm：只含 weight（D,)，无 bias。

    参数名就是 `weight`，挂在 block 下即 `stem.0.norm1.weight`，
    与 NumPy 参数名一一对应。
    """

    def __init__(self, d_model: int, eps: float = LN_EPS):
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与单体完全相同的数学：均值在输入 dtype 下算，r 用 rsqrt。
        r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * r * self.weight

    def extra_repr(self) -> str:
        return f"{self.d_model}, eps={self.eps}"
