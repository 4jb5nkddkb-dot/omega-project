"""前馈网络：SwiGLU（默认）与 GELU（单体旧模式）。

与单体 `_block_fwd` 的 FFN 分支逐式对应：

- SwiGLU：`act = silu(ff_gate(x)) * ff_up(x)`，再过 `ff2`。
  单体写的是 `g1 * sigmoid(g1) * u`，即 `silu(g1) * u`，与
  `F.silu` 完全相同。`ff_gate` / `ff_up`: (H, D, bias)；
  `ff2`: (D, H, bias)；`H = round(8*D/3)`。
- GELU：`ff1`: (4D, D, bias)，tanh 近似 GELU（单体 `_gelu_fwd`
  用 tanh 近似，而 torch 默认 GELU 是精确 erf 形式；这里显式用
  `approximate="tanh"` 以与单体一致），再过 `ff2`: (D, 4D, bias)。

键名说明：`TransformerBlock` 必须直接持有 `ff_gate` / `ff_up` /
`ff2`（或 `ff1` / `ff2`）这几个 `nn.Linear` 属性，不能把一个
`SwiGLU` 子模块挂在 block 下——那样 state_dict 会多一层
`ffn.` 前缀（`stem.0.ffn.ff_gate.weight`），与 NumPy 参数名
对不上。所以 block 用下面两个纯函数算 FFN；这里的 `SwiGLU` /
`GELUFeedForward` 类是同一数学的独立模块形态（直接用、单测用），
`FeedForward` 工厂按 `ff_mode` 返回其中之一。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ff_dim(d_model: int, ff_mode: str) -> int:
    """FFN 中间维：SwiGLU 为 round(8*D/3)，GELU 为 4*D（与单体一致）。"""
    if ff_mode == "swiglu":
        return max(1, int(round(8 * d_model / 3)))
    if ff_mode == "gelu":
        return 4 * d_model
    raise ValueError(f'ff_mode must be "swiglu" or "gelu", got {ff_mode!r}')


def swiglu_forward(x: torch.Tensor, ff_gate: nn.Linear, ff_up: nn.Linear,
                   ff2: nn.Linear) -> torch.Tensor:
    """SwiGLU 前向（函数式，供 TransformerBlock 用，见模块 docstring）。"""
    return ff2(F.silu(ff_gate(x)) * ff_up(x))


def gelu_forward(x: torch.Tensor, ff1: nn.Linear, ff2: nn.Linear) -> torch.Tensor:
    """GELU 前向（tanh 近似，与单体 `_gelu_fwd` 一致）。"""
    return ff2(F.gelu(ff1(x), approximate="tanh"))


class SwiGLU(nn.Module):
    """SwiGLU 前馈：ff_gate / ff_up (H, D, bias)，ff2 (D, H, bias)。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = ff_dim(d_model, "swiglu")
        self.ff_gate = nn.Linear(d_model, self.hidden_dim, bias=True)
        self.ff_up = nn.Linear(d_model, self.hidden_dim, bias=True)
        self.ff2 = nn.Linear(self.hidden_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu_forward(x, self.ff_gate, self.ff_up, self.ff2)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"


class GELUFeedForward(nn.Module):
    """GELU 前馈（tanh 近似）：ff1 (4D, D, bias)，ff2 (D, 4D, bias)。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = ff_dim(d_model, "gelu")
        self.ff1 = nn.Linear(d_model, self.hidden_dim, bias=True)
        self.ff2 = nn.Linear(self.hidden_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_forward(x, self.ff1, self.ff2)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"


def FeedForward(d_model: int, ff_mode: str = "swiglu") -> nn.Module:
    """按 ff_mode 返回 SwiGLU 或 GELUFeedForward。"""
    if ff_mode == "swiglu":
        return SwiGLU(d_model)
    if ff_mode == "gelu":
        return GELUFeedForward(d_model)
    raise ValueError(f'ff_mode must be "swiglu" or "gelu", got {ff_mode!r}')
