"""Transformer block：pre-norm + 注意力残差 + FFN 残差。

与单体 `_block_fwd` 逐式对应：

    n1 = norm1(x);  a = attn(n1);  a = dropout(a);  x1 = x + a
    n2 = norm2(x1); f = ffn(n2);   f = dropout(f);  y  = x1 + f

dropout 只作用在两条残差分支的输出上（注意力输出与 FFN 输出），
与单体 `_dropout_mask` 的 inverted dropout（保留项缩放 1/(1-p)）
一致——`nn.Dropout` 在训练模式下正是 inverted dropout，eval
模式下恒等。单体只在 `loss_and_grads` / 训练时传 dropout，推理
与生成永不 dropout；torch 里这对应模块的 train/eval 模式。

子模块名固定为 norm1 / attn / norm2 / ff_gate / ff_up / ff2
（或 ff1 / ff2），使 state_dict 键为 `<prefix>.norm1.weight`、
`<prefix>.attn.qkv.weight` 等。FFN 的 Linear 直接挂在 block 下
（不用 `SwiGLU` 子模块），否则键名会多一层 `ffn.` 前缀——
见 `feedforward.py` 的说明。

`prefix` 参数：仅作记录/校验用（例如 `"stem.0."`），state_dict
的键名实际由父模块的 ModuleList 位置自然决定；传与不传结果相同。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import ff_dim, gelu_forward, swiglu_forward
from model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """单个 pre-norm Transformer block。"""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None,
                 ff_mode: str = "swiglu", use_rms: bool = True,
                 pos_mode: str = "rope", dropout: float = 0.0,
                 prefix: str = ""):
        super().__init__()
        if ff_mode not in ("swiglu", "gelu"):
            raise ValueError(f'ff_mode must be "swiglu" or "gelu", got {ff_mode!r}')
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
        self.ff_mode = ff_mode
        self.use_rms = bool(use_rms)
        self.pos_mode = pos_mode
        self.prefix = prefix  # 仅记录；键名由父 ModuleList 位置决定
        norm = RMSNorm if use_rms else nn.LayerNorm
        self.norm1 = norm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, self.n_kv_heads, pos_mode)
        self.norm2 = norm(d_model)
        hidden = ff_dim(d_model, ff_mode)
        if ff_mode == "swiglu":
            self.ff_gate = nn.Linear(d_model, hidden, bias=True)
            self.ff_up = nn.Linear(d_model, hidden, bias=True)
            self.ff2 = nn.Linear(hidden, d_model, bias=True)
        else:
            self.ff1 = nn.Linear(d_model, hidden, bias=True)
            self.ff2 = nn.Linear(hidden, d_model, bias=True)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor,
                past: tuple[torch.Tensor, torch.Tensor] | None = None,
                offset: int = 0) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, S, D)。返回 (y, (k_full, v_full))。"""
        a, kv = self.attn(self.norm1(x), past=past, offset=offset)
        x1 = x + self.dropout(a)
        n2 = self.norm2(x1)
        if self.ff_mode == "swiglu":
            f = swiglu_forward(n2, self.ff_gate, self.ff_up, self.ff2)
        else:
            f = gelu_forward(n2, self.ff1, self.ff2)
        return x1 + self.dropout(f), kv

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_kv_heads={self.n_kv_heads}, ff_mode={self.ff_mode!r}")
