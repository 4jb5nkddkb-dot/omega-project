"""多头注意力：RoPE + GQA + 因果掩码 + KV cache。

与 NumPy 单体 `_attn_fwd` 逐式对应：

- 融合 QKV 投影：一个 `nn.Linear(D, D + 2*n_kv_heads*Dh, bias=False)`，
  名为 `attn.qkv`（block 下的键即 `stem.0.attn.qkv.weight`）。
- 输出投影：`nn.Linear(D, D, bias=False)`，名为 `attn.proj`。
- RoPE 只作用在 Q/K 上，交错 even/odd 旋转（单体 `_rope`），
  `inv_freq = 1 / (10000 ** (arange(0, Dh, 2) / Dh))`，cos/sin 先在
  float64 下算再 cast 到输入 dtype（与单体 `_rope_tables` 一致）。
- GQA：`n_kv_heads < n_heads` 时每个 KV 头服务 `group = n_heads //
  n_kv_heads` 个 Q 头，算注意力前 repeat（单体用 `np.repeat`）。
- 因果掩码带 offset：绝对位置 p 的 query 只能看到位置 <= p 的 key；
  增量解码时 offset = 已缓存长度。
- KV cache：每步返回拼接后的完整 K/V `(k_u, v_u)`，形状
  `(B, Hkv, T, Dh)`，调用方存在 pasts 里下次传回（与单体
  `forward_cached` 的 pasts 结构一致）。

注意：这里不用 `F.scaled_dot_product_attention`，手动算 scores +
softmax，与单体的掩码（-inf）与缩放（`1/sqrt(Dh)`）逐式一致，
便于与 NumPy 做 logits 奇偶校验。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ROPE_THETA = 10000.0


def rope_tables(offset: int, length: int, dh: int, dtype: torch.dtype,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE 的 cos/sin 表，形状均为 (length, dh//2)。

    与单体 `_rope_tables` 一致：角频率在 float64 下算，再 cast。
    """
    positions = torch.arange(offset, offset + length, dtype=torch.float64,
                             device=device)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dh, 2, dtype=torch.float64,
                                                  device=device) / dh))
    ang = positions[:, None] * inv_freq[None, :]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """交错 even/odd 旋转，与单体 `_rope` 逐式一致。

    x: (B, H, S, Dh)；cos/sin: (S, Dh//2)。
    """
    xe, xo = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


class MultiHeadAttention(nn.Module):
    """多头自注意力（RoPE + GQA + KV cache）。

    子模块名固定为 `qkv` / `proj`，使 state_dict 键为
    `<prefix>.attn.qkv.weight` / `<prefix>.attn.proj.weight`。
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None,
                 pos_mode: str = "rope"):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model/n_heads must be even")
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_kv_heads <= 0 or n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if pos_mode not in ("rope", "learned"):
            raise ValueError('pos_mode must be "rope" or "learned"')
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = d_model // n_heads
        self.pos_mode = pos_mode
        self.use_rope = pos_mode == "rope"
        qkv_out = d_model + 2 * self.n_kv_heads * self.head_dim
        self.qkv = nn.Linear(d_model, qkv_out, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h: torch.Tensor,
                past: tuple[torch.Tensor, torch.Tensor] | None = None,
                offset: int = 0) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """h: (B, S, D)。返回 (out, (k_full, v_full))，k/v 形状 (B, Hkv, T, Dh)。

        past 为上一轮返回的 (k, v)；offset 为当前 chunk 的绝对起始位置
       （增量解码时 = 已缓存长度），决定 RoPE 位置与因果掩码。
        """
        B, S, D = h.shape
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.head_dim
        Dkv = Hkv * Dh
        group = H // Hkv

        qkv = self.qkv(h)
        q = qkv[..., :D].view(B, S, H, Dh).transpose(1, 2)
        k = qkv[..., D:D + Dkv].view(B, S, Hkv, Dh).transpose(1, 2)
        v = qkv[..., D + Dkv:].view(B, S, Hkv, Dh).transpose(1, 2)

        if self.use_rope:
            cos, sin = rope_tables(offset, S, Dh, q.dtype, q.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)

        # GQA：每个 KV 头服务 group 个 Q 头。
        k_all = k.repeat_interleave(group, dim=1) if group > 1 else k
        v_all = v.repeat_interleave(group, dim=1) if group > 1 else v

        Tk = k_all.shape[2]
        q_pos = torch.arange(offset, offset + S, device=h.device)
        k_pos = torch.arange(Tk, device=h.device)
        # query 在绝对位置 p，只能看到 key 位置 <= p。
        allowed = k_pos[None, :] <= q_pos[:, None]  # (S, Tk)
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        probs = F.softmax(scores, dim=-1)
        o = torch.matmul(probs, v_all)  # (B, H, S, Dh)
        o = o.transpose(1, 2).reshape(B, S, D)
        return self.proj(o), (k, v)

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_kv_heads={self.n_kv_heads}, pos_mode={self.pos_mode!r}")
