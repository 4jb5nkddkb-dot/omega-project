"""DualTransformer：双半脑 Transformer（PyTorch 版）。

结构（与 NumPy 单体 `DualTransformerNP._forward` 一致）：

    tokens -> emb（+ learned wpe）-> stem（仅 gated，n_stem_blocks 块）
      -> 门 g（每序列一个数）分流 -> left / right 各跑剩余块
      -> 合并（gated：g*left + (1-g)*right；single：left；
         average：均值；concat：拼接）-> final_norm -> logits

门（generalist-v5，见 `dual_sft/GATED_DUAL_DESIGN.md`）：

    g_seq = sigmoid(w · mean_pool(h_stem) + b)，形状 (B, 1, 1)，
    同一序列内所有 token 共用一个 g，广播到 (B, S, 1) 后
    输入分流（g*h_stem / (1-g)*h_stem）与输出混合用同一个 g。

增量解码（`forward_cached`）：门要对“已看到的全部上下文”池化，
所以 stem 隐状态的和与 token 数存在 pasts 的 `"gate"` 项里随
cache 传递（`{"sum": (B, D), "count": int}`），否则每步只看到
一个 token，门就退化回每 token 查表。stem 隐状态本身是因果的，
池化和是精确的。

state_dict 键名与 NumPy 参数名逐一对应（这是 .npz -> .pt 1:1
转换的前提）：`emb` 没有 `.weight` 后缀，故用 plain
`nn.Parameter` + `F.embedding`，不用 `nn.Embedding`；stem/left/
right 是 `ModuleList`，键自然为 `stem.0.norm1.weight` 等；
`gate` 是 `nn.Linear(D, 1)`，键为 `gate.weight` (1, D) /
`gate.bias` (1,)；tied 时没有 `lm_head.weight`。

初始化：GPT-2 式（std 0.02，残差分支 attn.proj / ff2 的输出
缩放 `1/sqrt(2*n_blocks)`），与单体 `_init_params` 一致；
`init_style="legacy"` 恢复单体的均匀初始化。门 BCE 的 float32
eps 问题（单体 `loss_and_grads` 里 eps=1e-7 的典故）在新代码里
不存在——训练侧用 `F.binary_cross_entropy_with_logits`
作用在 pre-sigmoid logit 上（见 `training/loss.py`，Agent B）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import TransformerBlock
from model.norm import RMSNorm

MERGE_METHODS = ("single", "average", "concat", "gated")


class DualTransformer(nn.Module):
    def __init__(self, d_model: int = 1024, n_blocks: int = 4, n_heads: int = 16,
                 n_kv_heads: int | None = 8, merge_method: str = "gated",
                 vocab_size: int = 4622, max_seq: int = 512,
                 tie_embeddings: bool = True, dropout: float = 0.1,
                 use_rms: bool = True, pre_norm: bool = True,
                 pos_mode: str = "rope", ff_mode: str = "swiglu",
                 init_style: str = "gpt2", n_stem_blocks: int = 1,
                 final_norm: bool = True, dtype: torch.dtype = torch.float32,
                 scaled_init: bool = True):
        super().__init__()
        if merge_method not in MERGE_METHODS:
            raise ValueError(f"merge_method must be one of {MERGE_METHODS}")
        if d_model <= 0 or n_blocks <= 0 or max_seq <= 0:
            raise ValueError("d_model, n_blocks, max_seq must be positive")
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model/n_heads must be even")
        if not pre_norm:
            raise ValueError("post-norm is not supported (the NumPy model is pre-norm only)")
        self.d_model, self.max_seq = int(d_model), int(max_seq)
        self.n_blocks, self.n_heads, self.use_rms = int(n_blocks), int(n_heads), bool(use_rms)
        self.n_stem_blocks = int(n_stem_blocks)
        if merge_method == "gated" and not (1 <= self.n_stem_blocks < n_blocks):
            raise ValueError("gated needs 1 <= n_stem_blocks < n_blocks "
                             "(a shared stem plus at least one block per branch)")
        if tie_embeddings and merge_method == "concat":
            raise ValueError("tie_embeddings needs an output width equal to d_model (not concat)")
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_kv_heads <= 0 or n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if pos_mode not in ("rope", "learned"):
            raise ValueError('pos_mode must be "rope" or "learned"')
        if ff_mode not in ("gelu", "swiglu"):
            raise ValueError('ff_mode must be "gelu" or "swiglu"')
        if init_style not in ("gpt2", "legacy"):
            raise ValueError('init_style must be "gpt2" or "legacy"')
        self.merge_method = merge_method
        self.vocab_size = int(vocab_size)
        self.tie_embeddings = bool(tie_embeddings)
        self.scaled_init = bool(scaled_init)
        self.n_kv_heads = int(n_kv_heads)
        self.pos_mode = pos_mode
        self.ff_mode = ff_mode
        self.init_style = init_style
        self.pre_norm = True
        self.dropout = float(dropout)
        self.dtype = dtype

        D, V = self.d_model, self.vocab_size
        # 注意：emb 没有 .weight 后缀——plain Parameter + F.embedding，
        # 不用 nn.Embedding（它的键会是 emb.weight，与 NumPy 对不上）。
        self.emb = nn.Parameter(torch.empty(V, D))
        if self.pos_mode == "learned":
            self.wpe = nn.Parameter(torch.empty(self.max_seq, D))
        else:
            self.wpe = None

        def _blocks(n: int, prefix: str):
            return nn.ModuleList([
                TransformerBlock(D, n_heads, self.n_kv_heads, ff_mode, use_rms,
                                 pos_mode, dropout, prefix=f"{prefix}{i}.")
                for i in range(n)
            ])

        if merge_method == "gated":
            self.stem = _blocks(self.n_stem_blocks, "stem.")
            self.left = _blocks(self.n_blocks - self.n_stem_blocks, "left.")
            self.right = _blocks(self.n_blocks - self.n_stem_blocks, "right.")
            self.gate = nn.Linear(D, 1, bias=True)
        else:
            self.stem = None
            self.gate = None
            self.left = _blocks(self.n_blocks, "left.")
            self.right = (_blocks(self.n_blocks, "right.")
                          if merge_method in ("average", "concat") else None)

        out_dim = 2 * D if merge_method == "concat" else D
        self._out_dim = out_dim
        if final_norm:
            norm = RMSNorm if use_rms else nn.LayerNorm
            self.final_norm = norm(out_dim)
        else:
            self.final_norm = None
        if not tie_embeddings:
            # 注意参数顺序：NumPy linear("lm_head", V, out_dim) 的权重是
            # (V, out_dim)，即 nn.Linear(in=out_dim, out=V)。
            self.lm_head = nn.Linear(out_dim, V, bias=False)
        else:
            self.lm_head = None

        self._init_weights()
        if dtype != torch.float32:
            self.to(dtype)

    # ----- 结构 -----

    def config(self) -> dict:
        """与 NumPy `model.config()` 完全相同的 dict（键与值一致）。"""
        return {
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "merge_method": self.merge_method,
            "use_rms": self.use_rms,
            "max_seq": self.max_seq,
            "vocab_size": self.vocab_size,
            "final_norm": self.final_norm is not None,
            "tie_embeddings": self.tie_embeddings,
            "scaled_init": self.scaled_init,
            "n_kv_heads": self.n_kv_heads,
            "pos_mode": self.pos_mode,
            "ff_mode": self.ff_mode,
            "n_stem_blocks": self.n_stem_blocks,
        }

    def _head_weight(self) -> torch.Tensor:
        return self.emb if self.tie_embeddings else self.lm_head.weight

    def _branches(self) -> tuple[str, ...]:
        return ("left",) if self.merge_method == "single" else ("left", "right")

    def _n_branch_blocks(self) -> int:
        if self.merge_method == "gated":
            return self.n_blocks - self.n_stem_blocks
        return self.n_blocks

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def _init_weights(self) -> None:
        """GPT-2 式初始化（与单体 `_init_params` 同分布；随机流不同，
        故两个后端的权重数值不相同，只保证分布与参数量一致）。

        残差分支输出（attn.proj / ff2）缩放 1/sqrt(2*n_blocks)；
        norm 权重为 1、偏置为 0；emb std 0.02；gate bias 为 0，
        起步 g≈0.5（两边平分）。
        """
        gpt2 = self.init_style == "gpt2"
        res_scale = 1.0 / math.sqrt(2 * self.n_blocks) if self.scaled_init else 1.0
        with torch.no_grad():
            if gpt2:
                nn.init.normal_(self.emb, std=0.02)
            else:
                nn.init.normal_(self.emb, std=1.0)
            if self.wpe is not None:
                if gpt2:
                    nn.init.normal_(self.wpe, std=0.02)
                else:
                    nn.init.normal_(self.wpe, std=1.0)
            for name, mod in self.named_modules():
                if isinstance(mod, (RMSNorm, nn.LayerNorm)):
                    nn.init.ones_(mod.weight)
                    if getattr(mod, "bias", None) is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.Linear):
                    scale = res_scale if (name.endswith("attn.proj")
                                          or name.endswith("ff2")) else 1.0
                    if gpt2:
                        nn.init.normal_(mod.weight, std=0.02 * scale)
                        if mod.bias is not None:
                            nn.init.zeros_(mod.bias)
                    else:
                        bound = scale / math.sqrt(mod.in_features)
                        nn.init.uniform_(mod.weight, -bound, bound)
                        if mod.bias is not None:
                            nn.init.uniform_(mod.bias, -bound, bound)

    # ----- 前向 -----

    def _check_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] == 0:
            raise ValueError("empty sequences are not supported")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(f"token IDs must be in [0, {self.vocab_size - 1}]")
        return tokens

    @staticmethod
    def _past_len(pasts: dict | None) -> int:
        """已缓存长度（从 stem 或 left 任一分支的首层 K 取；与单体一致）。"""
        if not pasts:
            return 0
        for key in ("stem", "left"):
            branch = pasts.get(key)
            if branch and branch[0] is not None:
                return branch[0][0].shape[2]
        return 0

    def _stem_gate(self, x: torch.Tensor, pasts: dict | None, offset: int):
        """跑共享 stem 并算门。返回 (h_stem, g, gate_state, stem_kvs, z)。

        g: (B, S, 1)，同一序列内值相同；z: (B, 1, 1) 为 gate 的
        pre-sigmoid logit（训练走 BCEWithLogitsLoss 时用它，不用
        sigmoid 后再 clamp 反推）；gate_state 为 {"sum", "count"}，
        增量解码时随 pasts 传递，使门对全部已见上下文池化。
        """
        h_stem = x
        stem_pasts = pasts.get("stem") if pasts is not None else None
        stem_kvs = []
        for i, blk in enumerate(self.stem):
            past = stem_pasts[i] if stem_pasts is not None else None
            h_stem, kv = blk(h_stem, past=past, offset=offset)
            stem_kvs.append(kv)
        S = x.shape[1]
        if pasts is not None and pasts.get("gate") is not None:
            gate_sum = pasts["gate"]["sum"] + h_stem.sum(dim=1)  # (B, D)
            gate_count = pasts["gate"]["count"] + S
        else:
            gate_sum = h_stem.sum(dim=1)
            gate_count = S
        h_pool = (gate_sum / gate_count)[:, None, :]  # (B, 1, D)
        z = self.gate(h_pool)  # (B, 1, 1)，pre-sigmoid logit
        g_seq = torch.sigmoid(z)
        g = g_seq.expand(-1, S, -1)  # (B, S, 1)
        return h_stem, g, {"sum": gate_sum, "count": gate_count}, stem_kvs, z

    def _forward(self, tokens: torch.Tensor, pasts: dict | None = None):
        """内部前向。返回 (logits, new_pasts, g)，g 仅 gated 时非 None。"""
        B, S = tokens.shape
        offset = self._past_len(pasts)
        if offset + S > self.max_seq:
            raise ValueError("KV cache plus input chunk exceeds max_seq; reset/re-prefill cache")
        x = F.embedding(tokens, self.emb)
        if self.pos_mode == "learned":
            x = x + self.wpe[offset:offset + S].unsqueeze(0)

        new_pasts: dict = {}
        g = None
        branch_x = None
        if self.merge_method == "gated":
            h_stem, g, gate_state, stem_kvs, _ = self._stem_gate(x, pasts, offset)
            new_pasts["stem"] = stem_kvs
            new_pasts["gate"] = gate_state
            branch_x = {"left": g * h_stem, "right": (1.0 - g) * h_stem}

        outs = {}
        for name in self._branches():
            h = branch_x[name] if branch_x is not None else x
            branch_pasts = pasts.get(name) if pasts is not None else None
            kvs = []
            for i, blk in enumerate(getattr(self, name)):
                past = branch_pasts[i] if branch_pasts is not None else None
                h, kv = blk(h, past=past, offset=offset)
                kvs.append(kv)
            outs[name] = h
            new_pasts[name] = kvs

        if self.merge_method == "gated":
            merged = g * outs["left"] + (1.0 - g) * outs["right"]
        elif self.merge_method == "single":
            merged = outs["left"]
        elif self.merge_method == "average":
            merged = (outs["left"] + outs["right"]) * 0.5
        else:  # concat
            merged = torch.cat([outs["left"], outs["right"]], dim=-1)

        h = self.final_norm(merged) if self.final_norm is not None else merged
        logits = F.linear(h, self._head_weight())
        return logits, new_pasts, g

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, S) LongTensor。返回 logits (B, S, V)。"""
        tokens = self._check_tokens(tokens)
        if tokens.shape[1] > self.max_seq:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq={self.max_seq}")
        return self._forward(tokens)[0]

    def forward_cached(self, tokens: torch.Tensor, pasts: dict | None = None):
        """增量前向；返回 (logits, new_pasts)。

        pasts 的结构与 NumPy `forward_cached` 一致：
        `{"stem": [(k, v), ...], "left": [...], "right": [...],
        "gate": {"sum": (B, D), "count": int}}`，其中 (k, v) 形状为
        (B, Hkv, T, Dh)。注意 gated 模型只保证最后位置与全序列前向
        一致（设计文档的 kv_cache_parity_test 语义）：门是非因果池化，
        增量前向早期位置的门基于前缀、与全序列门不同，只有当增量
        前缀即全序列时（最后位置）两者才相同。
        """
        tokens = self._check_tokens(tokens)
        logits, new_pasts, _ = self._forward(tokens, pasts)
        return logits, new_pasts

    def gate_values(self, tokens: torch.Tensor) -> torch.Tensor:
        """返回左支百分比 g (B, S, 1)，同一序列内值相同（仅 gated）。

        推理用 helper（no_grad）；dropout 遵循模块的 train/eval
        模式，推理前请 `model.eval()`（与 NumPy gate_values 的
        dropout=0 语义对应）。
        """
        if self.merge_method != "gated":
            raise ValueError("gate_values needs merge_method='gated'")
        tokens = self._check_tokens(tokens)
        if tokens.shape[1] > self.max_seq:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq={self.max_seq}")
        with torch.no_grad():
            x = F.embedding(tokens, self.emb)
            if self.pos_mode == "learned":
                x = x + self.wpe[:tokens.shape[1]].unsqueeze(0)
            _, g, _, _, _ = self._stem_gate(x, None, 0)
        return g

    def gate_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """返回门控的 pre-sigmoid logit z (B, S, 1)，同一序列内值相同（仅 gated）。

        训练用：门 BCE 走 F.binary_cross_entropy_with_logits(z, t)，
        数值稳定，不需要 sigmoid 后再 clamp 反推（float32 下门饱和到
        恰好 1.0 时 clamp 反推会失真；见 training/loss.py 的注释）。
        与 gate_values 的关系：gate_values(tokens) == sigmoid(gate_logits(tokens))。
        训练用：建图（不要包 no_grad，否则 BCE 损失没有梯度）。
        推理时请在外层包 no_grad 并 `model.eval()`。
        """
        if self.merge_method != "gated":
            raise ValueError("gate_logits needs merge_method='gated'")
        tokens = self._check_tokens(tokens)
        if tokens.shape[1] > self.max_seq:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq={self.max_seq}")
        x = F.embedding(tokens, self.emb)
        if self.pos_mode == "learned":
            x = x + self.wpe[:tokens.shape[1]].unsqueeze(0)
        _, _, _, _, z = self._stem_gate(x, None, 0)
        return z.expand(-1, tokens.shape[1], -1)

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_blocks={self.n_blocks}, "
                f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
                f"merge_method={self.merge_method!r}, vocab_size={self.vocab_size}")
