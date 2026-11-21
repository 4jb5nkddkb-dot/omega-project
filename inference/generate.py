#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch 自回归生成：采样 + DecodeController + 门控查看。

Port of the NumPy sampling helpers in
~/workspace/omega-chat/dual_transformer_numpy.py (_filter_probs 478,
_pick_token 492, token_entropy 509, suffix_repeats 518,
banned_next_tokens 527, DecodeController 545, generate_cached 1043).

行为对齐点（与 NumPy 版一致）：
  - temperature == 0 时取 argmax（贪心），不采样；
  - top-k / top-p 在 softmax 后的概率上过滤再重归一化；
  - DecodeController 的语义（n-gram 阻塞、循环升温、熵自适应）逐条照搬；
  - KV 缓存用模型的 forward_cached 做增量解码，窗口满了（len == max_seq）
    时像 NumPy generate_cached 一样用最后 max_seq 个 token 重新 prefill
    （位置从 0 重启）。
"""
from __future__ import annotations

import collections
import math
from typing import Callable, Optional, Sequence, Union

import torch

__all__ = [
    "filter_probs",
    "pick_token",
    "token_entropy",
    "suffix_repeats",
    "banned_next_tokens",
    "DecodeController",
    "generate",
    "generate_with_gate",
]


# ----------------------------------------------------------------------------
# 采样 helper（_filter_probs / _pick_token 的 torch 版）
# ----------------------------------------------------------------------------

def filter_probs(p: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Keep the top-k entries and/or the smallest nucleus with mass >= top_p,
    then renormalise. p: 1-D probability tensor (sums to 1)."""
    p = p.clone()
    if top_k and 0 < top_k < p.numel():
        cutoff = torch.topk(p, top_k).values.min()
        p = torch.where(p >= cutoff, p, torch.zeros_like(p))
    if 0.0 < top_p < 1.0:
        order = torch.argsort(p, descending=True)
        sorted_p = p[order]
        cum_before = torch.cumsum(sorted_p, dim=0) - sorted_p
        keep = torch.zeros_like(p, dtype=torch.bool)
        keep[order[cum_before < top_p]] = True
        p = torch.where(keep, p, torch.zeros_like(p))
    return p / p.sum()


def _as_float_tensor(logits) -> torch.Tensor:
    if isinstance(logits, torch.Tensor):
        return logits.detach().to(dtype=torch.float64).reshape(-1)
    return torch.as_tensor(logits, dtype=torch.float64).reshape(-1)


def pick_token(logits, temperature: float = 0.0,
               generator: Optional[torch.Generator] = None,
               top_k: int = 0, top_p: float = 1.0) -> int:
    """Sample one token id from logits.

    temperature == 0 -> argmax. Otherwise softmax(logits / temperature),
    then top-k / top-p filtering, then multinomial sampling.
    generator: torch.Generator for reproducible sampling (None -> global RNG).
    """
    z = _as_float_tensor(logits)
    if temperature == 0:
        return int(torch.argmax(z).item())
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    z = z / temperature
    p = torch.softmax(z - z.max(), dim=0)
    p = filter_probs(p, top_k, top_p)
    # multinomial 在 CPU 上做：generator 是 CPU 的，而 logits 可能在 CUDA 上。
    return int(torch.multinomial(p.float().cpu(), 1,
                                 generator=generator).item())


def token_entropy(logits, temperature: float = 1.0) -> float:
    """Shannon entropy (nats) of softmax(logits / temperature).
    temperature <= 0 is treated as 1 (same as the NumPy version)."""
    z = _as_float_tensor(logits) / (temperature if temperature > 0 else 1.0)
    z = z - z.max()
    p = torch.softmax(z, dim=0)
    return float(-(p * torch.log(p + 1e-12)).sum().item())


def suffix_repeats(ids, n: int = 8, window: int = 128) -> bool:
    """True if the last n tokens already occurred earlier inside the recent window (a loop)."""
    if n <= 0 or len(ids) <= n:
        return False
    tail = list(ids[-window:])
    suffix = tuple(tail[-n:])
    return any(tuple(tail[i:i + n]) == suffix for i in range(len(tail) - n))


def banned_next_tokens(ids, n: int = 4, window: int = 64) -> set:
    """Tokens that would complete an n-gram already seen in the recent window.

    Blocking every such token guarantees the output never repeats an n-gram
    inside the window.
    """
    if n <= 0:
        return set()
    tail = list(ids[-window:])
    if len(tail) < n:
        return set()
    prefix = tuple(tail[len(tail) - (n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(tail) - n + 1):
        if tuple(tail[i:i + n - 1]) == prefix:
            banned.add(int(tail[i + n - 1]))
    return banned


class DecodeController:
    """Runtime controller for sampling (port of the NumPy DecodeController).

    - Blocks tokens that would recreate an n-gram already in the recent window (block_ngram).
    - Flags looping when the last `loop_len` tokens already occurred earlier in the window.
      While looping, the temperature is pushed up; with base temperature 0 the step samples
      at loop_temperature instead of taking the argmax.
    - Tracks the rolling mean entropy of the distribution actually sampled from and adjusts
      the temperature multiplicatively toward target_entropy: low entropy (collapse) raises
      it, high entropy (drift) lowers it. Disable with adapt=False.
    """

    def __init__(self, base_temperature: float = 0.0, target_entropy: float = 1.0,
                 gain: float = 0.5, window: int = 16, block_ngram: int = 4,
                 loop_len: int = 8, loop_temperature: float = 0.8,
                 t_min: float = 0.2, t_max: float = 1.5, adapt: bool = True):
        if base_temperature < 0 or t_min <= 0 or t_max < t_min:
            raise ValueError("need base_temperature >= 0 and 0 < t_min <= t_max")
        self.base = float(base_temperature)
        self.target = float(target_entropy)
        self.gain = float(gain)
        self.hist = collections.deque(maxlen=window)
        self.block_ngram = block_ngram
        self.loop_len = loop_len
        self.loop_temperature = float(loop_temperature)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.adapt = adapt
        self.temp = self.base
        self.trace = []  # one (entropy, temperature, looping, n_blocked) tuple per step

    def _next_temperature(self, looping: bool) -> float:
        if not self.adapt:
            return self.base
        if self.base == 0:
            return self.loop_temperature if looping else 0.0
        if looping:
            self.temp = min(self.temp * 1.25, self.t_max)
        elif self.hist:
            mean_h = float(sum(self.hist) / len(self.hist))
            self.temp = float(min(max(
                self.temp * math.exp(self.gain * (self.target - mean_h)),
                self.t_min), self.t_max))
        return self.temp

    def choose(self, logits, ids, generator=None, top_k: int = 0,
               top_p: float = 1.0) -> int:
        z = _as_float_tensor(logits).clone()
        looping = suffix_repeats(ids, self.loop_len) if self.loop_len else False
        banned = banned_next_tokens(ids, self.block_ngram) if self.block_ngram else set()
        if banned:
            masked = z.clone()
            for t in banned:
                masked[t] = -float("inf")
            if torch.isfinite(masked).any():
                z = masked
        temp = self._next_temperature(looping)
        h = token_entropy(z, temp)
        self.hist.append(h)
        self.trace.append((h, temp, looping, len(banned)))
        return pick_token(z, temp, generator, top_k, top_p)

    def summary(self) -> str:
        if not self.trace:
            return "decode: no steps"
        hs = [t[0] for t in self.trace]
        ts = [t[1] for t in self.trace]
        return (f"decode: steps={len(hs)} mean_entropy={sum(hs) / len(hs):.3f} "
                f"loop_steps={sum(t[2] for t in self.trace)} "
                f"blocked_candidates={sum(t[3] for t in self.trace)} "
                f"temp_range=[{min(ts):.2f},{max(ts):.2f}]")


# ----------------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------------

def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encode(tokenizer, prompt: str):
    """Prompt 文本 -> token id 列表（BOS 在前，与 NumPy Sidecar 一致）。"""
    if isinstance(prompt, (list, tuple)):
        return [int(t) for t in prompt]
    return tokenizer.encode(prompt, add_bos=True)


@torch.no_grad()
def generate_ids(model, tokenizer, prompt: Union[str, Sequence[int]],
                 max_new_tokens: int = 100, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 1.0,
                 device: Optional[Union[str, torch.device]] = None,
                 controller: Optional[DecodeController] = None,
                 logit_hook: Optional[Callable] = None,
                 stop_fn: Optional[Callable[[list], bool]] = None,
                 stop_on_eos: bool = False,
                 seed: Optional[int] = None) -> list:
    """KV-cached generation; returns the FULL id list (prompt + new tokens).

    Port of NumPy DualTransformerNP.generate_cached: incremental decode via
    model.forward_cached(tokens, pasts); when the context reaches max_seq
    the cache is rebuilt from the last max_seq tokens (positions restart
    at 0). If a DecodeController is given it picks each token (its base
    temperature replaces `temperature`); otherwise the plain sampler is used.
    logit_hook(result, logits) -> logits may mask tokens before sampling.
    stop_fn(result) -> bool ends generation early after the token just
    appended. stop_on_eos ends generation when the tokenizer's EOS id is
    emitted (off by default, matching the NumPy sidecar which never stops
    on EOS). seed seeds a torch.Generator for reproducible sampling.
    """
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens must be non-negative and temperature >= 0")
    dev = torch.device(device) if device is not None else _model_device(model)
    max_seq = int(getattr(model, "max_seq", 512))
    eos_id = getattr(tokenizer, "eos_id", None) if not isinstance(prompt, (list, tuple)) else None
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    result = _encode(tokenizer, prompt)
    if max_new_tokens == 0:
        return result
    ctx = result[-max_seq:]
    logits, pasts = model.forward_cached(
        torch.tensor([ctx], dtype=torch.long, device=dev))
    for generated_i in range(max_new_tokens):
        step_logits = logits[0, -1]
        if logit_hook is not None:
            step_logits = logit_hook(result, step_logits)
        if controller is not None:
            nxt = controller.choose(step_logits, result, generator, top_k, top_p)
        else:
            nxt = pick_token(step_logits, temperature, generator, top_k, top_p)
        result.append(nxt)
        if stop_on_eos and eos_id is not None and nxt == eos_id:
            break
        if stop_fn is not None and stop_fn(result):
            break
        if generated_i + 1 == max_new_tokens:
            break
        if len(ctx) < max_seq:
            logits, pasts = model.forward_cached(
                torch.tensor([[nxt]], dtype=torch.long, device=dev), pasts)
            ctx.append(nxt)
        else:
            # Window is full: re-prefill from the last max_seq tokens.
            ctx = result[-max_seq:]
            logits, pasts = model.forward_cached(
                torch.tensor([ctx], dtype=torch.long, device=dev))
    return result


def generate(model, tokenizer, prompt: Union[str, Sequence[int]],
             max_new_tokens: int = 100, temperature: float = 0.0,
             top_k: int = 0, top_p: float = 1.0,
             device: Optional[Union[str, torch.device]] = None,
             controller: Optional[DecodeController] = None,
             logit_hook: Optional[Callable] = None,
             stop_fn: Optional[Callable[[list], bool]] = None,
             stop_on_eos: bool = False,
             seed: Optional[int] = None) -> str:
    """Generate a continuation for `prompt` and return it as decoded text.

    `prompt` is a str (encoded with BOS) or an explicit id list. Only the
    NEW tokens are decoded and returned (the prompt itself is not echoed).
    """
    prompt_ids = _encode(tokenizer, prompt)
    full = generate_ids(model, tokenizer, prompt_ids,
                        max_new_tokens=max_new_tokens, temperature=temperature,
                        top_k=top_k, top_p=top_p, device=device,
                        controller=controller, logit_hook=logit_hook,
                        stop_fn=stop_fn, stop_on_eos=stop_on_eos, seed=seed)
    new_ids = full[len(prompt_ids):]
    return tokenizer.decode(new_ids)


@torch.no_grad()
def generate_with_gate(model, tokenizer, prompt: Union[str, Sequence[int]],
                       max_new_tokens: int = 100, temperature: float = 0.0,
                       top_k: int = 0, top_p: float = 1.0,
                       device: Optional[Union[str, torch.device]] = None,
                       controller: Optional[DecodeController] = None,
                       logit_hook: Optional[Callable] = None,
                       stop_fn: Optional[Callable[[list], bool]] = None,
                       stop_on_eos: bool = False,
                       seed: Optional[int] = None):
    """Like generate(), but also returns the gate value g of the prompt.

    Returns (text, gate): gate is the per-sequence left-branch percentage
    (float in [0, 1]) from model.gate_values(prompt_ids) — the SAME g the
    model uses to split the input and blend the outputs. Raises ValueError
    for a non-gated model (gate_values does), matching the contract.
    """
    prompt_ids = _encode(tokenizer, prompt)
    dev = torch.device(device) if device is not None else _model_device(model)
    g = model.gate_values(
        torch.tensor([prompt_ids], dtype=torch.long, device=dev))
    gate = float(g.reshape(-1)[0].item())
    text = generate(model, tokenizer, prompt_ids,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, top_p=top_p, device=device,
                    controller=controller, logit_hook=logit_hook,
                    stop_fn=stop_fn, stop_on_eos=stop_on_eos, seed=seed)
    return text, gate
