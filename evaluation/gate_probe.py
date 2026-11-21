#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门控分流探针（PyTorch 版）：门分不分得开 + 真实循环提示走哪支。

Port of dual_sft/probe_gate_v5_pilot_lean.py + probe_gate_loop_prompts.py。

A. 域窗口（lean probe）：每域取 N_WIN 个 128-token 窗口，测 mean g。
   左域（经典/代码/对话/常识/推理）应走左支（g → 1），世界域走右支
   （g → 0）。通过线：左均 > 0.6 且世界 < 0.4；趋势线：separation > 0.05。
B. 循环提示：真实循环喂给模型的提示（Q0..Q3 探针格式、各地点 Q0、
   到达叙述、探针格式训练记录）+ 语料窗对照，逐条测 g。2026-11-18
   的未解问题就靠这部分回答：QA 格式提示是不是被门送去了左支，
   导致模型不读状态摘要。

注意：合同 API 只暴露 gate_values()（g），不暴露门输入 ||h_pool||，
所以 v4/v5 诊断里的 ||h_pool|| / ||mean_left - mean_world|| 两列
在这里测不到——门输入的量级问题只能靠 gate 训练日志或 Agent A 的
内部探针看。

只读：不碰训练器，不写检查点。
用法：
  python -m evaluation.gate_probe --checkpoint checkpoints/generalist-v5_s600.pt
  python -m evaluation.gate_probe --checkpoint <ckpt.pt> --skip-domains
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation._common import load_model_and_tokenizer, probe_records_path  # noqa: E402
from evaluation.scoring import QUESTIONS, anchor_prompt  # noqa: E402
from evaluation.world_state import WorldState, world_prompt  # noqa: E402

SEQ = 128
N_WIN = 4
BATCH = 2
LEFT_DOMAINS = ("经典", "代码", "对话", "常识", "推理")

# training/dataset.py 导入失败时的最小回退（Agent B 并行改写中）：
# 只读 live 语料，与原版 generalist_pilot_train.domain_texts 同源。
_FALLBACK_WORLD_FILES = (
    "world_tool_records_v9.jsonl", "self_report_records_v9.jsonl",
    "probe_format_records.jsonl", "register_format_records.jsonl",
    "carried_register_format_records.jsonl", "daytime_register_format_records.jsonl",
    "character_register_format_records.jsonl", "video_making_records.jsonl",
)
_ZH = "零一二三四五六七八九十"


def _zh(n):
    if n < 10:
        return _ZH[n]
    tens, ones = divmod(n, 10)
    head = "十" if tens == 1 else _ZH[tens] + "十"
    return head + (_ZH[ones] if ones else "")


def _fallback_domain_texts(tok):
    """training.dataset 不可用时的最小域文本加载（live 语料，只读）。"""
    dual_sft = Path.home() / "workspace" / "dual_sft"
    mirror = Path.home() / "workspace" / "omega-chat"
    corpus = dual_sft / "generalist_corpus"

    def read_text(entries):
        parts = []
        for entry in entries:
            entry = str(entry)
            files = sorted(glob.glob(entry, recursive=True)) if any(
                ch in entry for ch in "*?[") else [entry]
            for f in files:
                p = Path(f)
                if "__pycache__" in p.parts or "/archive/" in str(p) or "/legacy/" in str(p):
                    continue
                parts.append(p.read_text(encoding="utf-8"))
        return "".join(parts)

    def read_world(entries):
        out = []
        for entry in entries:
            with open(entry, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line)["text"])
        return "\n".join(out)

    rng = np.random.default_rng(7)
    lines = []
    for _ in range(400):
        a, b = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        lines.append(f"问：{_zh(a)}加{_zh(b)}等于几？答案是{_zh(a + b)}。")
    for _ in range(200):
        a, b = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        lines.append(f"问：{_zh(a)}乘{_zh(b)}等于几？答案是{_zh(a * b)}。")
    reasoning = "\n".join(lines) + "\n"

    texts = {
        "经典": read_text([mirror / "Xiyouji.txt", corpus / "classics_rulin.txt"]),
        "代码": read_text([str(mirror / "**" / "*.py")]),
        "对话": read_text([corpus / "dialogue_pilot.txt"]),
        "推理": reasoning,
        "常识": read_text([corpus / "knowledge_wiki.txt", corpus / "knowledge_youxue.txt"]),
        "世界": read_world([dual_sft / name for name in _FALLBACK_WORLD_FILES]),
    }
    return {name: np.asarray(tok.encode(text), dtype=np.int64)
            for name, text in texts.items()}


def _load_domain_texts(tok):
    try:
        from training.dataset import domain_texts, live_corpus_paths
        return domain_texts(tok, paths=live_corpus_paths())
    except Exception as e:
        print(f"WARN: training.dataset 不可用（{e}），用最小回退加载 live 语料。",
              flush=True)
        return _fallback_domain_texts(tok)


def state_at(loc: str) -> WorldState:
    st = WorldState()
    if loc != st.visitor_location:
        st.move(loc)
    return st


@torch.no_grad()
def _gate_of_ids(model, ids, device) -> float:
    arr = torch.as_tensor(np.asarray(ids, dtype=np.int64),
                          device=device).unsqueeze(0)
    g = model.gate_values(arr)
    return float(g.reshape(-1)[0].item())


def probe_domains(model, tok, device, seq=SEQ, n_win=N_WIN, batch=BATCH):
    """A 部分：每域 N_WIN 个窗口的 mean g。返回 (results, left_avg, world_g)。"""
    ids = _load_domain_texts(tok)
    results = {}
    for name, arr in ids.items():
        arr = np.asarray(arr)
        starts = np.linspace(0, len(arr) - seq - 1, n_win).astype(int)
        gs = []
        for i in range(0, n_win, batch):
            wins = torch.tensor(
                np.stack([arr[s:s + seq] for s in starts[i:i + batch]]),
                dtype=torch.long, device=device)
            g = model.gate_values(wins)  # (B, S, 1)，每序列内处处相同
            gs.append(g.detach().cpu().numpy())
        g = np.concatenate(gs, axis=0)
        mean_g = float(g.mean())
        results[name] = mean_g
        side = "LEFT " if name in LEFT_DOMAINS else "RIGHT"
        print(f"{side} {name}: mean g={mean_g:.4f} (std {float(g.std()):.4f}) "
              f"n_tokens={g.size}", flush=True)
    left_avg = float(np.mean([results[n] for n in LEFT_DOMAINS]))
    world_g = results["世界"]
    print(f"LEFT mean g avg={left_avg:.4f} | WORLD mean g={world_g:.4f} "
          f"| separation={left_avg - world_g:.4f}", flush=True)
    print(f"PASS CRITERIA (left > 0.6 and world < 0.4): "
          f"{'PASS' if (left_avg > 0.6 and world_g < 0.4) else 'FAIL'}",
          flush=True)
    print(f"TREND BAR (separation > 0.05): "
          f"{'PASS' if (left_avg - world_g) > 0.05 else 'FAIL'}", flush=True)
    return results, left_avg, world_g


def probe_loop_prompts(model, tok, device, seq=SEQ):
    """B 部分：真实循环提示的 g。返回 rows 列表。"""
    def gate_of_text(text, bos=True):
        ids = tok.encode(text, add_bos=bos)
        if tok.UNK in ids:
            raise ValueError(f"UNK in prompt: {text[:40]!r}")
        return _gate_of_ids(model, ids, device), len(ids)

    rows = []
    # A. probe prompts Q0..Q3 at step-0 state (frozen loop step 0)
    s0 = WorldState()
    for qi, q in enumerate(QUESTIONS):
        g, n = gate_of_text(world_prompt(s0, q))
        rows.append(("probe", f"S0-取经路 Q{qi} {q}", g, n))
    # B. Q0 at six other locations
    for loc in ("花果山", "水帘洞", "高老庄", "流沙河", "南海", "天宫"):
        st = state_at(loc)
        g, n = gate_of_text(world_prompt(st, QUESTIONS[0]))
        rows.append(("probe", f"Q0 @{loc}", g, n))
    # C. arrival narration prompts (causal probe scored these 5/5)
    for loc in ("花果山", "水帘洞"):
        st = state_at(loc)
        g, n = gate_of_text(anchor_prompt(st) + f"\n你来到了{loc}，说说这里。")
        rows.append(("arrival", f"到达叙述 @{loc}", g, n))
    # D. probe-format training records (summary+Q+answer as stored)
    rec_path = probe_records_path()
    if rec_path.exists():
        recs = [json.loads(line)["text"] for line in
                open(rec_path, encoding="utf-8")][:2]
        for i, r in enumerate(recs):
            g, n = gate_of_text(r)
            rows.append(("train-rec", f"探针格式训练记录 #{i}（{r[60:80]}…）", g, n))
    else:
        print(f"WARN: 探针格式记录不存在，跳过 D 部分：{rec_path}", flush=True)
    # E. corpus-window controls, lean-probe sampling
    ids = _load_domain_texts(tok)
    for name in ("世界", "经典"):
        arr = np.asarray(ids[name])
        starts = np.linspace(0, len(arr) - seq - 1, 4).astype(int)
        for j in starts[:2]:
            g = _gate_of_ids(model, arr[int(j):int(j) + seq], device)
            rows.append(("control", f"{name} 语料窗 @{int(j)}", g, seq))

    print(f"\n{'kind':9} {'prompt':28} {'g(left)':>8} {'ntok':>5}")
    for kind, label, g, n in rows:
        print(f"{kind:9} {label:28} {g:8.4f} {n:5d}", flush=True)
    probe_gs = [g for k, _, g, _ in rows if k == "probe"]
    arr_gs = [g for k, _, g, _ in rows if k == "arrival"]
    print(f"\nprobe prompts mean g={float(np.mean(probe_gs)):.4f} (n={len(probe_gs)})",
          flush=True)
    print(f"arrival prompts mean g={float(np.mean(arr_gs)):.4f} (n={len(arr_gs)})",
          flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="门控分流探针：域窗口 + 循环提示")
    ap.add_argument("--checkpoint", required=True, help="已转换的 .pt 检查点路径")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--n-win", type=int, default=N_WIN)
    ap.add_argument("--skip-domains", action="store_true", help="跳过 A 部分（域窗口）")
    ap.add_argument("--skip-prompts", action="store_true", help="跳过 B 部分（循环提示）")
    args = ap.parse_args()

    t0 = time.time()
    try:
        model, tok, step, config, extra = load_model_and_tokenizer(
            args.checkpoint, args.device)
    except Exception as e:
        # 无检查点 / 检查点损坏 / model 包未就绪：报清楚的错，不抛堆栈。
        print(f"ERROR: 无法加载检查点：{e}", file=sys.stderr)
        sys.exit(2)
    if config.get("merge_method") != "gated":
        print(f"ERROR: 门控探针需要 merge_method='gated'，"
              f"检查点是 {config.get('merge_method')!r}", file=sys.stderr)
        sys.exit(2)
    print(f"checkpoint step={step} recipe={extra.get('recipe')} "
          f"merge={config.get('merge_method')} d={config.get('d_model')} "
          f"n_stem_blocks={config.get('n_stem_blocks')}", flush=True)

    if not args.skip_domains:
        probe_domains(model, tok, args.device, seq=args.seq, n_win=args.n_win)
    if not args.skip_prompts:
        probe_loop_prompts(model, tok, args.device, seq=args.seq)
    print(f"step={step} last_train_loss={extra.get('last_train_loss')}", flush=True)
    print(f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
