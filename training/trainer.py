"""训练循环（PyTorch 版）。

train() 的签名（合同要求 Agent B 自定，见这里）::

    train(model, *, batch_fn=None, dataloader=None, steps=None,
          total_steps=None, batch_size=8, seq_len=128, lr=1e-3,
          schedule="cosine", warmup_steps=10, min_lr_ratio=0.1,
          weight_decay=0.01, max_grad_norm=1.0, label_smoothing=0.1,
          gate_loss_weight=1.0, seed=7, device=None, amp=False,
          log_every=None, save_every=10, checkpoint_path=None,
          resume_from=None, init_from=None, extra_meta=None,
          log_csv=None, grad_accum_steps=1, gated=None)

- steps：本块优化步数；None 且无 resume 时默认 100（与 NumPy 试点一致）。
- resume_from：_last.pt 断点。断点续跑时 steps 可省略，剩余步数按
  schedule_total_steps - start_step 自动算（与 NumPy 训练器同一规则：
  显式传 steps 会跑满给定步数，不按 schedule_total 封顶——续跑请省略）。
- init_from：只取参数热启动，优化器与学习率曲线全新（与 --init 一致，
  和 resume_from 互斥）。
- batch_fn / dataloader 二选一：batch_fn(rng) -> (inputs, targets)
  或 (inputs, targets, gate_targets)，或带答案区 loss 掩码的四元组
  (inputs, targets, gate_targets, loss_mask)——loss_mask 为真才计
  loss（先读后答：世界域只算答案区，见 training/dataset.py 的
  make_batch_fn(answer_masks=...)）；dataloader 每批同样返回二元组、
  三元组或四元组（见 training.dataset）。两者都不给时报 ValueError。
- device：None 时自动选 CUDA（若可用）否则 CPU；amp 只在 CUDA 上生效。
- 返回 dict：{"optimizer", "last_loss", "completed_step",
  "schedule_total", "start_step"}。

与 NumPy train_steps 的对应关系：
- 学习率：每步手动按 lr_at() 设 param group 的 lr（不用 LambdaLR，
  精确且断点续跑安全）；dropout 由 model.train() 控制，是建模型时
  的参数，不再是训练器的入参。
- 损失：交叉熵（label_smoothing=0.1）+ 门 BCE（只在 gated 时，
  权重 gate_loss_weight）；门 BCE 用 PRE-sigmoid logit 算
  （见 training.loss.gate_bce_loss），模型若提供 gate_logits() 就
  直接用，否则从 gate_values() 的概率反推 logit（带截断的后备路径，
  见 _gate_logit 的注释——建议 model 侧补一个 gate_logits()）。
- 梯度裁剪 max_grad_norm=1.0；log_every 默认 max(1, steps//10)，
  打印格式与 NumPy 版一致（step/train_loss/grad_norm/lr）。

断点恢复的内存教训（2026-11-21，NumPy 第 9–11 次被杀的根因）：
NumPy 续跑时 read_checkpoint 被调 3 次（flags 恢复＋trainer 恢复块＋
train_steps 恢复），三份约 1 GB 的参数＋优化器拷贝常驻，RSS 顶到
5.3 GB 撞 6 GB 上限 OOM。这里只 torch.load 一次：同一份 ck dict
里同时取出 model / optimizer / step / config / extra / rng_state，
恢复完立即 `del ck`，绝不再为读 meta 另调一次 load（需要 meta 时
读同一份 dict 里的字段）。
检查点格式见 CONTRACT.md（model/checkpoint.py 的 .pt 格式）：
{"model", "optimizer": {"m", "v", "t"} 或 None, "step", "config",
"extra", "rng_state"}。optimizer 的 m/v 是 NumPy 风格命名
（= torch AdamW 的 exp_avg / exp_avg_sq），恢复时直接赋到
optimizer.state 的对应槽位上（先建出 state 条目）。

RNG 状态：保存/恢复 torch RNG + Python random 的状态。batch_fn
内部的分层采样 rng 由 make_batch_fn 建时定种子（seed），与 NumPy
版一致——NumPy 版保存的也是 train_steps 的 rng（dropout 用），
batch_fn 自己的 rng 从不进检查点。
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from training.dataset import (RATIOS, domain_texts, live_corpus_paths,
                              make_batch_fn, resolve_paths)
from training.loss import cross_entropy_loss, gate_bce_loss, masked_loss
from training.optimizer import build_optimizer, clip_grad_norm, lr_at

__all__ = ["train", "main"]

# --- 检查点读写：优先用 model/checkpoint.py（合同规定的 canonical 位置）；
# Agent A 还没落地时用本地同格式实现顶上，行为完全一致。---
try:
    from model.checkpoint import (last_checkpoint_path, load_checkpoint,
                                  save_checkpoint)
except Exception:  # pragma: no cover - model/ 还在并行开发时
    def last_checkpoint_path(path):
        p = Path(path)
        return p.with_name(f"{p.stem}_last{p.suffix}")

    def save_checkpoint(model, optimizer, path, step, extra, rng_state=None):
        opt_state = None
        if optimizer is not None:
            opt_state = _optimizer_to_numpy_style(model, optimizer)
        payload = {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": opt_state,
            "step": int(step),
            "config": model.config() if hasattr(model, "config") else {},
            "extra": dict(extra or {}),
            "rng_state": rng_state,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    def load_checkpoint(path):
        return torch.load(path, map_location="cpu", weights_only=False)


def _optimizer_to_numpy_style(model, optimizer):
    """把 torch AdamW 的 state 转成检查点格式 {"m", "v", "t"}。

    m/v 按参数名组织（= NumPy 版 optimizer.m/v 的键），t 取各参数
    state["step"] 的最大值（正常训练里所有参数步数相同）。
    注意：必须按参数对象本身映射，不能按序号——AdamW 的 param
    group 顺序（ndim>=2 的 decay 组在前）与 named_parameters 的
    顺序不同，按序号映射会把 state 赋给错的参数（2026-11-21
    smoke 实测：exp_avg 形状对不上 grad 而报错）。
    """
    names_by_id = {id(p): n for n, p in model.named_parameters()}
    m, v, t = {}, {}, 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if "exp_avg" not in st:
                continue
            name = names_by_id[id(p)]
            m[name] = st["exp_avg"].detach().cpu().clone()
            v[name] = st["exp_avg_sq"].detach().cpu().clone()
            t = max(t, int(torch.as_tensor(st["step"]).item()))
    return {"m": m, "v": v, "t": int(t)}


def _restore_optimizer_state(model, optimizer, opt_state):
    """把检查点的 NumPy 风格 optimizer state 赋到 torch AdamW 上。

    先给每个参数建出 state 条目，再直接写 exp_avg / exp_avg_sq /
    step——不经过 load_state_dict，因为 .pt 的键是参数名（m/v），
    而 torch 原生 state_dict 的键是参数序号。torch 原生格式
    （含 "state" 键）也接受，直接走 load_state_dict。
    """
    if opt_state is None:
        return
    if "state" in opt_state and "param_groups" in opt_state:
        optimizer.load_state_dict(opt_state)
        return
    m, v, t = opt_state.get("m", {}), opt_state.get("v", {}), int(opt_state.get("t", 0))
    params = dict(model.named_parameters())
    for name, p in params.items():
        if name not in m:
            continue
        state = optimizer.state[p]
        state["step"] = torch.tensor(float(t))
        state["exp_avg"] = m[name].detach().to(device=p.device, dtype=p.dtype).clone()
        state["exp_avg_sq"] = v[name].detach().to(device=p.device, dtype=p.dtype).clone()


def _capture_rng_state():
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state):
    if not isinstance(rng_state, dict):
        return
    if "torch" in rng_state:
        try:
            torch.set_rng_state(torch.as_tensor(rng_state["torch"], dtype=torch.uint8))
        except Exception:
            pass
    if "python" in rng_state:
        try:
            random.setstate(rng_state["python"])
        except Exception:
            pass
    if torch.cuda.is_available() and "torch_cuda" in rng_state:
        try:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        except Exception:
            pass


def _config_matches(saved_cfg, model_cfg):
    """检查点 config 是否与模型 config 一致。

    只比较模型 config 里声明的键；检查点缺失的键按 NumPy
    train_steps 恢复时的老默认值补（n_kv_heads 默认 n_heads，
    pos_mode 默认 rope，ff_mode 默认 gelu，n_stem_blocks 默认 1）。
    这样老检查点（缺新键）能续跑，而模型 config 里没有的键
    不会被硬补出来导致永远 mismatch。
    """
    legacy_defaults = {
        "n_kv_heads": saved_cfg.get("n_heads", model_cfg.get("n_heads")),
        "pos_mode": "rope",
        "ff_mode": "gelu",
        "n_stem_blocks": 1,
    }
    for key, want in model_cfg.items():
        got = saved_cfg.get(key, legacy_defaults.get(key, ...))
        if got is ... or got != want:
            return False, key, got, want
    return True, None, None, None


def _gate_logit(model, tokens):
    """取门的 PRE-sigmoid logit (B, S)，供 gate_bce_loss 用。

    首选 model.gate_logits(tokens)（若 Agent A 提供）；否则用
    gate_values() 的概率反推 logit，并截断到 [1e-7, 1-1e-7]——这是
    后备路径：概率饱和时截断会丢一点梯度信息，但不会 NaN。
    建议 model 侧补 gate_logits()，让训练走稳定的 BCEWithLogits 路径。
    """
    if hasattr(model, "gate_logits"):
        z = model.gate_logits(tokens)
        return z.squeeze(-1) if z.dim() == 3 else z
    if hasattr(model, "gate_values"):
        g = model.gate_values(tokens)
        g = g.squeeze(-1) if g.dim() == 3 else g
        return torch.logit(g.clamp(1e-7, 1.0 - 1e-7))
    return None


def train(model, *, batch_fn=None, dataloader=None, steps=None,
          total_steps=None, batch_size=8, seq_len=128, lr=1e-3,
          schedule="cosine", warmup_steps=10, min_lr_ratio=0.1,
          weight_decay=0.01, max_grad_norm=1.0, label_smoothing=0.1,
          gate_loss_weight=1.0, seed=7, device=None, amp=False,
          log_every=None, save_every=10, checkpoint_path=None,
          resume_from=None, init_from=None, extra_meta=None,
          log_csv=None, grad_accum_steps=1, gated=None):
    if batch_fn is None and dataloader is None:
        raise ValueError("provide batch_fn or dataloader")
    if batch_fn is not None and dataloader is not None:
        raise ValueError("batch_fn and dataloader are mutually exclusive")
    if init_from is not None and resume_from is not None:
        raise SystemExit("--init 与 --resume 互斥：新块热启动用 --init，"
                         "同块断点续跑用 --resume")
    if schedule not in ("constant", "cosine"):
        raise ValueError("schedule must be 'constant' or 'cosine'")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp) and device.type == "cuda"
    if amp and device.type != "cuda":
        print("amp requested but device is CPU: AMP off (torch.amp 只在 CUDA 上开)")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    model.to(device)

    gated = (getattr(model, "merge_method", "gated") == "gated") if gated is None else bool(gated)
    if not gated:
        gate_loss_weight = 0.0

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    start_step = 0
    schedule_total = None
    last_loss = math.inf

    # --- 热启动 / 断点续跑：检查点只 load 一次，恢复完立即 del ---
    if init_from is not None:
        ck = load_checkpoint(init_from)
        if hasattr(model, "config"):
            ok, key, got, want = _config_matches(ck.get("config", {}), model.config())
            if not ok:
                raise ValueError(f"init config mismatch on {key}: checkpoint has {got}, "
                                 f"model has {want}")
        model.load_state_dict(ck["model"])
        print(f"warm start from {init_from} (step {ck.get('step', '?')}), "
              "fresh optimizer + fresh cosine schedule")
        del ck
    if resume_from is not None:
        ck = load_checkpoint(resume_from)  # 本函数内唯一一次 torch.load
        if hasattr(model, "config"):
            ok, key, got, want = _config_matches(ck.get("config", {}), model.config())
            if not ok:
                raise ValueError(f"resume config mismatch on {key}: checkpoint has {got}, "
                                 f"model has {want}")
        if ck.get("optimizer") is None:
            raise ValueError("断点文件没有优化器状态，无法续跑")
        model.load_state_dict(ck["model"])
        _restore_optimizer_state(model, optimizer, ck["optimizer"])
        start_step = int(ck["step"])
        extra = ck.get("extra") or {}
        last_loss = float(extra.get("last_train_loss", math.inf))
        schedule_total = int(extra.get("schedule_total_steps", start_step))
        _restore_rng_state(ck.get("rng_state"))
        if steps is None:
            steps = schedule_total - start_step
            print(f"resume from {resume_from} (step {start_step}/{schedule_total}), "
                  f"remaining steps = {steps}")
        print(f"resume keeps optimizer + cosine schedule (total {schedule_total})")
        del ck  # 参数/优化器已拷进模型与优化器，立即释放（约 1 GB）

    if steps is None:
        steps = 100
    if steps <= 0:
        raise SystemExit(f"步数为 {steps}，断点已跑满，无需续跑")
    end_step = start_step + steps
    if schedule_total is None:
        schedule_total = total_steps if total_steps is not None else end_step
    if log_every is None:
        log_every = max(1, steps // 10)
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    if save_every and checkpoint_path is None:
        print("save_every 已设但未给 checkpoint_path：不存断点")
        save_every = 0
    periodic_path = last_checkpoint_path(checkpoint_path) if (checkpoint_path and save_every) else None

    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    csv_file = csv_writer = None
    if log_csv:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not (resume_from and log_path.exists())
        csv_file = open(log_path, "w" if fresh else "a", newline="")
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        if fresh:
            csv_writer.writerow(["step", "train_loss", "val_loss", "grad_norm", "lr"])

    data_iter = None
    if dataloader is not None:
        def _cycle():
            while True:
                for batch in dataloader:
                    yield batch
        data_iter = _cycle()

    def _next_batch():
        if data_iter is not None:
            return next(data_iter)
        return batch_fn(rng)

    model.train()
    t0 = time.time()
    completed_step = start_step
    try:
        for step in range(start_step + 1, end_step + 1):
            lr_now = lr_at(step, lr, schedule, warmup_steps, schedule_total, min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(grad_accum_steps):
                batch = _next_batch()
                if len(batch) == 4:
                    x, y, gt, loss_mask = batch
                    gt = gt if gated else None
                elif len(batch) == 3:
                    x, y, gt = batch
                    gt = gt if gated else None
                    loss_mask = None
                else:
                    x, y, gt = batch[0], batch[1], None
                    loss_mask = None
                x = torch.as_tensor(x, dtype=torch.long, device=device)
                y = torch.as_tensor(y, dtype=torch.long, device=device)
                if gt is not None:
                    gt = torch.as_tensor(gt, dtype=torch.float32, device=device)
                if loss_mask is not None:
                    loss_mask = torch.as_tensor(loss_mask, dtype=torch.bool,
                                                 device=device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    if loss_mask is not None:
                        # 答案区 loss：只算掩码内的答案 token（先读后答），
                        # 其余位置不计 loss；门 BCE 不受掩码影响。
                        loss = masked_loss(logits, y, loss_mask,
                                           label_smoothing=label_smoothing)
                    else:
                        loss = cross_entropy_loss(logits, y, label_smoothing=label_smoothing)
                    if gt is not None and gate_loss_weight:
                        z_gate = _gate_logit(model, x)
                        if z_gate is not None:
                            loss = loss + gate_bce_loss(z_gate, gt, weight=gate_loss_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                if scaler is None:
                    (loss / grad_accum_steps).backward()
                else:
                    scaler.scale(loss / grad_accum_steps).backward()
                loss_acc += float(loss.detach()) / grad_accum_steps
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            completed_step, last_loss = step, float(loss_acc)
            if periodic_path is not None and step % save_every == 0 and step != end_step:
                extra = {"last_train_loss": float(last_loss),
                         "schedule_total_steps": schedule_total,
                         **(extra_meta or {})}
                save_checkpoint(model, optimizer, periodic_path, step, extra,
                                _capture_rng_state())
                print(f"periodic checkpoint: step={step} -> {periodic_path}")
            if step == 1 or step % log_every == 0:
                print(f"step={step:5d} train_loss={last_loss:.4f} "
                      f"grad_norm={grad_norm:.4f} lr={lr_now:.2e}")
                if csv_writer:
                    csv_writer.writerow([step, f"{last_loss:.6f}", "",
                                         f"{grad_norm:.6f}", f"{lr_now:.8g}"])
            if csv_file:
                csv_file.flush()
    finally:
        if csv_file:
            csv_file.close()

    if checkpoint_path:
        extra = {"last_train_loss": float(last_loss),
                 "schedule_total_steps": schedule_total,
                 **(extra_meta or {})}
        save_checkpoint(model, optimizer, checkpoint_path, completed_step, extra,
                        _capture_rng_state())
    dt = time.time() - t0
    print(f"done: {steps} steps in {dt:.1f}s ({dt / max(steps, 1):.2f} s/step) "
          f"-> {checkpoint_path}")
    return {"optimizer": optimizer, "last_loss": last_loss,
            "completed_step": completed_step, "schedule_total": schedule_total,
            "start_step": start_step}


# ----------------------------------------------------------------------------
# CLI：与 generalist_pilot_train.py 的调用约定对齐
# ----------------------------------------------------------------------------

def _load_vocab_meta():
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "processed" / "generalist_vocab.json",
        Path.home() / "workspace" / "dual_sft" / "generalist_vocab.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    raise FileNotFoundError(
        "generalist_vocab.json not found in data/processed/ or ~/workspace/dual_sft/")


def _read_flat_config(path):
    """读扁平 key: value 配置（configs/*.yaml 的格式，无嵌套/锚点）。"""
    cfg = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip('"').strip("'")
        if value in ("true", "false"):
            cfg[key.strip()] = value == "true"
        else:
            try:
                cfg[key.strip()] = float(value) if "." in value else int(value)
            except ValueError:
                cfg[key.strip()] = value
    return cfg


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="PyTorch 通用模型训练器（generalist-v5 门控）")
    ap.add_argument("--config", default=None, help="扁平 yaml 配置（默认 configs/train_pytorch.yaml 的值作后备）")
    ap.add_argument("--steps", type=int, default=None,
                    help="本块步数；不给则默认 100，断点续跑时默认跑满原块剩余步数")
    ap.add_argument("--out", default="checkpoints/generalist3_d1024_s300.pt")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", default=None, help="断点续跑：_last.pt 文件路径")
    ap.add_argument("--init", default=None, help="上一块产品路径：只取参数热启动")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--merge", choices=("single", "average", "concat", "gated"), default="gated")
    ap.add_argument("--recipe", default="generalist-v5")
    ap.add_argument("--gate-loss-weight", type=float, default=1.0)
    ap.add_argument("--tie", action="store_true", default=True)
    ap.add_argument("--no-tie", dest="tie", action="store_false")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--device", default=None, help="cpu/cuda，默认自动")
    ap.add_argument("--amp", action="store_true", help="CUDA 上开 torch.amp")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--live-corpus", action="store_true",
                    help="强制用 live 语料（~/workspace/dual_sft + ~/workspace/omega-chat）")
    args = ap.parse_args(argv)
    if args.init and args.resume:
        raise SystemExit("--init 与 --resume 互斥：新块热启动用 --init，"
                         "同块断点续跑用 --resume")

    from tokenizer import CharTokenizer
    from model.transformer import DualTransformer

    meta, vocab_path = _load_vocab_meta()
    assert meta["type"] == "char"
    tok = CharTokenizer(meta["chars"])
    print(f"vocab: {tok.vocab_size} (BOS/EOS/UNK 0/1/2) from {vocab_path}")
    paths = live_corpus_paths() if args.live_corpus else None
    ids, answer_masks = domain_texts(tok, paths=paths, return_answer_masks=True)

    model = DualTransformer(
        d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, merge_method=args.merge,
        vocab_size=tok.vocab_size, tie_embeddings=args.tie,
        dropout=args.dropout)
    n_params = sum(int(v.numel()) for v in model.parameters())
    print(f"model: d={args.d_model} blocks={args.n_blocks} heads={args.n_heads} "
          f"kv={args.n_kv_heads} merge={args.merge} tied={args.tie} "
          f"dropout={args.dropout} label_smoothing={args.label_smoothing} params={n_params:,}")
    print(f"train: seq_len={args.seq} batch_size={args.batch}")

    batch_fn = make_batch_fn(ids, args.seq, batch_size=args.batch, seed=args.seed,
                             answer_masks=answer_masks)
    train(
        model, batch_fn=batch_fn, steps=args.steps, batch_size=args.batch,
        seq_len=args.seq, lr=args.lr, schedule="cosine", warmup_steps=10,
        seed=args.seed, device=args.device, amp=args.amp,
        label_smoothing=args.label_smoothing,
        gate_loss_weight=(args.gate_loss_weight if args.merge == "gated" else 0.0),
        checkpoint_path=args.out, resume_from=args.resume, init_from=args.init,
        save_every=args.save_every,
        extra_meta={"tokenizer": tok.to_meta(), "recipe": args.recipe,
                    "ratios": dict(RATIOS),
                    "seq_len": args.seq, "merge": args.merge, "tied": args.tie,
                    "dropout": args.dropout, "label_smoothing": args.label_smoothing,
                    "gate_loss_weight": (args.gate_loss_weight
                                         if args.merge == "gated" else 0.0),
                    "gate": ("left=经典/代码/对话/常识/推理, right=世界"
                             if args.merge == "gated" else None)})


if __name__ == "__main__":
    main()
