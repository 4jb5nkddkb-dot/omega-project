#!/usr/bin/env python3
"""NumPy .npz 检查点 -> PyTorch .pt 检查点（自包含转换器）。

只依赖 numpy + torch，**不 import 项目内任何模块**，所以即使
model/、training/ 还没就绪也能独立跑——训练权重（s5100 / s600 / s900）
就靠它活过框架切换。

NumPy 检查点格式（见单体 read_checkpoint / CONTRACT.md）：
    p_<name>   模型参数（float32）
    m_<name>   Adam 一阶矩 exp_avg
    v_<name>   Adam 二阶矩 exp_avg_sq
    meta_json  JSON：{format_version, config, dtype, step, has_optimizer,
                      optimizer_t, rng_state, extra{...}}
    注意：meta 里的 step 是**当块步数**（每块 300 步），不是文件名里的
    累计步数——generalist3_d1024_s600.npz 的 meta.step 是 300，
    累计步数只在文件名里。extra.schedule_total_steps 也是当块总数。

转换映射（1:1，参数名不变）：
    p_emb            -> model["emb"]
    p_stem.0.…       -> model["stem.0.…"]   （与 CONTRACT.md 的 state_dict 键完全一致）
    m_<name>/v_<name> -> optimizer["m"][<name>] / optimizer["v"][<name>]
    meta_json        -> config / step / extra / rng_state
    meta.optimizer_t -> optimizer["t"]

输出 .pt dict（CONTRACT.md 的 model/checkpoint.save_checkpoint 同款）：
    {"model": {name: tensor}, "optimizer": {"m":..., "v":..., "t": int} | None,
     "step": int, "config": dict, "extra": dict, "rng_state": ...}

用法：
    python scripts/convert_npz_to_pt.py <in.npz> <out.pt> [--verify] [--no-optimizer]

--verify：转完立刻用 torch.load 读回，核对参数个数、形状、数值
（逐参数 allclose + 全量 sha256 校验和）并打印报告。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def read_npz(path):
    """只读 .npz，返回 dict（numpy 数组，不依赖 torch）。

    返回：{"model": {name: np.ndarray}, "optimizer": {"m":..., "v":..., "t": int} | None,
           "step": int, "config": dict, "extra": dict, "rng_state": ...}
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        files = list(data.files)
        if "meta_json" not in files:
            raise ValueError(f"{path}: 缺 meta_json，不是本项目格式的检查点")
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version={meta.get('format_version')}，"
                f"本转换器只支持 {FORMAT_VERSION}")
        model = {k[2:]: np.array(data[k]) for k in files if k.startswith("p_")}
        m = {k[2:]: np.array(data[k]) for k in files if k.startswith("m_")}
        v = {k[2:]: np.array(data[k]) for k in files if k.startswith("v_")}
    if not model:
        raise ValueError(f"{path}: 没有找到 p_* 参数")
    if (m and not v) or (v and not m):
        raise ValueError(f"{path}: 优化器状态不完整（m={len(m)}，v={len(v)}）")
    if m and set(m) != set(model):
        raise ValueError(f"{path}: m_* 键与 p_* 键不一致")
    if v and set(v) != set(model):
        raise ValueError(f"{path}: v_* 键与 p_* 键不一致")
    optimizer = None
    if m:
        optimizer = {"m": m, "v": v, "t": int(meta.get("optimizer_t", meta.get("step", 0)))}
    return {
        "model": model,
        "optimizer": optimizer,
        "step": int(meta["step"]),
        "config": dict(meta["config"]),
        "extra": dict(meta.get("extra") or {}),
        "rng_state": meta.get("rng_state"),
    }


def to_pt_dict(parsed):
    """numpy 解析结果 -> .pt dict（这里才需要 torch）。"""
    import torch

    model = {name: torch.from_numpy(arr) for name, arr in parsed["model"].items()}
    optimizer = None
    if parsed["optimizer"] is not None:
        optimizer = {
            "m": {k: torch.from_numpy(a) for k, a in parsed["optimizer"]["m"].items()},
            "v": {k: torch.from_numpy(a) for k, a in parsed["optimizer"]["v"].items()},
            "t": int(parsed["optimizer"]["t"]),
        }
    return {
        "model": model,
        "optimizer": optimizer,
        "step": int(parsed["step"]),
        "config": dict(parsed["config"]),
        "extra": dict(parsed["extra"]),
        "rng_state": parsed["rng_state"],
    }


def param_count(model):
    """参数总数（dict of array/tensor -> int）。
    注意：numpy 数组的 .size 是 int 属性，torch 张量的 .size() 是方法，
    不能用 hasattr(a, "size") 区分——按类型分派。"""
    total = 0
    for a in model.values():
        if isinstance(a, np.ndarray):
            total += int(a.size)
        else:  # torch Tensor
            total += int(a.numel())
    return total


def checksum(model):
    """全量校验和：按参数名排序，拼字节流取 sha256。转换前后应一致。"""
    h = hashlib.sha256()
    for name in sorted(model):
        arr = model[name]
        if hasattr(arr, "numpy"):  # torch Tensor
            arr = arr.detach().cpu().numpy()
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def convert(in_path, out_path, include_optimizer=True):
    """读 .npz、写 .pt，返回 (pt_dict, report_dict)。"""
    import torch

    parsed = read_npz(in_path)
    if not include_optimizer:
        parsed["optimizer"] = None
    pt = to_pt_dict(parsed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 预估写出大小并检查目标文件系统余量：完整 .pt 约为参数字节数
    # （模型）+ 2 倍（Adam m/v），d1024 门控约 1.03 GB。/tmp 是 512 MB
    # tmpfs，装不下完整文件——写一半会在 close 时报 iostream error。
    # 完整转换请写到 home 磁盘（如 ~/workspace/convert-scratch/），
    # 或用 --no-optimizer 只转模型参数（约 342 MB，能进 /tmp）。
    nbytes = sum(int(a.nbytes) for a in parsed["model"].values())
    if include_optimizer and parsed["optimizer"] is not None:
        nbytes *= 3
    free = shutil.disk_usage(out_path.parent).free
    if nbytes > free:
        raise OSError(
            f"目标磁盘余量不足：{out_path.parent} 剩余 {free / 2**20:.0f} MB，"
            f"本次写出约需 {nbytes / 2**20:.0f} MB。"
            f"完整 .pt 请写到 home 磁盘，或加 --no-optimizer 只转模型参数。")
    torch.save(pt, out_path)
    report = {
        "params": len(pt["model"]),
        "param_count": param_count(pt["model"]),
        "checksum": checksum(pt["model"]),
        "has_optimizer": pt["optimizer"] is not None,
        "optimizer_t": pt["optimizer"]["t"] if pt["optimizer"] else None,
        "step": pt["step"],
        "config": pt["config"],
        "out_bytes": out_path.stat().st_size,
    }
    return pt, report


def verify(out_path, expected_checksum=None, expected_params=None):
    """读回 .pt 并核对：键、形状、数值（与源 .npz 逐参数比对由调用方做；
    这里做自洽检查 + 可选的校验和/参数个数核对）。"""
    import torch

    try:
        pt = torch.load(out_path, map_location="cpu", weights_only=True)
    except Exception:
        pt = torch.load(out_path, map_location="cpu", weights_only=False)
    model = pt["model"]
    ok = True
    problems = []
    if expected_params is not None and len(model) != expected_params:
        ok = False
        problems.append(f"参数个数 {len(model)} != 期望 {expected_params}")
    if expected_checksum is not None:
        got = checksum(model)
        if got != expected_checksum:
            ok = False
            problems.append(f"校验和 {got[:16]}… != 期望 {expected_checksum[:16]}…")
    for name, t in model.items():
        if not torch.isfinite(t.float()).all():
            ok = False
            problems.append(f"参数 {name} 含非有限值")
    return ok, problems, {
        "params": len(model),
        "param_count": param_count(model),
        "checksum": checksum(model),
        "has_optimizer": pt.get("optimizer") is not None,
        "step": pt.get("step"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="转换 NumPy .npz 检查点为 PyTorch .pt")
    ap.add_argument("input", help="输入 .npz 检查点")
    ap.add_argument("output", help="输出 .pt 文件")
    ap.add_argument("--verify", action="store_true",
                    help="转完读回并核对参数个数、形状与校验和")
    ap.add_argument("--no-optimizer", action="store_true",
                    help="不转换 Adam m/v（只转模型参数）")
    args = ap.parse_args(argv)

    parsed = read_npz(args.input)
    print(f"读入 {args.input}：{len(parsed['model'])} 个参数，"
          f"{param_count(parsed['model']):,} 个标量，"
          f"step={parsed['step']}（当块步数，见模块 docstring），"
          f"优化器={'有' if parsed['optimizer'] else '无'}")
    print(f"配置：d={parsed['config'].get('d_model')} "
          f"blocks={parsed['config'].get('n_blocks')} "
          f"heads={parsed['config'].get('n_heads')} "
          f"merge={parsed['config'].get('merge_method')} "
          f"tied={parsed['config'].get('tie_embeddings')}")
    src_checksum = checksum(parsed["model"])
    _, report = convert(args.input, args.output,
                        include_optimizer=not args.no_optimizer)
    print(f"写出 {args.output}（{report['out_bytes']:,} 字节），"
          f"校验和 {report['checksum'][:16]}…")
    if report["checksum"] != src_checksum:
        print("错误：转换前后校验和不一致", file=sys.stderr)
        return 1
    if args.verify:
        ok, problems, vrep = verify(args.output, expected_checksum=src_checksum,
                                    expected_params=len(parsed["model"]))
        print(f"验证：参数 {vrep['params']} 个，{vrep['param_count']:,} 标量，"
              f"step={vrep['step']}，优化器={'有' if vrep['has_optimizer'] else '无'}")
        if not ok:
            print("验证失败：", "; ".join(problems), file=sys.stderr)
            return 1
        print("验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
