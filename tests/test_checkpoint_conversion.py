"""检查点转换测试：.npz -> .pt 往返。

转换器（scripts/convert_npz_to_pt.py）是自包含的（numpy + torch，
不 import 项目模块），所以本测试不依赖 model 包：
- read_npz 的映射逻辑（p_/m_/v_/meta_json）在无 torch 时也能测；
- 完整的 .pt 写出 + 读回比对只在 torch 可用时跑，缺 torch 则 skip。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CONVERTER = ROOT / "scripts" / "convert_npz_to_pt.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_npz_to_pt", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_npz(path):
    """与真实检查点同键格式的微型 .npz：3 个参数 + Adam m/v + meta。"""
    rng = np.random.default_rng(0)
    params = {
        "emb": rng.normal(size=(64, 8)).astype(np.float32),
        "gate.weight": rng.normal(size=(1, 8)).astype(np.float32),
        "gate.bias": rng.normal(size=(1,)).astype(np.float32),
    }
    meta = {
        "format_version": 1,
        "config": {"d_model": 8, "n_blocks": 1, "n_heads": 2, "merge_method": "gated",
                   "use_rms": True, "max_seq": 32, "vocab_size": 64,
                   "final_norm": True, "tie_embeddings": True, "scaled_init": True,
                   "n_kv_heads": 1, "pos_mode": "rope", "ff_mode": "swiglu",
                   "n_stem_blocks": 1},
        "dtype": "float32",
        "step": 7,
        "has_optimizer": True,
        "optimizer_t": 7,
        "rng_state": None,
        "extra": {"recipe": "test-fixture", "schedule_total_steps": 10},
    }
    arrays = {}
    for k, v in params.items():
        arrays[f"p_{k}"] = v
        arrays[f"m_{k}"] = (0.1 * v).astype(np.float32)
        arrays[f"v_{k}"] = (0.01 * np.abs(v)).astype(np.float32)
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)
    return params, meta


def test_read_npz_mapping_no_torch(tmp_path):
    """无 torch 也能测：p_/m_/v_ 去前缀、meta 解析、step/config/extra。"""
    conv = _load_converter()
    npz_path = tmp_path / "tiny.npz"
    params, meta = _tiny_npz(npz_path)
    parsed = conv.read_npz(npz_path)
    assert set(parsed["model"]) == set(params)
    for k, v in params.items():
        assert parsed["model"][k].shape == v.shape
        assert np.array_equal(parsed["model"][k], v)
    assert parsed["optimizer"] is not None
    assert parsed["optimizer"]["t"] == 7
    assert set(parsed["optimizer"]["m"]) == set(params)
    assert set(parsed["optimizer"]["v"]) == set(params)
    assert parsed["step"] == 7
    assert parsed["config"]["d_model"] == 8
    assert parsed["extra"]["recipe"] == "test-fixture"
    assert conv.param_count(parsed["model"]) == sum(v.size for v in params.values())


def test_read_npz_rejects_bad_format(tmp_path):
    conv = _load_converter()
    bad = tmp_path / "bad.npz"
    np.savez(bad, x=np.zeros(3))
    with pytest.raises(ValueError, match="meta_json"):
        conv.read_npz(bad)


def test_full_round_trip(tmp_path):
    """完整 .pt 写出 + 读回：键、形状、数值、优化器状态、step、config 全对。"""
    pytest.importorskip("torch", reason="torch 未安装：.pt 写出部分待装好后补测")
    import torch
    conv = _load_converter()
    npz_path = tmp_path / "tiny.npz"
    pt_path = tmp_path / "tiny.pt"
    params, meta = _tiny_npz(npz_path)
    _, report = conv.convert(npz_path, pt_path)
    assert report["params"] == 3
    assert report["param_count"] == sum(v.size for v in params.values())
    assert report["has_optimizer"] is True
    assert report["optimizer_t"] == 7
    assert report["step"] == 7
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert set(pt["model"]) == set(params)
    for k, v in params.items():
        assert pt["model"][k].shape == tuple(v.shape)
        assert torch.equal(pt["model"][k], torch.from_numpy(v))
        assert torch.equal(pt["optimizer"]["m"][k], torch.from_numpy(0.1 * v))
    assert pt["step"] == 7
    assert pt["config"]["merge_method"] == "gated"
    assert pt["extra"]["recipe"] == "test-fixture"
    ok, problems, _ = conv.verify(pt_path, expected_checksum=report["checksum"],
                                  expected_params=3)
    assert ok, problems
