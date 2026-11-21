"""配置测试：3 份 YAML 能解析，且值与真实运行一致。

配置是扁平 key: value（见 configs/*.yaml 头注释）。优先用 PyYAML，
缺 PyYAML 时退回 scripts/train.py 的内置简易读取器。
不依赖 torch，可独立运行。
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return dict(yaml.safe_load(text) or {})
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_cli", ROOT / "scripts" / "train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_config(path)


def test_gated_config_matches_real_run():
    # 来源：GATED_V5_S300.log（d=1024 blocks=4 heads=16 kv=8 gated）
    cfg = _load(ROOT / "configs" / "model_gated.yaml")
    assert cfg["recipe"] == "generalist-v5"
    assert cfg["merge_method"] == "gated"
    assert cfg["d_model"] == 1024
    assert cfg["n_blocks"] == 4
    assert cfg["n_heads"] == 16
    assert cfg["n_kv_heads"] == 8
    assert cfg["n_stem_blocks"] == 1
    assert cfg["vocab_size"] == 4622
    assert cfg["tie_embeddings"] is True
    assert cfg["dropout"] == pytest.approx(0.1)
    assert cfg["label_smoothing"] == pytest.approx(0.1)
    assert cfg["params"] == 85542235
    assert cfg["gate_loss_weight"] == pytest.approx(1.0)
    # 混合配比 v2：0.25 / 0.20 / 0.10 / 0.15 / 0.15 / 0.15
    ratios = [cfg[f"ratio_{k}"] for k in
              ("classics", "code", "dialogue", "knowledge", "reasoning", "world")]
    assert ratios == pytest.approx([0.25, 0.20, 0.10, 0.15, 0.15, 0.15])
    assert sum(ratios) == pytest.approx(1.0)
    assert cfg["gate_left_domains"] == "classics,code,dialogue,knowledge,reasoning"
    assert cfg["gate_right_domains"] == "world"


def test_small_config_matches_real_run():
    # 来源：generalist_pilot_train.py 默认值（d512 single tied，13,916,328 参数）
    cfg = _load(ROOT / "configs" / "model_small.yaml")
    assert cfg["recipe"] == "generalist-v2"
    assert cfg["merge_method"] == "single"
    assert cfg["d_model"] == 512
    assert cfg["n_blocks"] == 4
    assert cfg["n_heads"] == 8
    assert cfg["n_kv_heads"] == 4
    assert cfg["vocab_size"] == 4622
    assert cfg["params"] == 13916328
    assert cfg["batch_size"] == 8
    assert cfg["seq_len"] == 128


def test_train_default_config():
    cfg = _load(ROOT / "configs" / "train_default.yaml")
    assert cfg["recipe"] == "generalist-v5"
    assert cfg["steps"] == 300
    assert cfg["save_every"] == 10
    assert cfg["batch_size"] == 8
    assert cfg["seq_len"] == 128
    assert cfg["lr"] == pytest.approx(0.001)
    assert cfg["schedule"] == "cosine"
    assert cfg["min_lr_ratio"] == pytest.approx(0.1)
    assert cfg["weight_decay"] == pytest.approx(0.01)
    assert cfg["max_grad_norm"] == pytest.approx(1.0)
    assert cfg["seed"] == 7
    assert cfg["model_config"] == "configs/model_gated.yaml"
