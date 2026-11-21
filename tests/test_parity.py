"""PyTorch vs NumPy 对数一致性（parity）测试。

把真实检查点 `~/workspace/dual_sft/out/generalist3_d1024_s600.npz`
经 scripts/convert_npz_to_pt.py 转成 .pt（默认放 /tmp，不进 staging），
分别装进 PyTorch 模型（CONTRACT.md 的 model.transformer.DualTransformer）
和 NumPy 单体模型（DualTransformerNP），在同一输入上比 logits。

状态：PENDING INTEGRATION——model 包（Agent A）未就绪，且 torch
尚未装进 .venv 时本测试 skip。协调员运行步骤：
    1. .venv 装好 torch 后：
       .venv/bin/python scripts/convert_npz_to_pt.py \
           ~/workspace/dual_sft/out/generalist3_d1024_s600.npz \
           /tmp/generalist3_d1024_s600.pt --verify
    2. .venv/bin/python -m pytest tests/test_parity.py -s
    3. 看打印的 max abs diff（期望量级 1e-5~1e-4，float32 同数学不同求和顺序）。

注意：NumPy 的 forward() 是确定性的（dropout 只在训练前向里，默认 0），
PyTorch 侧用 model.eval() 关 dropout，两边才可比。
另注意：s600.npz 的 meta.step 是当块步数 300（见转换器 docstring），
文件名里的 600 才是累计步数——不影响本测试。
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 未安装")
DualTransformer = pytest.importorskip(
    "model.transformer", reason="model 包未就绪（Agent A）").DualTransformer
load_checkpoint = pytest.importorskip(
    "model.checkpoint", reason="model/checkpoint.py 未就绪（Agent A）").load_checkpoint

MONOLITH = Path.home() / "workspace" / "omega-chat" / "dual_transformer_numpy.py"
NPZ = Path.home() / "workspace" / "dual_sft" / "out" / "generalist3_d1024_s600.npz"
PT = Path(os.environ.get("PT_CHECKPOINT", "/tmp/generalist3_d1024_s600.pt"))

# float32 两边求和顺序不同，1e-3 是宽松通过线；真实差异应远小于它，
# 报告里以打印的实际数值为准。
MAX_ABS_DIFF = 1e-3


def _load_numpy_model():
    spec = importlib.util.spec_from_file_location("dual_transformer_numpy", MONOLITH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with np.load(NPZ, allow_pickle=False) as data:
        params = {k[2:]: np.array(data[k]) for k in data.files if k.startswith("p_")}
        cfg = dict(__import__("json").loads(bytes(data["meta_json"]).decode("utf-8"))["config"])
    model = mod.DualTransformerNP(**cfg)
    model.params = params
    return model, cfg


def test_logits_parity_s600():
    if not PT.exists():
        pytest.skip(f"转换后的 .pt 不存在：{PT}（先跑转换器，见本文件头注释）")
    if not NPZ.exists():
        pytest.skip(f"源 .npz 不存在：{NPZ}")
    ckpt = load_checkpoint(str(PT))
    cfg = {k: v for k, v in ckpt["config"].items()
           if k in ("d_model", "n_blocks", "n_heads", "n_kv_heads", "merge_method",
                    "vocab_size", "max_seq", "tie_embeddings", "use_rms",
                    "pre_norm", "pos_mode", "ff_mode", "init_style",
                    "n_stem_blocks", "final_norm")}
    pt_model = DualTransformer(**cfg)
    pt_model.load_state_dict(ckpt["model"])
    pt_model.eval()

    np_model, np_cfg = _load_numpy_model()
    assert np_cfg == ckpt["config"], "两边 config 不一致，转换或加载有问题"

    rng = np.random.default_rng(7)
    tokens = rng.integers(0, cfg["vocab_size"], size=(2, 32)).astype(np.int64)
    # 再加一段真实文本开头（西游记），覆盖高频中文 token
    from tokenizer import CharTokenizer
    import json
    vocab = json.loads((Path(__file__).resolve().parent.parent
                        / "data" / "processed" / "generalist_vocab.json").read_text())
    tok = CharTokenizer(vocab["chars"])
    text_ids = np.asarray([tok.encode("孙悟空来到花果山。")], dtype=np.int64)

    with torch.no_grad():
        pt_logits = pt_model(torch.from_numpy(tokens)).numpy()
        pt_logits_text = pt_model(torch.from_numpy(text_ids)).numpy()
    np_logits = np_model.forward(tokens)
    np_logits_text = np_model.forward(text_ids)

    diff = float(np.max(np.abs(pt_logits - np_logits)))
    diff_text = float(np.max(np.abs(pt_logits_text - np_logits_text)))
    print(f"\nparity s600: max abs diff (random ids) = {diff:.3e}, "
          f"(西游记文本) = {diff_text:.3e}")
    assert diff < MAX_ABS_DIFF, f"logits 差异 {diff:.3e} 超过 {MAX_ABS_DIFF}"
    assert diff_text < MAX_ABS_DIFF
