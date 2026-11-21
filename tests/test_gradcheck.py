"""梯度检查：微型门控模型的前向 + 门 BCE 损失。

torch.autograd.gradcheck（float64）验证 PyTorch 实现的反向与数值梯度一致，
覆盖前向 logits 与门控分支。模型 tiny：d=16，2 blocks，2 heads，vocab 64，
dropout=0（gradcheck 要求确定性）。

门 logit 的约定（Agent B 2026-11-21 要求、CONTRACT 补充）：
模型应提供 `gate_logits(tokens) -> (B, S, 1)`，返回 PRE-sigmoid 的门
logit z；`gate_values(tokens)` 仍是 sigmoid 后的 g。训练器用
F.binary_cross_entropy_with_logits 直接吃 z（稳定路径），不再对
概率 clamp 再反推 logit。本测试优先用 gate_logits（若模型提供），
否则退回 clamp 概率的等价形式——两者数学上同一损失。

注意：training.loss.gate_bce_loss 内部把输入转成 float32，本测试
不用它（float64 gradcheck 会被 float32 舍入破坏），而是直接调
F.binary_cross_entropy_with_logits（保持 float64）。

状态：PENDING INTEGRATION——model 包（Agent A）未就绪时 skip。
按 PAUSE 指令（2026-11-21 04:26，chunk 3 训练中）：本测试涉及
torch 模型加载，暂缓运行，等协调员通知训练结束后再跑。
"""
import pytest
import torch.nn.functional as F

torch = pytest.importorskip("torch", reason="torch 未安装")
DualTransformer = pytest.importorskip(
    "model.transformer", reason="model 包未就绪（Agent A）").DualTransformer
cross_entropy_loss = pytest.importorskip(
    "training.loss", reason="training/loss.py 未就绪（Agent B）").cross_entropy_loss


def _tiny_model():
    return DualTransformer(
        d_model=16, n_blocks=2, n_heads=2, n_kv_heads=1,
        merge_method="gated", vocab_size=64, max_seq=32,
        tie_embeddings=True, dropout=0.0, use_rms=True, pre_norm=True,
        pos_mode="rope", ff_mode="swiglu", init_style="gpt2",
        n_stem_blocks=1, final_norm=True, dtype=torch.float64)


def _functional_call(model, params, tokens):
    names = [n for n, _ in model.named_parameters()]
    state = dict(zip(names, params))
    return torch.func.functional_call(model, state, (tokens,))


class _LossWrapper(torch.nn.Module):
    """把模型包一层，使 forward 返回标量损失。

    functional_call 按 dotted 键（"model.<name>"）替换整棵树的参数，
    所以包一层后 gradcheck 才能覆盖门分支的参数；直接对
    model.gate_values(tokens) 调 gradcheck 会漏掉门参数（它用的
    是模型自带参数，不是待检的 ps）。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, targets, gate_targets):
        logits = self.model(tokens)
        ce = cross_entropy_loss(logits, targets, label_smoothing=0.0)
        if hasattr(self.model, "gate_logits"):
            # 稳定路径：PRE-sigmoid logit 直接进 BCEWithLogits。
            z = self.model.gate_logits(tokens)  # (B, S, 1)
            bce = F.binary_cross_entropy_with_logits(
                z.squeeze(-1), gate_targets)
        else:
            # 后备路径：sigmoid 概率 + clamp（数值上同一损失，仅作过渡）。
            g = self.model.gate_values(tokens).clamp(1e-12, 1 - 1e-12)
            t = gate_targets.unsqueeze(-1)
            bce = -(t * torch.log(g) + (1 - t) * torch.log(1 - g)).mean()
        return ce + bce


def test_gradcheck_forward():
    model = _tiny_model()
    tokens = torch.randint(0, 64, (2, 8))
    params = tuple(p.detach().clone().double().requires_grad_(True)
                   for p in model.parameters())
    assert torch.autograd.gradcheck(
        lambda *ps: _functional_call(model, ps, tokens),
        params, eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_loss_with_gate():
    model = _tiny_model()
    wrapper = _LossWrapper(model)
    tokens = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    gate_targets = torch.tensor([[1.0] * 8, [0.0] * 8], dtype=torch.float64)
    names = [n for n, _ in wrapper.named_parameters()]
    params = tuple(p.detach().clone().double().requires_grad_(True)
                   for p in wrapper.parameters())
    assert torch.autograd.gradcheck(
        lambda *ps: torch.func.functional_call(
            wrapper, dict(zip(names, ps)), (tokens, targets, gate_targets)),
        params, eps=1e-6, atol=1e-4, rtol=1e-3)
