"""训练用损失函数（PyTorch 版）。

与 NumPy 版（training/loss.py 旧版，底层在 model.transformer._cross_entropy）
同一数学定义，改由 autograd 求梯度：本模块只返回 loss 张量，不再手写
dlogits。

约定（与单体 dual_transformer_numpy.py 及 generalist-v2/v5 配方一致）：
- 标签平滑 label_smoothing = 0.1（训练时生效，推理不受影响）。
- 门 BCE 权重 gate_loss_weight = 1.0（只在 merge=gated 时生效）。
- 门的监督目标 t=1 走左支，t=0 走右支；左支吃 经典/代码/对话/常识/推理，
  右支吃 世界（见 training/dataset.py 的 GATE_LEFT_DOMAINS /
  GATE_RIGHT_DOMAINS）。

门 BCE 的数值稳定性说明：
- NumPy 版用 eps=1e-7 截断 sigmoid 后的 g，因为 float32 下 sigmoid
  饱和到恰好 1.0 时 log(0) -> NaN（v5 试点 2026-11-17 在 step 10
  就是这么死的）。
- 本模块的 gate_bce_loss 直接对 PRE-sigmoid 的 logit 用
  F.binary_cross_entropy_with_logits（log-sum-exp 形式），
  logit 再大（±30 甚至更大）也不会 NaN——不再需要 eps 截断。
  调用方必须传 logit（z_gate），不要先过 sigmoid 再传进来。
"""
import torch
import torch.nn.functional as F

__all__ = ["cross_entropy_loss", "gate_bce_loss", "masked_loss"]


def cross_entropy_loss(logits, targets, label_smoothing=0.1):
    """Mean next-token cross-entropy（返回 loss 张量，可反向）。

    logits: (B, S, V) 或 (N, V)；targets: (B, S) 或 (N,) long。
    label_smoothing=eps 时目标分布为真 token 上 (1 - eps) 加全词表
    均匀 eps / V——与 NumPy _cross_entropy 同一定义，F.cross_entropy
    的 label_smoothing 参数正是这个公式。训练配方用 0.1。
    """
    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets, dtype=torch.long)
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    return F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))


def gate_bce_loss(z_gate, targets, weight=1.0):
    """门控二元交叉熵（按 token 平均，返回 loss 张量）。

    z_gate: 门的 PRE-sigmoid logit，(B, S) 或 (B, S, 1)；
    targets: 同形状（去掉末维 1 后对齐），float，1 = 左支，0 = 右支。
    用 F.binary_cross_entropy_with_logits：logit 饱和时仍稳定，
    替代 NumPy 版的 eps=1e-7 截断（见模块 docstring）。
    """
    z = torch.as_tensor(z_gate, dtype=torch.float32)
    t = torch.as_tensor(targets, dtype=torch.float32)
    if z.dim() == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)
    if t.dim() == 3 and t.size(-1) == 1:
        t = t.squeeze(-1)
    if t.shape != z.shape:
        t = t.reshape(z.shape)
    return float(weight) * F.binary_cross_entropy_with_logits(z, t)


def masked_loss(logits, targets, mask, label_smoothing=0.0):
    """只在 mask 为真的位置上算平均交叉熵（mask 形状 (B, S) 布尔）。

    返回 loss 张量。mask 全假时报 ValueError（与 NumPy 版一致）。
    """
    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets, dtype=torch.long)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.shape != targets.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match targets shape {tuple(targets.shape)}")
    if not bool(mask.any()):
        raise ValueError("masked_loss: mask selects no tokens")
    flat_logits = logits.reshape(-1, logits.size(-1))[mask.reshape(-1)]
    flat_targets = targets.reshape(-1)[mask.reshape(-1)]
    return F.cross_entropy(flat_logits, flat_targets,
                           label_smoothing=float(label_smoothing))
