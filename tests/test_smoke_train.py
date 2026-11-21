"""3 步训练冒烟测试：微型模型在 CPU 上跑 3 步，损失有限、参数真的变了。

用 model + training.loss + training.optimizer 的公开 API 手写 3 步循环
（不依赖 training.trainer.train 的未定签名；trainer 级冒烟待 Agent B
的签名确定后补）。状态：PENDING INTEGRATION——model/loss/optimizer
未就绪时 skip。
"""
import pytest

torch = pytest.importorskip("torch", reason="torch 未安装")
DualTransformer = pytest.importorskip(
    "model.transformer", reason="model 包未就绪（Agent A）").DualTransformer
cross_entropy_loss = pytest.importorskip(
    "training.loss", reason="training/loss.py 未就绪（Agent B）").cross_entropy_loss
build_optimizer = pytest.importorskip(
    "training.optimizer", reason="training/optimizer.py 未就绪（Agent B）").build_optimizer


def test_three_step_smoke_cpu():
    torch.manual_seed(7)
    model = DualTransformer(
        d_model=16, n_blocks=2, n_heads=2, n_kv_heads=1,
        merge_method="gated", vocab_size=64, max_seq=32,
        tie_embeddings=True, dropout=0.0, use_rms=True, pre_norm=True,
        pos_mode="rope", ff_mode="swiglu", init_style="gpt2",
        n_stem_blocks=1, final_norm=True, dtype=torch.float32)
    model.train()
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = []
    for _ in range(3):
        tokens = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        logits = model(tokens)
        loss = cross_entropy_loss(logits, targets, label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        assert torch.isfinite(loss), "损失非有限"
        losses.append(float(loss))
    changed = [n for n, p in model.named_parameters()
               if not torch.equal(p.detach(), before[n])]
    assert changed, "3 步后没有任何参数变化"
    print(f"\nsmoke train: losses={[f'{x:.4f}' for x in losses]}, "
          f"changed {len(changed)}/{len(before)} params")
