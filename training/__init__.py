"""训练包（PyTorch 版）：数据集、分层采样、损失、优化器、训练循环、DDP。"""
from training.dataset import (GATE_LEFT_DOMAINS, GATE_RIGHT_DOMAINS, RATIOS,
                              GeneralistDataset, StratifiedBatchSampler,
                              default_paths, domain_texts, get_dataloader,
                              live_corpus_paths, make_batch_fn, reasoning_text)
from training.distributed import maybe_wrap_ddp
from training.loss import cross_entropy_loss, gate_bce_loss, masked_loss
from training.optimizer import build_optimizer, clip_grad_norm, lr_at
from training.trainer import train

__all__ = [
    "GeneralistDataset", "StratifiedBatchSampler", "get_dataloader",
    "make_batch_fn", "domain_texts", "reasoning_text", "RATIOS",
    "GATE_LEFT_DOMAINS", "GATE_RIGHT_DOMAINS", "default_paths",
    "live_corpus_paths", "cross_entropy_loss", "gate_bce_loss", "masked_loss",
    "build_optimizer", "clip_grad_norm", "lr_at", "train", "maybe_wrap_ddp",
]
