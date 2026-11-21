"""分布式训练包装（torch DDP）。

当前配方（generalist-v5：d1024 门控，batch 8，seq 128）是单设备训练，
本模块只提供未来多卡用的包装，不改变现在的训练路径：

- maybe_wrap_ddp(model, ...) 在未初始化进程组时原样返回模型
  （单卡/CPU 训练零开销、零行为变化）；
- 只有调用方先调 torch.distributed.init_process_group（或传
  init_method/world_size/rank 由本模块 init）后，才包成
  torch.nn.parallel.DistributedDataParallel。

多卡用法（将来）::

    from training.distributed import maybe_wrap_ddp, setup_ddp
    setup_ddp()  # 按 torchrun 的环境变量 init 进程组
    model = maybe_wrap_ddp(model)
    # 训练循环不变；checkpoint 只在 rank 0 写（见 trainer 的调用方）。

注意：DDP 下每个进程各有一份 DataLoader，采样器种子应按 rank
偏移（StratifiedBatchSampler seed + rank），否则各卡吃到同一批。
本模块不替训练器做这件事——当前训练器是单进程，留待多卡接入时
在调用方处理。
"""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = ["maybe_wrap_ddp", "setup_ddp", "cleanup_ddp", "is_distributed",
           "get_rank", "get_world_size", "is_main_process"]


def is_distributed():
    """进程组是否已初始化。"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def setup_ddp(backend=None, init_method=None, world_size=None, rank=None):
    """初始化默认进程组（torchrun 方式：直接调 setup_ddp() 即可，
    backend 默认 nccl（有 CUDA 时）否则 gloo）。

    已初始化时直接返回，不重复 init。
    """
    if is_distributed():
        return
    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend, init_method=init_method,
                            world_size=world_size, rank=rank)


def cleanup_ddp():
    if is_distributed():
        dist.destroy_process_group()


def maybe_wrap_ddp(model, device_ids=None, output_device=None, **ddp_kwargs):
    """可能包一层 DDP：未初始化进程组时原样返回模型。

    device_ids 不给时：CUDA 上用当前设备单卡（[torch.cuda.current_device()]），
    CPU 上传 None。find_unused_parameters 默认 False——门控模型的前向
    每次都走全部参数（门与双支都参与），不需要 True；若将来有条件
    分支导致部分参数不用，再显式传 True。
    """
    if not is_distributed():
        return model
    if device_ids is None and torch.cuda.is_available():
        device_ids = [torch.cuda.current_device()]
    return DDP(model, device_ids=device_ids, output_device=output_device,
               **ddp_kwargs)


def ddp_seed_offset():
    """给采样器种子加的 rank 偏移（各卡吃不同批，见模块 docstring）。"""
    return get_rank()
