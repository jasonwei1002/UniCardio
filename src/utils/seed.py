"""实验可复现性随机种子，遵循 ~/.claude/rules/experiment-reproducibility.md。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """同时为 Python、NumPy、PyTorch 设置随机种子。

    Args:
        seed: 应用于所有 RNG 的整数种子。
        deterministic: 为 True 时强制 cuDNN 使用确定性算法。默认关闭以保证
            训练速度；复现性实验时可通过配置开启。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 初始化函数，为每个 worker 重设 NumPy/random 种子。"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
