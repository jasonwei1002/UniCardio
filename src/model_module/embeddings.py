"""输入编码器与时间条件 embedding。


``FlowTimeEmbedding`` 在 Rectified Flow 中替换了原来的 ``DiffusionEmbedding``。
它将连续时间 ``t ∈ [0, 1]`` 映射到正弦特征向量，再经过两层 SiLU MLP。
频率表经过缩放，使动态范围大致与原来的整数步数 embedding（step ∈ [0, 1000]）
保持一致。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_KERNELS: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
DEFAULT_CHANNELS_PER_KERNEL: int = 48


def conv1d_kaiming(
    in_channels: int, out_channels: int, kernel_size: int
) -> nn.Conv1d:
    """带 Kaiming-normal 初始化的 Conv1d（保留自原始代码）。"""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SignalEncoder(nn.Module):
    """多核 1D 卷积分支，输出 ``len(kernels) * per_kernel`` 通道。

    Args:
        in_channels: 每个 slot 的输入通道数（通常为 1）。
        kernel_sizes: 奇数卷积核列表，每条使用 ``padding=ks // 2``。
        per_kernel_channels: 每条分支的输出通道数（默认 48）。
    """

    def __init__(
        self,
        in_channels: int = 1,
        kernel_sizes: Sequence[int] = DEFAULT_KERNELS,
        per_kernel_channels: int = DEFAULT_CHANNELS_PER_KERNEL,
    ) -> None:
        super().__init__()
        self.per_kernel_channels = per_kernel_channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.out_channels = per_kernel_channels * len(self.kernel_sizes)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    per_kernel_channels,
                    kernel_size=ks,
                    padding=ks // 2,
                )
                for ks in self.kernel_sizes
            ]
        )
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, in_channels, L) -> (B, out_channels, L)``。"""
        outputs = [conv(x) for conv in self.conv_layers]
        return torch.cat(outputs, dim=1)


class FlowTimeEmbedding(nn.Module):
    """Rectified Flow 使用的连续时间 ``t ∈ [0, 1]`` 正弦 embedding。

    实现说明：

    * 在做正弦基之前把 ``t`` 乘以 ``scale``（默认 1000），使有效相位范围
      与原扩散代码中的整数步数 embedding（step ∈ [0, num_steps)）对齐。
    * MLP 在两次 projection 之后都跟一个 SiLU，与原 ``DiffusionEmbedding``
      模块一致，保证下游权重形状完全相同。
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        projection_dim: int | None = None,
        *,
        scale: float = 1000.0,
        max_freq_exp: float = 4.0,
    ) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even, got {embedding_dim}"
            )
        if projection_dim is None:
            projection_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.scale = scale
        half = embedding_dim // 2
        # 频率定义与原 DiffusionEmbedding._build_embedding 保持一致。
        frequencies = 10.0 ** (
            torch.arange(half).float() / max(half - 1, 1) * max_freq_exp
        )
        self.register_buffer(
            "frequencies", frequencies.unsqueeze(0), persistent=False
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, t: Tensor) -> Tensor:
        """``(B,) float ∈ [0, 1] -> (B, projection_dim)``（经过 SiLU MLP 后）。"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        scaled = (t.float() * self.scale).unsqueeze(-1)  # (B, 1)
        phase = scaled * self.frequencies  # (B, half)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (B, dim)
        emb = self.projection1(emb)
        emb = F.silu(emb)
        emb = self.projection2(emb)
        emb = F.silu(emb)
        return emb
