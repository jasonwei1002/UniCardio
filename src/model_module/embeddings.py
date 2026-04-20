"""Input encoders and time-conditioning embeddings.

``SignalEncoder`` is preserved verbatim from the original UniCardio code: a bank
of six parallel Conv1d branches (kernels 1/3/5/7/9/11, 48 output channels each)
whose outputs are concatenated along channels to produce a 288-channel feature
map. Kaiming-normal weight initialization matches the original.

``FlowTimeEmbedding`` replaces ``DiffusionEmbedding`` for Rectified Flow. It
maps a continuous time ``t ∈ [0, 1]`` to a sinusoidal feature vector followed
by a two-layer SiLU MLP. The frequency table is scaled so that the dynamic
range roughly matches the original integer-step embedding (step ∈ [0, 1000]).
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
    """Conv1d with Kaiming-normal init (preserved from original code)."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SignalEncoder(nn.Module):
    """Multi-kernel 1D conv bank producing ``len(kernels) * per_kernel`` channels.

    Args:
        in_channels: Number of input channels per slot (typically 1).
        kernel_sizes: Odd kernel sizes, each paired with ``padding=ks // 2``.
        per_kernel_channels: Output channels per kernel branch (default 48).
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
        """``(B, in_channels, L) -> (B, out_channels, L)``."""
        outputs = [conv(x) for conv in self.conv_layers]
        return torch.cat(outputs, dim=1)


class FlowTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous ``t ∈ [0, 1]`` used by Rectified Flow.

    Implementation notes:

    * We multiply ``t`` by ``scale`` (default 1000) before the sinusoidal basis
      so the effective phase range matches the integer-step embedding in the
      original diffusion code (step ∈ [0, num_steps)).
    * The MLP applies SiLU after both projections, matching the original
      ``DiffusionEmbedding`` module so downstream weight shapes are identical.
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
        # Frequencies mirror the original DiffusionEmbedding._build_embedding.
        frequencies = 10.0 ** (
            torch.arange(half).float() / max(half - 1, 1) * max_freq_exp
        )
        self.register_buffer(
            "frequencies", frequencies.unsqueeze(0), persistent=False
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, t: Tensor) -> Tensor:
        """``(B,) float in [0, 1] -> (B, projection_dim)`` after SiLU MLP."""
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
