"""UniCardio 的 3-slot Rectified Flow 主干网络。

替换了原扩散代码库中的 4-slot ``diff_CSDI`` 模块。主要差异：

* 3 个 slot（ECG=0、PPG=1、ABP=2）——去掉了原先零填充的占位 slot 以及
  相关的 ``borrow_mode`` 输出头路由。
* 使用 :class:`FlowTimeEmbedding`（``t ∈ [0, 1]``）做连续时间条件，取代
  原本基于整数步数的 ``DiffusionEmbedding``。
* mask 由调用方传入（在别处按任务构造），不再从 7 个硬编码的预注册 mask
  中选择。

原样保留的结构：:class:`SignalEncoder`、:class:`ResidualBlock`、
按 slot 划分的 :class:`LayerNorm`、每个 slot 的 2 级输出头、以及
乘以 ``1 / sqrt(n_layers)`` 的 skip-sum 聚合。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .embeddings import (
    DEFAULT_CHANNELS_PER_KERNEL,
    DEFAULT_KERNELS,
    FlowTimeEmbedding,
    SignalEncoder,
    conv1d_kaiming,
    sinusoidal_position_embedding,
)
from .residual_block import ResidualBlock

N_SLOTS: int = 3


@dataclass(frozen=True)
class BackboneConfig:
    """纯 dataclass 配置，方便在不依赖 Hydra 的场景下实例化 backbone。"""

    slot_length: int = 500
    channels: int = 288
    n_layers: int = 5
    nheads: int = 8
    time_embedding_dim: int = 256
    kernel_sizes: tuple[int, ...] = DEFAULT_KERNELS
    per_kernel_channels: int = DEFAULT_CHANNELS_PER_KERNEL
    ffn_dim: int = 64

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "BackboneConfig":
        """从普通 dict（或 OmegaConf DictConfig）构造配置。"""
        return cls(
            slot_length=int(cfg["slot_length"]),
            channels=int(cfg["channels"]),
            n_layers=int(cfg.get("n_layers", cfg.get("layers", 5))),
            nheads=int(cfg["nheads"]),
            time_embedding_dim=int(cfg["time_embedding_dim"]),
            kernel_sizes=tuple(cfg.get("kernel_sizes", DEFAULT_KERNELS)),
            per_kernel_channels=int(
                cfg.get("per_kernel_channels", DEFAULT_CHANNELS_PER_KERNEL)
            ),
            ffn_dim=int(cfg.get("ffn_dim", 64)),
        )


class _OutputHead(nn.Module):
    """每个 slot 独立的 2 级输出头（``C -> C -> 1``，中间接 ReLU）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj1 = conv1d_kaiming(channels, channels, 1)
        self.proj2 = conv1d_kaiming(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj2(F.relu(self.proj1(x)))


class UniCardioBackbone(nn.Module):
    """按 slot 独立编码、并通过任务头输出的 Transformer 主干。

    Forward 签名：``forward(x, t, mask, target_slot)``，其中

    * ``x``: ``(B, 1, 3 * L_slot)`` —— 拼接后的 slot 张量。
    * ``t``: ``(B,)`` —— 取值于 ``[0, 1]`` 的连续时间。
    * ``mask``: ``(3 * L_slot, 3 * L_slot)`` —— 加性注意力 mask。
    * ``target_slot``: ``int`` —— 选择使用的输出头。
    """

    def __init__(self, config: Mapping[str, Any] | BackboneConfig) -> None:
        super().__init__()
        cfg = (
            config
            if isinstance(config, BackboneConfig)
            else BackboneConfig.from_mapping(config)
        )
        self.cfg = cfg
        self.L = cfg.slot_length
        self.channels = cfg.channels
        self.n_layers = cfg.n_layers

        # 编码器 bank 的特征维必须与 self.channels 一致。
        encoded_channels = cfg.per_kernel_channels * len(cfg.kernel_sizes)
        if encoded_channels != cfg.channels:
            raise ValueError(
                f"channels ({cfg.channels}) must equal "
                f"per_kernel_channels * len(kernel_sizes) "
                f"({cfg.per_kernel_channels} * {len(cfg.kernel_sizes)} "
                f"= {encoded_channels})."
            )

        self.input_encoders = nn.ModuleList(
            [
                SignalEncoder(
                    in_channels=1,
                    kernel_sizes=cfg.kernel_sizes,
                    per_kernel_channels=cfg.per_kernel_channels,
                )
                for _ in range(N_SLOTS)
            ]
        )
        self.input_norms = nn.ModuleList(
            [nn.LayerNorm([self.channels, self.L]) for _ in range(N_SLOTS)]
        )

        self.time_embedding = FlowTimeEmbedding(
            embedding_dim=cfg.time_embedding_dim
        )

        transformer_total_length = self.L * N_SLOTS
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    time_embedding_dim=cfg.time_embedding_dim,
                    nheads=cfg.nheads,
                    ffn_dim=cfg.ffn_dim,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Sinusoidal PE 一次性注入到 residual stack 输入端，让每层 self-attn
        # 都能通过残差路径看到 token-level 位置信息。预转置到 (C, L_total)
        # 与残差流 ``(B, C, L_total)`` 对齐，避免每次 forward 重复 transpose。
        pe = sinusoidal_position_embedding(transformer_total_length, self.channels)
        self.register_buffer("pe", pe.t().contiguous(), persistent=False)

        self.output_heads = nn.ModuleList(
            [_OutputHead(self.channels) for _ in range(N_SLOTS)]
        )

    @property
    def total_length(self) -> int:
        return self.L * N_SLOTS

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        target_slot: int,
    ) -> Tensor:
        """只对 target slot 预测速度。

        返回形状为 ``(B, 1, L_slot)`` 的张量。
        """
        B, input_dim, L_total = x.shape
        if L_total != self.total_length:
            raise ValueError(
                f"Expected L_total={self.total_length}, got {L_total}"
            )
        if input_dim != 1:
            raise ValueError(f"Expected input channel dim 1, got {input_dim}")
        if not 0 <= target_slot < N_SLOTS:
            raise ValueError(
                f"target_slot must be in [0, {N_SLOTS}), got {target_slot}"
            )

        slot_feats: list[Tensor] = []
        for i in range(N_SLOTS):
            start, end = i * self.L, (i + 1) * self.L
            encoded = self.input_encoders[i](x[:, :, start:end])
            encoded = encoded.reshape(B, self.channels, self.L)
            encoded = self.input_norms[i](encoded)
            slot_feats.append(encoded)

        h = torch.cat(slot_feats, dim=-1)
        h = h + self.pe.unsqueeze(0)        # (1, C, L_total) 广播
        time_emb = self.time_embedding(t)

        skip_list: list[Tensor] = []
        for layer in self.residual_layers:
            h, skip = layer(h, time_emb, mask)
            skip_list.append(skip)

        h = torch.stack(skip_list, dim=0).sum(dim=0) / math.sqrt(self.n_layers)
        start = target_slot * self.L
        end = start + self.L
        return self.output_heads[target_slot](h[:, :, start:end])
