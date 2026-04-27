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
    downsample_factor: int = 1

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
            downsample_factor=int(cfg.get("downsample_factor", 1)),
        )


class _OutputHead(nn.Module):
    """每个 slot 独立的 2 级输出头（``C -> C -> 1``，中间接 ReLU）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj1 = conv1d_kaiming(channels, channels, 1)
        self.proj2 = conv1d_kaiming(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj2(F.relu(self.proj1(x)))


def _make_resampler(
    channels: int, ds: int, *, transpose: bool
) -> nn.Module:
    """slot 内 (上/下) 采样模块；``ds=1`` 退化为 ``nn.Identity``。

    使用 ``kernel = 2 * ds, stride = ds, padding = ds // 2``：偶数 ds 且
    ``L % ds == 0`` 时 ``Conv1d`` L → L/ds 与 ``ConvTranspose1d`` L/ds → L
    严格 round-trip。Kaiming-normal 初始化与其他卷积一致。
    """
    if ds == 1:
        return nn.Identity()
    cls = nn.ConvTranspose1d if transpose else nn.Conv1d
    layer = cls(channels, channels, kernel_size=2 * ds, stride=ds, padding=ds // 2)
    nn.init.kaiming_normal_(layer.weight)
    return layer


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
        self.transformer_slot_length = self._validate_downsample(
            self.L, cfg.downsample_factor
        )

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

        # ds=1 时退化成 nn.Identity，避免 forward 里 if-None 分支。
        ds = cfg.downsample_factor
        self.downsamplers = nn.ModuleList(
            [_make_resampler(self.channels, ds, transpose=False) for _ in range(N_SLOTS)]
        )
        self.upsamplers = nn.ModuleList(
            [_make_resampler(self.channels, ds, transpose=True) for _ in range(N_SLOTS)]
        )

        self.time_embedding = FlowTimeEmbedding(
            embedding_dim=cfg.time_embedding_dim
        )

        transformer_total_length = self.transformer_slot_length * N_SLOTS
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    time_embedding_dim=cfg.time_embedding_dim,
                    nheads=cfg.nheads,
                    length=transformer_total_length,
                    ffn_dim=cfg.ffn_dim,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.output_heads = nn.ModuleList(
            [_OutputHead(self.channels) for _ in range(N_SLOTS)]
        )

    @staticmethod
    def _validate_downsample(slot_length: int, ds: int) -> int:
        # 仅允许 1 与 2 的幂：偶数 ds 时下采样/上采样的 kernel/stride/padding
        # 公式可严格 round-trip，奇数 ds 会引入非整数长度。
        if ds < 1:
            raise ValueError(f"downsample_factor must be >= 1; got {ds}")
        if ds > 1 and (ds & (ds - 1)) != 0:
            raise ValueError(
                f"downsample_factor must be a power of 2 (or 1); got {ds}"
            )
        if slot_length % ds != 0:
            raise ValueError(
                f"slot_length ({slot_length}) must be divisible by "
                f"downsample_factor ({ds})."
            )
        return slot_length // ds

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
            encoded = self.downsamplers[i](encoded)
            slot_feats.append(encoded)

        L_inner = self.transformer_slot_length
        h = torch.cat(slot_feats, dim=-1)
        time_emb = self.time_embedding(t)

        skip_list: list[Tensor] = []
        for layer in self.residual_layers:
            h, skip = layer(h, time_emb, mask)
            skip_list.append(skip)

        h = torch.stack(skip_list, dim=0).sum(dim=0) / math.sqrt(self.n_layers)
        start = target_slot * L_inner
        end = start + L_inner
        target_feat = self.upsamplers[target_slot](h[:, :, start:end])
        return self.output_heads[target_slot](target_feat)
