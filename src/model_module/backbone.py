"""UniCardio 的 3-slot Rectified Flow 主干网络（patch-tokenization 版）。

token 粒度从 sample (8 ms @125 Hz) 重构为 patch (``patch_size`` 个 sample)，
让 attention 在事件级（R 峰 / systolic foot / dicrotic notch ≈ 80–200 ms）
跑关联，而不是浪费在亚事件冗余上。详见
``reports/vit-patch-model-reactive-barto.md``。

数据流（保持外部契约 ``(B, 1, 3*L_slot) → (B, 1, L_slot)`` 不变）：

* 输入 ``(B, 1, 3*L)`` 按 slot 切片 → 3× :class:`SignalEncoder`（多尺度
  Conv1d，stride=1 same-padding）→ 3× ``PatchProj`` (``Conv1d(C, C,
  kernel=patch_size, stride=patch_size)``) → 3× ``LayerNorm(C)``
  channel-last。
* concat → ``(B, C, 3 * n_patches_per_slot)``，加入按 patch 数生成的
  sinusoidal PE。
* 5–8 层 :class:`ResidualBlock`（block 本身 L-agnostic，无需改动）。
* CSDI 风格 skip-sum / sqrt(n_layers)。
* 切出 target slot 的 patch 区块 → ``PatchUnproj`` (``ConvTranspose1d(C,
  C, kernel=patch_size, stride=patch_size)``) 还原到 sample-rate → 复用
  现有 :class:`_OutputHead` (Conv1d 288→288→1) 输出 ``(B, 1, L_slot)``。

注意 attention mask 的 ``L_slot`` 参数语义在 patch 化后变为 *token count*
（即 ``n_patches_per_slot``），由调用方（``rf_train_step`` /
``euler_sample``）通过 :attr:`UniCardioBackbone.n_patches_per_slot` 读取。
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
    # patch-tokenization：每个 transformer token 由 ``patch_size`` 个 sample 折叠而成。
    # ``slot_length`` 必须能被 ``patch_size`` 整除，否则 ConvTranspose 还原长度对不齐。
    patch_size: int = 25

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive; got {self.patch_size}")
        if self.slot_length % self.patch_size != 0:
            raise ValueError(
                f"slot_length ({self.slot_length}) must be divisible by "
                f"patch_size ({self.patch_size}); got remainder "
                f"{self.slot_length % self.patch_size}"
            )

    @property
    def n_patches_per_slot(self) -> int:
        """每 slot 折叠后的 transformer token 数。"""
        return self.slot_length // self.patch_size

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
            patch_size=int(cfg.get("patch_size", 25)),
        )


class _OutputHead(nn.Module):
    """每个 slot 独立的 2 级输出头（``C -> C -> 1``，中间接 ReLU）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj1 = conv1d_kaiming(channels, channels, 1)
        self.proj2 = conv1d_kaiming(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj2(F.relu(self.proj1(x)))


def _patch_proj_conv(channels: int, patch_size: int) -> nn.Conv1d:
    """patch projection：``stride=patch_size`` 无重叠卷积，把 ``(B, C, L)`` 折叠到 ``(B, C, L/patch_size)``。"""
    layer = nn.Conv1d(
        channels, channels, kernel_size=patch_size, stride=patch_size
    )
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _patch_unproj_conv(channels: int, patch_size: int) -> nn.ConvTranspose1d:
    """patch un-projection：与 ``_patch_proj_conv`` 镜像，把 patch 序列展开回 sample-rate。

    ``kernel_size == stride`` 保证输出长度恰好 ``n_patches × patch_size = L_slot``。
    """
    layer = nn.ConvTranspose1d(
        channels, channels, kernel_size=patch_size, stride=patch_size
    )
    nn.init.kaiming_normal_(layer.weight)
    return layer


class UniCardioBackbone(nn.Module):
    """按 slot 独立编码、patch 折叠、并通过任务头输出的 Transformer 主干。

    Forward 签名：``forward(x, t, mask, target_slot)``，其中

    * ``x``: ``(B, 1, 3 * L_slot)`` —— 按 slot 顺序 (ECG, PPG, ABP) 拼接、
      已把 target slot 替换成 ``x_t`` 的张量。
    * ``t``: ``(B,)`` —— 取值于 ``[0, 1]`` 的连续时间。
    * ``mask``: ``(3 * n_patches_per_slot, 3 * n_patches_per_slot)`` 的注意力
      mask；调用方传入时 ``L_slot`` 参数 = :attr:`n_patches_per_slot`。
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
        self.patch_size = cfg.patch_size
        self.n_patches_per_slot = cfg.n_patches_per_slot

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
        # patch projection 把每 slot 从 sample-rate (B, C, L) 折叠到 (B, C, L/patch_size)。
        self.patch_proj = nn.ModuleList(
            [_patch_proj_conv(self.channels, self.patch_size) for _ in range(N_SLOTS)]
        )
        # 切到 channel-last 维度（patch token），LayerNorm 不再被 L 耦合。
        self.input_norms = nn.ModuleList(
            [nn.LayerNorm(self.channels) for _ in range(N_SLOTS)]
        )

        self.time_embedding = FlowTimeEmbedding(
            embedding_dim=cfg.time_embedding_dim
        )

        transformer_total_length = self.n_patches_per_slot * N_SLOTS
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

        # Sinusoidal PE 在 patch 粒度上注入。预转置到 (C, transformer_total_length)
        # 与残差流 (B, C, L_total_patches) 对齐，避免 forward 里重复 transpose。
        pe = sinusoidal_position_embedding(transformer_total_length, self.channels)
        self.register_buffer("pe", pe.t().contiguous(), persistent=False)

        # patch un-projection 把 target slot 的 patch 序列还原回 sample-rate；后接已有的 _OutputHead。
        self.patch_unproj = nn.ModuleList(
            [_patch_unproj_conv(self.channels, self.patch_size) for _ in range(N_SLOTS)]
        )
        self.output_heads = nn.ModuleList(
            [_OutputHead(self.channels) for _ in range(N_SLOTS)]
        )

    @property
    def total_length(self) -> int:
        """sample-rate 下的拼接长度（input 仍按 sample-rate 给）。"""
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
            encoded = self.input_encoders[i](x[:, :, start:end])  # (B, C, L_slot)
            patched = self.patch_proj[i](encoded)                  # (B, C, n_patches)
            # channel-last 上 LayerNorm 后再回到 (B, C, n_patches)。
            normed = self.input_norms[i](patched.transpose(1, 2)).transpose(1, 2)
            slot_feats.append(normed)

        h = torch.cat(slot_feats, dim=-1)        # (B, C, 3 * n_patches)
        h = h + self.pe.unsqueeze(0)              # (1, C, 3 * n_patches) 广播
        time_emb = self.time_embedding(t)

        skip_list: list[Tensor] = []
        for layer in self.residual_layers:
            h, skip = layer(h, time_emb, mask)
            skip_list.append(skip)

        h = torch.stack(skip_list, dim=0).sum(dim=0) / math.sqrt(self.n_layers)
        start = target_slot * self.n_patches_per_slot
        end = start + self.n_patches_per_slot
        slice_ = h[:, :, start:end]               # (B, C, n_patches)
        upsampled = self.patch_unproj[target_slot](slice_)  # (B, C, L_slot)
        return self.output_heads[target_slot](upsampled)
