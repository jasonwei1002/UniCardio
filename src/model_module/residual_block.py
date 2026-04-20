""":class:`UniCardioBackbone` 使用的残差 Transformer block。

结构保留自 UniCardio 原始代码：

* 单层 ``nn.TransformerEncoder``（``d_model=channels``、GELU、
  ``dim_feedforward=64``）。
* 沿 token 维度的正弦位置编码（注册为 non-persistent buffer，
  以便随 ``.to(device)`` 一起搬运）。
* mid-projection 后使用门控激活 ``sigmoid(gate) * tanh(filter)``。
* 残差乘以 ``1 / sqrt(2)``；skip 单独返回，由 backbone 做聚合。

相比扩散版本唯一的行为差异：block 不再接收 ``device`` 参数。设备放置交由
调用方通过 ``module.to(device)`` 统一处理，更符合 PyTorch 惯例，也避免
重复的 tensor 搬运逻辑。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .embeddings import conv1d_kaiming


def _transformer_layer(
    channels: int,
    nheads: int,
    *,
    ffn_dim: int = 64,
    dropout: float = 0.0,
) -> nn.TransformerEncoder:
    """与原代码 ``get_torch_trans`` 对齐的 Transformer 层。"""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=nheads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=1)


def _sinusoidal_position_embedding(length: int, d_model: int) -> Tensor:
    """标准正弦位置编码，形状为 ``(length, d_model)``。"""
    pe = torch.zeros(length, d_model)
    position = torch.arange(length).unsqueeze(1).float()
    div_term = 1.0 / torch.pow(
        10000.0, torch.arange(0, d_model, 2).float() / d_model
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ResidualBlock(nn.Module):
    """一个带 flow-time 条件的门控 Transformer block。

    Args:
        channels: token 序列的特征维 ``C``。
        time_embedding_dim: 时间 embedding 的维度；通过一个线性层映射到
            ``channels`` 后再广播相加。
        nheads: 注意力头数。
        length: token 序列总长度（本项目为 ``3 * L_slot``）。
        ffn_dim: Transformer FFN 维度；与原代码的 64 保持一致。
    """

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int,
        nheads: int,
        length: int,
        *,
        ffn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.length = length
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.cond_projection = conv1d_kaiming(channels, 2 * channels, 1)
        self.mid_projection = conv1d_kaiming(channels, 2 * channels, 1)
        self.output_projection = conv1d_kaiming(channels, 2 * channels, 1)
        self.time_layer = _transformer_layer(channels, nheads, ffn_dim=ffn_dim)
        self.register_buffer(
            "pe",
            _sinusoidal_position_embedding(length, channels),
            persistent=False,
        )

    def forward(
        self, x: Tensor, time_emb: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """前向传播。

        Args:
            x: ``(B, C, L)`` 输入序列（C = ``channels``）。
            time_emb: ``(B, time_embedding_dim)`` 的 flow-time embedding。
            mask: ``(L, L)`` 加性注意力 mask（0 / -inf）。

        Returns:
            ``(x_out, skip)``，两者形状均为 ``(B, C, L)``。
        """
        B, C, L = x.shape
        base_shape = x.shape

        # 将时间 embedding 广播加到每个时空位置。
        t_emb = self.time_projection(time_emb).unsqueeze(-1)  # (B, C, 1)
        y = x + t_emb

        # 沿 token 轴的 Transformer self-attention，使用任务 mask。
        y_seq = y.permute(2, 0, 1)  # (L, B, C)
        y_seq = self.time_layer(y_seq, mask=mask)
        y = y_seq.permute(1, 2, 0).reshape(B, C, L)

        # 带位置偏置的门控 mid-projection。
        y = self.mid_projection(y)  # (B, 2C, L)
        pos_bias = (
            self.pe.unsqueeze(0)
            .expand(B, -1, -1)
            .permute(0, 2, 1)
            .contiguous()
        )  # (B, C, L)
        y = y + self.cond_projection(pos_bias)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B, C, L)

        y = self.output_projection(y)  # (B, 2C, L)
        residual, skip = torch.chunk(y, 2, dim=1)

        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
