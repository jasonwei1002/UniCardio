"""Path A refinement model — per-vital HF PatchTSMixer + demographics MLP.

Faithful reproduction of MD-ViSCo's refinement model (IEEE JBHI 2026,
Sec III.D / §6.2.2): each source vital (ECG slot 0, PPG slot 1) is encoded by
its own HuggingFace ``PatchTSMixerModel``; the GAP-pooled patch embedding is
fused with a single shared demographics embedding and mapped to ``(SBP, DBP)``
by a per-vital head. Per-vital predictions are averaged over the active vitals.

Intentional differences vs MD-ViSCo:
- demographics via a numeric MLP (PulseDB has a fixed 5-feature schema), not
  DistilBERT (which exists for cross-dataset schema agnosticism we do not need);
- GAP pooling over patches instead of flatten + ProjectionHead (keeps the head
  at ~22M instead of ~40M+);
- no WCL (weighted contrastive loss) — MD-ViSCo's ablation shows architecture >
  WCL; revisit only if this head still falls short of AAMI.

``demographics`` is ``(B, 6) = [age_z, gender, height_z, weight_z, bmi_z, mask]``
(see :mod:`src.data_module.cardiac_dataset`). Pass ``None`` for waveform-only
paths; the demographic branch is then bypassed (``demo_emb = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from transformers import PatchTSMixerConfig, PatchTSMixerModel

# Model slot order is ECG=0, PPG=1 (see src/model_module/tasks.py). The BP head
# receives ``signal[:, :2, :]`` so these indices line up directly.
_VITAL_CHANNEL: dict[str, int] = {"ecg": 0, "ppg": 1}


@dataclass(frozen=True)
class BPHeadConfig:
    """Immutable config for :class:`BPHead` (HF PatchTSMixer hyperparameters
    follow MD-ViSCo §6.2.2)."""

    slot_length: int = 1250
    vitals: tuple[str, ...] = ("ecg", "ppg")
    # HF PatchTSMixer encoder hyperparameters.
    patch_len: int = 5            # MD-ViSCo uses 4 @ L=1280; 5 divides 1250 → 250 patches
    patch_stride: int = 5
    d_model: int = 64
    num_layers: int = 15
    expansion_factor: int = 5
    # Demographics + fusion head.
    demo_in: int = 6              # 5 numeric features + 1 missingness mask
    demo_hidden: int = 64
    fusion_hidden: int = 128

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | "BPHeadConfig") -> "BPHeadConfig":
        if isinstance(cfg, cls):
            return cfg
        d = cls()
        vitals = cfg.get("vitals", d.vitals)
        return cls(
            slot_length=int(cfg.get("slot_length", d.slot_length)),
            vitals=tuple(str(v).lower() for v in vitals),
            patch_len=int(cfg.get("patch_len", d.patch_len)),
            patch_stride=int(cfg.get("patch_stride", d.patch_stride)),
            d_model=int(cfg.get("d_model", d.d_model)),
            num_layers=int(cfg.get("num_layers", d.num_layers)),
            expansion_factor=int(cfg.get("expansion_factor", d.expansion_factor)),
            demo_in=int(cfg.get("demo_in", d.demo_in)),
            demo_hidden=int(cfg.get("demo_hidden", d.demo_hidden)),
            fusion_hidden=int(cfg.get("fusion_hidden", d.fusion_hidden)),
        )


def _build_encoder(cfg: BPHeadConfig) -> PatchTSMixerModel:
    """One single-channel HF PatchTSMixer encoder per vital."""
    return PatchTSMixerModel(
        PatchTSMixerConfig(
            context_length=cfg.slot_length,
            num_input_channels=1,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            expansion_factor=cfg.expansion_factor,
            patch_length=cfg.patch_len,
            patch_stride=cfg.patch_stride,
        )
    )


class BPHead(nn.Module):
    """(ECG, PPG, demographics) -> (SBP, DBP) mmHg regressor (per-vital averaged)."""

    def __init__(self, config: BPHeadConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = BPHeadConfig.from_mapping(config)
        cfg = self.config
        for v in cfg.vitals:
            if v not in _VITAL_CHANNEL:
                raise ValueError(
                    f"unknown vital '{v}'; BP head supports {sorted(_VITAL_CHANNEL)}"
                )

        self.encoders = nn.ModuleDict({v: _build_encoder(cfg) for v in cfg.vitals})

        self.demo_encoder = nn.Sequential(
            nn.Linear(cfg.demo_in, cfg.demo_hidden),
            nn.GELU(),
            nn.Linear(cfg.demo_hidden, cfg.demo_hidden),
        )

        self.heads = nn.ModuleDict(
            {
                v: nn.Sequential(
                    nn.Linear(cfg.d_model + cfg.demo_hidden, cfg.fusion_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.fusion_hidden, 2),
                )
                for v in cfg.vitals
            }
        )

    def _encode(self, vital: str, slot: Tensor) -> Tensor:
        """``(B, 1, L)`` raw waveform -> ``(B, d_model)`` GAP patch embedding."""
        # HF PatchTSMixer expects (B, seq_len, channels); we hold (B, 1, L).
        hidden = self.encoders[vital](slot.transpose(1, 2)).last_hidden_state
        # last_hidden_state: (B, channels=1, num_patches, d_model). GAP over
        # the patch axis, drop the singleton channel axis -> (B, d_model).
        return hidden.mean(dim=2).squeeze(1)

    def forward(
        self,
        ecg_ppg: Tensor,
        demographics: Optional[Tensor] = None,
        active_vitals: Optional[Iterable[str]] = None,
    ) -> Tensor:
        """Predict ``(SBP, DBP)`` mmHg, averaged over the active vitals.

        Args:
            ecg_ppg: ``(B, 2, slot_length)`` raw ECG (ch 0) + PPG (ch 1).
            demographics: ``(B, demo_in)`` z-scored demographics with missingness
                mask in the last column; ``None`` bypasses the demo branch.
            active_vitals: vital names to average over; ``None`` = all configured
                vitals (training uses this default — call site unchanged).

        Returns:
            ``(B, 2)`` with column 0 = SBP, column 1 = DBP, both in mmHg.
        """
        cfg = self.config
        b, c, length = ecg_ppg.shape
        if length != cfg.slot_length or c < 2:
            raise ValueError(
                f"BPHead expects (B, >=2, {cfg.slot_length}); got (B, {c}, {length})."
            )

        vitals = (
            tuple(str(v).lower() for v in active_vitals)
            if active_vitals is not None
            else cfg.vitals
        )
        if not vitals:
            raise ValueError("active_vitals resolved to an empty set.")

        if demographics is None:
            demo_emb = ecg_ppg.new_zeros(b, cfg.demo_hidden)
        else:
            demo_emb = self.demo_encoder(demographics)

        preds = []
        for v in vitals:
            ch = _VITAL_CHANNEL[v]
            wave_emb = self._encode(v, ecg_ppg[:, ch : ch + 1, :])
            preds.append(self.heads[v](torch.cat([wave_emb, demo_emb.to(wave_emb.dtype)], dim=-1)))
        return torch.stack(preds, dim=0).mean(dim=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_bp_head(config: Mapping[str, Any] | BPHeadConfig) -> BPHead:
    """Factory helper mirroring other model-module entry points."""
    return BPHead(config)
