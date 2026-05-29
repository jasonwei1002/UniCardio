"""BP refinement model — faithful MD-ViSCo ``VitalEncoder`` (pi=true).

Faithful reproduction of MD-ViSCo's BP refinement model on PulseDB
(IEEE JBHI 2026, §6.2.2; upstream ``src/model/mdvisco.py`` ``VitalEncoder`` /
``BPModel``). Per source vital (ECG slot 0, PPG slot 1):

    waveform_encoder (HF PatchTSMixer)  -> last_hidden_state (B,1,np1,d)
      -> flatten (B, np1*d)
      -> ProjectionHead (-> projection_dim)            -> wav_emb (B, proj)
      -> stack with demo_emb as a 2nd channel          -> (B, 2, proj)
      -> second_encoder (HF PatchTSMixer, 2 channels)  -> (B, 2, np2, d)
      -> flatten (B, 2*np2*d)
      -> MlpBP sbp_head / dbp_head                      -> (B, 2)

Per-vital ``(SBP, DBP)`` predictions are averaged over the active vitals
(``BPModel.forward`` aggregation).

Intentional differences vs MD-ViSCo's PulseDB-faithful config (both deferred
by the user, not architectural simplifications):
- **demographics via a numeric MLP, not DistilBERT.** MD-ViSCo's pi=true path
  fuses a DistilBERT text embedding as the 2nd channel of the second encoder;
  we keep that exact fusion point but source the embedding from a numeric MLP
  over PulseDB's fixed 5-feature schema (+ missingness mask).
- **demographics still numeric, but WCL is now implemented.** WCL (the multi-WCL
  contrastive objective) lives in the trainer (:mod:`src.trainer_module.wcl` +
  ``bp_head_trainer``); this module exposes the embeddings it needs via
  ``forward(return_embeddings=True)`` and :meth:`BPHead.encode_embeddings`. Only
  the DistilBERT demographics encoder remains deferred (numeric MLP instead).

``demographics`` is ``(B, 6) = [age_z, gender, height_z, weight_z, bmi_z, mask]``
(see :mod:`src.data_module.cardiac_dataset`). Pass ``None`` for waveform-only
paths; the demo channel is then zero-filled so ``in_features`` is unchanged.
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
    """Immutable config for :class:`BPHead` (MD-ViSCo ``VitalEncoder`` §6.2.2)."""

    slot_length: int = 1250
    vitals: tuple[str, ...] = ("ecg", "ppg")
    # HF PatchTSMixer encoder hyperparameters (shared by both encoder stages).
    patch_len: int = 5            # MD-ViSCo uses 4 @ L=1280; 5 divides 1250 → 250 patches
    patch_stride: int = 5
    d_model: int = 64
    num_layers: int = 15
    expansion_factor: int = 5
    # ProjectionHead + second-encoder fusion.
    projection_dim: int = 512
    proj_dropout: float = 0.1
    mlp_dropout: float = 0.2      # MlpBP dropout (MD-ViSCo uses 0.2)
    # Demographics MLP (replaces DistilBERT; projects to projection_dim).
    demo_in: int = 6              # 5 numeric features + 1 missingness mask
    demo_hidden: int = 64

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
            projection_dim=int(cfg.get("projection_dim", d.projection_dim)),
            proj_dropout=float(cfg.get("proj_dropout", d.proj_dropout)),
            mlp_dropout=float(cfg.get("mlp_dropout", d.mlp_dropout)),
            demo_in=int(cfg.get("demo_in", d.demo_in)),
            demo_hidden=int(cfg.get("demo_hidden", d.demo_hidden)),
        )


def _num_patches(context_length: int, patch_len: int, patch_stride: int) -> int:
    """``floor((ctx - patch) / stride) + 1`` (MD-ViSCo ``calculate_image_embedding``)."""
    return (context_length - patch_len) // patch_stride + 1


def _build_encoder(cfg: BPHeadConfig, context_length: int, channels: int) -> PatchTSMixerModel:
    """One HF PatchTSMixer encoder (shared hyperparameters, variable ctx/channels)."""
    return PatchTSMixerModel(
        PatchTSMixerConfig(
            context_length=context_length,
            num_input_channels=channels,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            expansion_factor=cfg.expansion_factor,
            patch_length=cfg.patch_len,
            patch_stride=cfg.patch_stride,
        )
    )


class ProjectionHead(nn.Module):
    """MD-ViSCo ``ProjectionHead`` (mdvisco.py:3237).

    ``Linear -> GELU -> Linear -> Dropout -> + residual(projected) -> LayerNorm``.
    """

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)


class MlpBP(nn.Module):
    """MD-ViSCo ``MlpBP`` (mdvisco.py:2948): ``Dropout -> Linear -> GELU -> Dropout -> Linear``."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


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

        np1 = _num_patches(cfg.slot_length, cfg.patch_len, cfg.patch_stride)
        np2 = _num_patches(cfg.projection_dim, cfg.patch_len, cfg.patch_stride)
        if np1 <= 0 or np2 <= 0:
            raise ValueError(
                f"invalid patch geometry: np1={np1}, np2={np2} "
                f"(slot_length={cfg.slot_length}, projection_dim={cfg.projection_dim}, "
                f"patch_len={cfg.patch_len}, patch_stride={cfg.patch_stride})"
            )
        image_embedding = np1 * cfg.d_model              # stage-1 flatten dim
        mlp_in = 2 * np2 * cfg.d_model                   # stage-2 flatten dim (2 channels)

        # Per-vital stage-1 waveform encoders + projection heads + 2-channel
        # stage-2 fusion encoders. Each vital owns an independent stack
        # (MD-ViSCo's per-vital ``VitalEncoder``).
        self.waveform_encoders = nn.ModuleDict(
            {v: _build_encoder(cfg, cfg.slot_length, 1) for v in cfg.vitals}
        )
        self.projections = nn.ModuleDict(
            {
                v: ProjectionHead(image_embedding, cfg.projection_dim, cfg.proj_dropout)
                for v in cfg.vitals
            }
        )
        self.fusion_encoders = nn.ModuleDict(
            {v: _build_encoder(cfg, cfg.projection_dim, 2) for v in cfg.vitals}
        )
        self.sbp_heads = nn.ModuleDict(
            {v: MlpBP(mlp_in, mlp_in // 2, 1, cfg.mlp_dropout) for v in cfg.vitals}
        )
        self.dbp_heads = nn.ModuleDict(
            {v: MlpBP(mlp_in, mlp_in // 2, 1, cfg.mlp_dropout) for v in cfg.vitals}
        )

        # Shared demographics MLP -> projection_dim (the PI 2nd channel; replaces
        # MD-ViSCo's DistilBERT text-projection branch).
        self.demo_encoder = nn.Sequential(
            nn.Linear(cfg.demo_in, cfg.demo_hidden),
            nn.GELU(),
            nn.Linear(cfg.demo_hidden, cfg.projection_dim),
        )

    def _wav_embed(self, vital: str, slot: Tensor) -> Tensor:
        """``(B, 1, L)`` raw waveform -> ``(B, proj)`` ProjectionHead embedding.

        This is MD-ViSCo's waveform embedding ``e_W^(i)`` — the WCL waveform
        terms contrast on it, and it is the only part the contrastive
        pretraining stage needs (fusion + MlpBP heads are skipped).
        """
        b = slot.shape[0]
        # HF PatchTSMixer expects (B, seq_len, channels); we hold (B, 1, L).
        feat = self.waveform_encoders[vital](slot.transpose(1, 2)).last_hidden_state
        return self.projections[vital](feat.reshape(b, -1))            # (B, proj)

    def _encode_vital(
        self, vital: str, slot: Tensor, demo_emb: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``(B, 1, L)`` raw waveform + ``(B, proj)`` demo embedding.

        Returns ``(pred, wav_emb)`` where ``pred`` is ``(B, 2)`` (SBP, DBP) and
        ``wav_emb`` is the ``(B, proj)`` ProjectionHead output — the waveform
        embedding ``e_W^(i)`` that MD-ViSCo's WCL waveform terms contrast on.
        """
        b = slot.shape[0]
        wav_emb = self._wav_embed(vital, slot)                         # (B, proj)
        # Fuse demo as the 2nd channel: (B, 2, proj) -> (B, proj, 2) for the encoder.
        combined = torch.stack([wav_emb, demo_emb.to(wav_emb.dtype)], dim=1)  # (B, 2, proj)
        fused = self.fusion_encoders[vital](combined.transpose(1, 2)).last_hidden_state
        final = fused.reshape(b, -1)                                    # (B, 2*np2*d)
        sbp = self.sbp_heads[vital](final)
        dbp = self.dbp_heads[vital](final)
        return torch.cat([sbp, dbp], dim=-1), wav_emb                   # (B, 2), (B, proj)

    def _resolve_vitals(self, active_vitals: Optional[Iterable[str]]) -> tuple[str, ...]:
        vitals = (
            tuple(str(v).lower() for v in active_vitals)
            if active_vitals is not None
            else self.config.vitals
        )
        if not vitals:
            raise ValueError("active_vitals resolved to an empty set.")
        return vitals

    def encode_embeddings(
        self,
        ecg_ppg: Tensor,
        demographics: Optional[Tensor] = None,
        active_vitals: Optional[Iterable[str]] = None,
    ) -> dict[str, Tensor]:
        """Per-vital waveform embeddings + demo embedding, WITHOUT fusion/heads.

        Returns ``{"{vital}_embeddings": (B, proj), ["text_embeddings": (B, proj)]}``
        — exactly the tensors MD-ViSCo's multi-WCL contrasts on. Used by the
        contrastive *pretraining* stage, which trains only the waveform encoders
        + projection heads (and the demographics MLP), so it skips the
        parameter-heavy fusion encoders and MlpBP heads entirely.
        """
        cfg = self.config
        b, c, length = ecg_ppg.shape
        if length != cfg.slot_length or c < 2:
            raise ValueError(
                f"BPHead expects (B, >=2, {cfg.slot_length}); got (B, {c}, {length})."
            )
        vitals = self._resolve_vitals(active_vitals)
        embeddings: dict[str, Tensor] = {
            f"{v}_embeddings": self._wav_embed(
                v, ecg_ppg[:, _VITAL_CHANNEL[v] : _VITAL_CHANNEL[v] + 1, :]
            )
            for v in vitals
        }
        if demographics is not None:
            embeddings["text_embeddings"] = self.demo_encoder(demographics)
        return embeddings

    def forward(
        self,
        ecg_ppg: Tensor,
        demographics: Optional[Tensor] = None,
        active_vitals: Optional[Iterable[str]] = None,
        return_embeddings: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Predict ``(SBP, DBP)`` mmHg, averaged over the active vitals.

        Args:
            ecg_ppg: ``(B, 2, slot_length)`` raw ECG (ch 0) + PPG (ch 1).
            demographics: ``(B, demo_in)`` z-scored demographics with missingness
                mask in the last column; ``None`` zero-fills the demo channel.
            active_vitals: vital names to average over; ``None`` = all configured
                vitals (training uses this default — call site unchanged).
            return_embeddings: if ``True``, also return the per-vital waveform
                embeddings (keyed ``"{vital}_embeddings"``) and, when
                demographics are present, the demographic embedding (keyed
                ``"text_embeddings"``) — the inputs MD-ViSCo's multi-WCL
                contrasts on. The keys match ``src.trainer_module.wcl`` term
                ``embedding_key``s.

        Returns:
            ``(B, 2)`` (SBP, DBP) in the head's training label space (normalized
            ``[0, 1]`` when ``data.bp_label_norm`` is set, else raw mmHg). When
            ``return_embeddings`` is ``True``, returns ``(preds, embeddings)``
            where ``embeddings`` is a ``{key: (B, proj)}`` dict.
        """
        cfg = self.config
        b, c, length = ecg_ppg.shape
        if length != cfg.slot_length or c < 2:
            raise ValueError(
                f"BPHead expects (B, >=2, {cfg.slot_length}); got (B, {c}, {length})."
            )

        vitals = self._resolve_vitals(active_vitals)

        if demographics is None:
            demo_emb = ecg_ppg.new_zeros(b, cfg.projection_dim)
        else:
            demo_emb = self.demo_encoder(demographics)

        preds: list[Tensor] = []
        embeddings: dict[str, Tensor] = {}
        for v in vitals:
            slot = ecg_ppg[:, _VITAL_CHANNEL[v] : _VITAL_CHANNEL[v] + 1, :]
            pred, wav_emb = self._encode_vital(v, slot, demo_emb)
            preds.append(pred)
            if return_embeddings:
                embeddings[f"{v}_embeddings"] = wav_emb

        out = torch.stack(preds, dim=0).mean(dim=0)
        if not return_embeddings:
            return out
        if demographics is not None:
            embeddings["text_embeddings"] = demo_emb
        return out, embeddings

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_bp_head(config: Mapping[str, Any] | BPHeadConfig) -> BPHead:
    """Factory helper mirroring other model-module entry points."""
    return BPHead(config)
