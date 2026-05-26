"""BP head training entry point.

Mirrors :mod:`run.pipeline.train` but instantiates :class:`BPHead` instead of
:class:`UniCardioRF` and dispatches to :func:`bp_head_trainer.train`.

Usage::

    bash train_bp_head.sh                            # PulseDB Train_Subset pretrain
    bash finetune_bp_head.sh run/outputs/<run>/checkpoints/best.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import swanlab
import torch
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.bp_head import build_bp_head
from src.trainer_module.bp_head_trainer import train
from src.utils.normalization import BPLabelNorm
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(name)


def _init_swanlab(cfg: DictConfig) -> None:
    sw_cfg = cfg.get("swanlab", {}) or {}
    mode = "disabled" if not bool(sw_cfg.get("enabled", True)) else str(
        sw_cfg.get("mode", "cloud")
    )
    swanlab.init(
        project=str(sw_cfg.get("project", "UniCardio")),
        experiment_name=sw_cfg.get("experiment_name"),
        description=sw_cfg.get("description"),
        tags=list(sw_cfg.get("tags", []) or []),
        config=OmegaConf.to_container(cfg, resolve=True),
        logdir=str(Path(cfg.output_dir) / "swanlog"),
        mode=mode,
    )


@hydra.main(
    config_path="../conf",
    config_name="config_bp_head",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Resolved BP head config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _resolve_device(str(cfg.device))
    logger.info("Using device: %s", device)

    if device.type == "cuda" and not bool(cfg.deterministic):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    stage = str(cfg.trainer.get("stage", "pretrain"))
    train_loader, val_loader, test_loader = build_loaders(cfg.data, mode=stage)

    model = build_bp_head(cfg.model)
    logger.info(
        "BPHead parameters: %.1fk",
        sum(p.numel() for p in model.parameters()) / 1e3,
    )

    compile_cfg = (cfg.trainer.get("compile") or {}) if "compile" in cfg.trainer else {}
    if bool(compile_cfg.get("enabled", False)) and device.type == "cuda":
        compile_mode = str(compile_cfg.get("mode", "default"))
        logger.info("torch.compile enabled (mode=%s, fullgraph=False)", compile_mode)
        model = torch.compile(model, mode=compile_mode, fullgraph=False)

    _init_swanlab(cfg)
    bp_norm = BPLabelNorm.from_cfg(cfg.data)
    try:
        train(
            model,
            cfg.trainer,
            train_loader,
            val_loader,
            device=device,
            output_dir=cfg.output_dir,
            test_loader=test_loader if stage == "finetune" else None,
            bp_norm=bp_norm,
        )
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
