"""训练入口：Hydra 配置 + UniCardioRF + Rectified Flow 训练器。

请在仓库根目录运行，这样 Hydra 才能正确定位 ``run/conf``：

    python run/pipeline/train.py
    python run/pipeline/train.py device=cpu trainer.epochs=2 data.num_workers=0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# 不论是直接运行还是通过 ``python -m`` 调用，都让 ``import src.*`` 生效。
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.trainer import train
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    """将配置中的 device 字符串映射为本机可用的 ``torch.device``。"""
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu.")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        logger.warning("mps requested but unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(name)


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _resolve_device(str(cfg.device))
    logger.info("Using device: %s", device)

    train_loader, val_loader, _ = build_loaders(cfg.data)

    model = UniCardioRF(cfg.model)
    logger.info(
        "UniCardioRF parameters: %.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    train(
        model,
        cfg.trainer,
        train_loader,
        val_loader,
        device=device,
        output_dir=cfg.output_dir,
    )


if __name__ == "__main__":
    main()
