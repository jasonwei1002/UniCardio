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
import swanlab
import torch
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_module.datamodule import build_loaders
from src.model_module.unicardio_rf import UniCardioRF
from src.trainer_module.trainer import train
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    """仅支持 ``cuda`` 和 ``cpu``；cuda 不可用时自动降级到 cpu。"""
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(name)


def _init_swanlab(cfg: DictConfig) -> None:
    """根据 cfg.swanlab 初始化 SwanLab；完整 cfg 作为实验配置写入。"""
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


    if device.type == "cuda" and not bool(cfg.deterministic):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, _ = build_loaders(cfg.data)

    model = UniCardioRF(cfg.model)
    logger.info(
        "UniCardioRF parameters: %.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    compile_cfg = (cfg.trainer.get("compile") or {}) if "compile" in cfg.trainer else {}
    if bool(compile_cfg.get("enabled", False)) and device.type == "cuda":
        compile_mode = str(compile_cfg.get("mode", "reduce-overhead"))
        logger.info("torch.compile enabled (mode=%s, fullgraph=False)", compile_mode)
        model = torch.compile(model, mode=compile_mode, fullgraph=False)
    elif bool(compile_cfg.get("enabled", False)):
        logger.info("torch.compile requested but device=%s; skipping.", device.type)

    _init_swanlab(cfg)
    try:
        train(
            model,
            cfg.trainer,
            train_loader,
            val_loader,
            device=device,
            output_dir=cfg.output_dir,
            sampler_n_steps=int(cfg.sampler.n_steps),
            srate=int(cfg.data.srate),
        )
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
