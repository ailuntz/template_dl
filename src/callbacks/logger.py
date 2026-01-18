from pathlib import Path
from typing import List

from lightning.pytorch.loggers import TensorBoardLogger


def build_loggers(cfg: dict) -> List:
    save_dir = Path(cfg["paths"]["experiments_dir"]) / cfg["experiment_name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    loggers: List = [TensorBoardLogger(save_dir=tb_dir.as_posix(), name="", version="")]

    logging_cfg = cfg.get("logging", {})
    if not logging_cfg.get("use_wandb", False):
        return loggers

    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError as exc:
        raise ImportError("需要先安装 wandb") from exc
    wandb_dir = save_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    loggers.append(
        WandbLogger(
            project=logging_cfg.get("project", "ml_proj"),
            name=cfg.get("experiment_name", save_dir.name),
            save_dir=wandb_dir.as_posix(),
        )
    )
    return loggers
