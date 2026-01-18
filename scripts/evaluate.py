import argparse
from pathlib import Path

import lightning as L

from src.data import create_datamodule
from src.lit_modules import create_lit_module
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--override", type=Path, nargs="*", default=None, help="额外配置文件，后加载覆盖前者")
    args, cli_overrides = parser.parse_known_args()

    if not args.override:
        raise ValueError("需要提供 --override 指向 custom 配置文件")

    cfg = load_config(args.config, args.override, cli_overrides)
    save_dir = Path(cfg.paths.experiments_dir) / cfg.experiment_name
    ckpt_dir = save_dir / "checkpoints"
    ckpt_path = next((p for p in (ckpt_dir / "best.ckpt", ckpt_dir / "last.ckpt") if p.exists()), None)
    if not ckpt_path:
        raise FileNotFoundError(f"未找到 best.ckpt 或 last.ckpt: {ckpt_dir}")

    datamodule = create_datamodule(cfg)
    model = create_lit_module(cfg, ckpt_path=ckpt_path, weights_only=False)
    trainer_cfg = cfg.trainer
    trainer = L.Trainer(
        logger=False,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
    )
    metrics = trainer.test(model, datamodule=datamodule)
    print(metrics)


if __name__ == "__main__":
    main()
