import argparse
import json
from pathlib import Path

import lightning as L

from src.callbacks.checkpoint import build_checkpoint
from src.callbacks.early_stopping import build_early_stopping
from src.callbacks.logger import build_loggers
from src.data import create_datamodule
from src.lit_modules import create_lit_module
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--override", type=Path, nargs="*", default=None, help="额外配置文件，后加载覆盖前者")
    parser.add_argument("--resume", action="store_true", help="从last.ckpt断点续练")
    args, cli_overrides = parser.parse_known_args()

    if not args.override:
        raise ValueError("需要提供 --override 指向 custom 配置文件")

    cfg = load_config(args.config, args.override, cli_overrides)
    save_dir = Path(cfg.paths.experiments_dir) / cfg.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = "last" if args.resume and (save_dir / "checkpoints" / "last.ckpt").exists() else None

    L.seed_everything(cfg.get("seed", 42), workers=True)
    datamodule = create_datamodule(cfg)
    model = create_lit_module(cfg)

    callbacks = [
        build_checkpoint(cfg),
        build_early_stopping(cfg),
    ]
    loggers = build_loggers(cfg)
    trainer_cfg = cfg.trainer
    trainer = L.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        precision=trainer_cfg.get("precision", 32),
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=save_dir,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    ckpt_path = "best" if cfg.callbacks.get("save_top_k", 0) else None
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
