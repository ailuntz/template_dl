import argparse
from pathlib import Path

import lightning as L

from src.data import create_datamodule
from src.lit_modules import create_lit_module
from src.predict import save_preds
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
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        logger=False,
    )
    preds = trainer.predict(model, datamodule=datamodule)
    out_path = save_preds(cfg, preds, save_dir)
    print(f"预测批次数: {len(preds)}")
    print(f"预测结果已保存: {out_path}")


if __name__ == "__main__":
    main()
