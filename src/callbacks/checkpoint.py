from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint


def build_checkpoint(cfg: dict) -> ModelCheckpoint:
    cb_cfg = cfg["callbacks"]["model_checkpoint"]

    save_dir = Path(cfg["paths"]["experiments_dir"]) / cfg["experiment_name"]
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor=cb_cfg["monitor"],
        mode=cb_cfg["mode"],
        save_top_k=cb_cfg.get("save_top_k", 1),
        save_last=cb_cfg.get("save_last", True),
        every_n_epochs=cb_cfg.get("every_n_epochs"),
        every_n_train_steps=cb_cfg.get("every_n_train_steps"),
        train_time_interval=cb_cfg.get("train_time_interval"),
        save_weights_only=cb_cfg.get("save_weights_only", False),
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
