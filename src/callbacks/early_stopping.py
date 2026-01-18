from lightning.pytorch.callbacks import EarlyStopping


def build_early_stopping(cfg: dict) -> EarlyStopping:
    cb_cfg = cfg["callbacks"]["early_stopping"]
    return EarlyStopping(
        monitor=cb_cfg["monitor"],
        mode=cb_cfg["mode"],
        patience=cb_cfg.get("patience", 5),
        min_delta=cb_cfg.get("min_delta", 0.0),
        verbose=False,
    )
