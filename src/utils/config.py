from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(path: Path, overrides: Optional[Sequence[Path]], cli_overrides: Sequence[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        override_cfgs = [OmegaConf.load(p) for p in overrides]
        cfg = OmegaConf.merge(cfg, *override_cfgs)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    OmegaConf.set_struct(cfg, False)

    base_name = cfg.get("experiment_name") or "exp"
    overwrite = cfg.get("experiment_overwrite")
    if overwrite is None:
        overwrite = True
    if not overwrite:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{ts}"
    cfg["experiment_name"] = base_name

    model_name = cfg.get("model_name")
    if not model_name:
        raise ValueError("缺少 model_name")
    model_root = cfg.get("model")
    if model_root is None or model_root.get(model_name) is None:
        raise ValueError(f"缺少配置: model.{model_name}")
    model_cfg = model_root[model_name]
    task_name = str(model_cfg.get("task_name") or "").lower()
    if not task_name:
        raise ValueError("缺少 task_name")
    if task_name not in {"forecast", "imputation", "classification", "anomaly"}:
        raise ValueError(f"不支持的任务类型: {task_name}")

    dataset_name = cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("缺少 dataset_name")
    dataset_root = cfg.get("dataset")
    if dataset_root is None or dataset_root.get(dataset_name) is None:
        raise ValueError(f"缺少配置: dataset.{dataset_name}")

    loss_name = cfg.get("loss_name")
    if not loss_name:
        raise ValueError("缺少 loss_name")
    loss_root = cfg.get("loss")
    if loss_root is None or loss_root.get(loss_name) is None:
        raise ValueError(f"缺少配置: loss.{loss_name}")

    optim_name = cfg.get("optim_name")
    if not optim_name:
        raise ValueError("缺少 optim_name")
    optim_root = cfg.get("optim")
    if optim_root is None or optim_root.get(optim_name) is None:
        raise ValueError(f"缺少配置: optim.{optim_name}")

    data_cfg = cfg.get("data")
    if not data_cfg or data_cfg.get("loader") is None:
        raise ValueError("缺少 data.loader")

    if task_name in {"forecast", "imputation", "anomaly"}:
        task_cfg = (cfg.get("task") or {}).get(task_name)
        if task_cfg is None:
            raise ValueError("缺少 task 配置")
        if data_cfg.get("split") is None:
            raise ValueError("缺少 data.split")

    OmegaConf.set_struct(cfg, True)
    return cfg
