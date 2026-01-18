from src.data.datamodule import BaseDataModule
from src.data.ett_small_forecast import ETTSmallForecastDataset
from src.data.ett_small_imputation import ETTSmallImputationDataset
from src.data.illness_forecast import IllnessForecastDataset
from src.data.m4_forecast import M4ForecastDataset
from src.data.swat_anomaly import SWaTAnomalyDataset
from src.data.spoken_arabic_digits_classification import SpokenArabicDigitsDataset

REGISTRY_DATASET_FORECAST = {
    "ett-small": ETTSmallForecastDataset,
    "illness": IllnessForecastDataset,
    "m4": M4ForecastDataset,
}
REGISTRY_DATASET_IMPUTATION = {
    "ett-small": ETTSmallImputationDataset,
}
REGISTRY_DATASET_CLASSIFICATION = {
    "spokenarabicdigits": SpokenArabicDigitsDataset,
}
REGISTRY_DATASET_ANOMALY = {
    "swat": SWaTAnomalyDataset,
}

_DATASET_REGISTRY = {
    "forecast": REGISTRY_DATASET_FORECAST,
    "imputation": REGISTRY_DATASET_IMPUTATION,
    "classification": REGISTRY_DATASET_CLASSIFICATION,
    "anomaly": REGISTRY_DATASET_ANOMALY,
}


def create_datamodule(cfg: dict):
    model_cfg = cfg["model"][cfg["model_name"]]
    task_name = str(model_cfg.get("task_name") or "").lower()
    if not task_name:
        raise ValueError("缺少 task_name")
    task_name = str(task_name).lower()
    registry = _DATASET_REGISTRY.get(task_name)
    if not registry:
        raise ValueError(f"不支持的任务类型: {task_name}")
    data_cfg = cfg["dataset"][cfg["dataset_name"]]
    name = str(data_cfg.get("name") or cfg["dataset_name"]).lower()
    dataset_cls = registry.get(name)
    if not dataset_cls:
        raise ValueError(f"任务 {task_name} 不支持的数据类型: {name}")
    return BaseDataModule(cfg, dataset_cls)


__all__ = [
    "ETTSmallForecastDataset",
    "ETTSmallImputationDataset",
    "IllnessForecastDataset",
    "M4ForecastDataset",
    "SWaTAnomalyDataset",
    "SpokenArabicDigitsDataset",
    "BaseDataModule",
    "create_datamodule",
]
