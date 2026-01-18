from src.lit_modules.exp_anomaly import ExpAnomaly
from src.lit_modules.exp_classification import ExpClassification
from src.lit_modules.exp_forecast import ExpForecast
from src.lit_modules.exp_imputation import ExpImputation

_LIT_REGISTRY = {
    "forecast": ExpForecast,
    "imputation": ExpImputation,
    "classification": ExpClassification,
    "anomaly": ExpAnomaly,
}


def create_lit_module(cfg: dict, ckpt_path=None, weights_only: bool = False):
    model_cfg = cfg["model"][cfg["model_name"]]
    task_name = str(model_cfg.get("task_name") or "").lower()
    if not task_name:
        raise ValueError("缺少 task_name")
    registry = _LIT_REGISTRY.get(task_name)
    if not registry:
        raise ValueError(f"不支持的任务类型: {task_name}")
    builder = registry
    if ckpt_path:
        return builder.load_from_checkpoint(ckpt_path, cfg=cfg, weights_only=weights_only)
    return builder(cfg)


__all__ = [
    "ExpForecast",
    "ExpImputation",
    "ExpAnomaly",
    "ExpClassification",
    "create_lit_module",
]
