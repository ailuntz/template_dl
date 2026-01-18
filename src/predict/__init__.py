import torch

from src.predict.classification import save_classification_preds
from src.predict.dlinear import save_dlinear_preds

_SAVER_REGISTRY = {
    "forecast": save_dlinear_preds,
    "imputation": save_dlinear_preds,
    "anomaly": save_dlinear_preds,
    "classification": save_classification_preds,
}


def _merge_preds(preds):
    if not preds:
        raise ValueError("预测结果为空")
    if isinstance(preds[0], list):
        preds = preds[0]
    return torch.cat([p.detach().cpu() for p in preds], dim=0)


def save_preds(cfg: dict, preds, save_dir):
    model_cfg = cfg["model"][cfg["model_name"]]
    task_name = str(model_cfg.get("task_name") or "").lower()
    saver = _SAVER_REGISTRY.get(task_name)
    if not saver:
        raise ValueError(f"不支持的保存类型: {task_name}")
    merged = _merge_preds(preds)
    return saver(merged.numpy(), save_dir)


__all__ = ["save_preds"]
