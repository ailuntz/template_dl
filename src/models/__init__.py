from types import SimpleNamespace

from src.models.autoformer import Autoformer
from src.models.dlinear import DLinear
from src.models.tsmixer import TSMixer

_MODEL_REGISTRY = {
    "autoformer": Autoformer,
    "dlinear": DLinear,
    "tsmixer": TSMixer,
}


def create_model(cfg: dict):
    model_cfg = dict(cfg["model"][cfg["model_name"]])
    dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
    schema = dataset_cfg.get("schema") or {}
    if "target_index" in schema:
        model_cfg["target_index"] = int(schema["target_index"])
    if not model_cfg.get("task_name"):
        raise ValueError("缺少 model.task_name")
    name = str(model_cfg.get("name") or cfg["model_name"]).lower()
    builder = _MODEL_REGISTRY.get(name)
    if not builder:
        raise ValueError(f"不支持的模型类型: {name}")
    return builder(SimpleNamespace(**model_cfg))


__all__ = [
    "Autoformer",
    "DLinear",
    "TSMixer",
    "create_model",
]
