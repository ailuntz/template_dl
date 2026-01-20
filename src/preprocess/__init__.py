from src.preprocess.preprocesstest import preprocesstest

_PREPROCESS_REGISTRY = {
    "mock": preprocesstest,
}


def create_preprocess(cfg: dict):
    name = str(cfg.get("preprocess_name") or "").lower()
    if not name:
        raise ValueError("缺少 preprocess_name")
    preprocess_root = cfg.get("preprocess") or {}
    if preprocess_root.get(name) is None:
        raise ValueError(f"缺少配置: preprocess.{name}")
    builder = _PREPROCESS_REGISTRY.get(name)
    if not builder:
        raise ValueError(f"不支持的预处理类型: {name}")
    return builder(preprocess_root[name])


__all__ = [
    "preprocesstest",
    "create_preprocess",
]
