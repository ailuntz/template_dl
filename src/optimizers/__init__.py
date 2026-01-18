import torch


def _build_adam(params, cfg):
    return torch.optim.Adam(
        params,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


def _build_sgd(params, cfg):
    return torch.optim.SGD(
        params,
        lr=float(cfg.get("lr", 1e-3)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


_OPTIM_REGISTRY = {
    "adam": _build_adam,
    "sgd": _build_sgd,
}


def build_optimizer(cfg: dict, params):
    name = str(cfg.get("name", "adam")).lower()
    builder = _OPTIM_REGISTRY.get(name)
    if not builder:
        raise ValueError(f"不支持的优化器: {name}")
    optimizer = builder(params, cfg)

    lradj = str(cfg.get("lradj", "")).lower()
    if lradj == "type1":
        gamma = float(cfg.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
    return optimizer


__all__ = ["build_optimizer"]
