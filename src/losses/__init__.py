import torch
import torch.nn as nn


class SmapeLoss(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        denom = preds.abs() + targets.abs()
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        loss = (preds - targets).abs() / denom
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(loss.dtype)
            loss = loss * mask
        return 200.0 * loss.mean()

_LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "ce": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
    "smape": SmapeLoss,
}


def build_loss(cfg: dict) -> nn.Module:
    name = cfg.get("name", "cross_entropy").lower()
    builder = _LOSS_REGISTRY.get(name)
    if not builder:
        raise ValueError(f"不支持的损失: {name}")
    return builder()


__all__ = ["SmapeLoss", "build_loss"]
