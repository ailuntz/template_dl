from typing import Any, Dict

import lightning as L
import torch

from src.losses import build_loss
from src.metrics import accuracy, classification_prf
from src.models import create_model
from src.optimizers import build_optimizer


class ExpClassification(L.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
        self.loss_cfg = cfg["loss"][cfg["loss_name"]]
        self.optim_cfg = cfg["optim"][cfg["optim_name"]]
        self.model = create_model(cfg)
        self.criterion = build_loss(self.loss_cfg)
        self.max_seq_len = int(self.dataset_cfg.get("max_seq_len", 0))
        self.num_features = int(self.dataset_cfg.get("num_features", 0))
        self.save_hyperparameters({"cfg": cfg})

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and self.max_seq_len > 0 and self.num_features > 0:
            if x.size(1) == self.max_seq_len * self.num_features:
                return x.reshape(x.size(0), self.max_seq_len, self.num_features)
        return x

    def _step(self, batch, stage: str):
        if len(batch) == 2:
            x, y = batch
            x_mark = None
        else:
            x, y, x_mark = batch
        x = self._reshape_input(x)
        if hasattr(self.model, "forward_classification"):
            logits, _ = self.model.forward_classification(x, x_mark, y)
        else:
            logits = self.model(x, x_mark, None, None)
        loss = self.criterion(logits, y)
        acc = accuracy(logits, y)
        precision, recall, f1 = classification_prf(logits, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_recall", recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_f1", f1, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if len(batch) == 2:
            x, _ = batch
            x_mark = None
        else:
            x, _, x_mark = batch
        x = self._reshape_input(x)
        if hasattr(self.model, "forward_classification"):
            logits, _ = self.model.forward_classification(x, x_mark, None)
        else:
            logits = self.model(x, x_mark, None, None)
        return torch.softmax(logits, dim=-1)

    def configure_optimizers(self):
        return build_optimizer(self.optim_cfg, self.parameters())


__all__ = ["ExpClassification"]
