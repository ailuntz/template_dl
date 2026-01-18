from typing import Any, Dict

import lightning as L
import numpy as np
import torch

from src.losses import build_loss
from src.metrics import mae, mse
from src.models import create_model
from src.lit_modules.lit_utils_task import adjustment, binary_metrics
from src.optimizers import build_optimizer


class ExpAnomaly(L.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
        model_cfg = cfg["model"][cfg["model_name"]]
        task_name = str(model_cfg.get("task_name") or "").lower()
        task_cfg = (cfg.get("task") or {}).get(task_name)
        if task_cfg is None:
            raise ValueError("缺少 task 配置")
        self.task_cfg = task_cfg
        self.loss_cfg = cfg["loss"][cfg["loss_name"]]
        self.optim_cfg = cfg["optim"][cfg["optim_name"]]
        self.model = create_model(cfg)
        self.criterion = build_loss(self.loss_cfg)
        self.anomaly_ratio = float(self.task_cfg.get("anomaly_ratio", 0.25))

        self._train_energy = []
        self._test_energy = []
        self._test_labels = []

        self.save_hyperparameters({"cfg": cfg})

    def _step(self, batch, stage: str):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y, _, _ = batch
        pred, y = self.model.forward_anomaly(x, y)
        loss = self.criterion(pred, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
            labels = None
        else:
            x, y, labels, _ = batch
        pred, y = self.model.forward_anomaly(x, y)
        loss = self.criterion(pred, y)
        stage = "test"
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        score = torch.mean((pred - y) ** 2, dim=-1)
        self._test_energy.append(score.detach().cpu())
        if labels is not None:
            self._test_labels.append(labels.detach().cpu())
        return loss

    def on_test_start(self) -> None:
        self._train_energy = []
        self._test_energy = []
        self._test_labels = []
        dm = self.trainer.datamodule
        if getattr(dm, "train_set", None) is None:
            dm.setup("fit")
        loader = dm.train_dataloader()
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                pred, x = self.model.forward_anomaly(x, x)
                score = torch.mean((pred - x) ** 2, dim=-1)
                self._train_energy.append(score.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_energy:
            return
        train_energy = torch.cat(self._train_energy, dim=0).reshape(-1).numpy()
        test_energy = torch.cat(self._test_energy, dim=0).reshape(-1).numpy()
        if self._test_labels:
            labels = torch.cat(self._test_labels, dim=0).reshape(-1).numpy()
            gt = (labels > 0.5).astype(int)
            combined = np.concatenate([train_energy, test_energy], axis=0)
            threshold = np.percentile(combined, 100 - self.anomaly_ratio)
            pred = (test_energy > threshold).astype(int)
            gt, pred = adjustment(gt, pred)
            precision, recall, f1, acc = binary_metrics(pred, gt)
            stage = "test"
            self.log(f"{stage}_precision", float(precision), on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_recall", float(recall), on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x = batch[0]
        return self.model(x, None, None, None)

    def configure_optimizers(self):
        return build_optimizer(self.optim_cfg, self.parameters())


__all__ = ["ExpAnomaly"]
