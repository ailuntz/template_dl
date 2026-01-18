from typing import Any, Dict

import lightning as L

from src.losses import build_loss
from src.metrics import mae, mse, rmse
from src.models import create_model
from src.optimizers import build_optimizer


class ExpImputation(L.LightningModule):
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
        self.seq_len = int(self.task_cfg["seq_len"])
        self.pred_len = int(self.task_cfg.get("pred_len", self.seq_len))
        if self.seq_len != self.pred_len:
            raise ValueError("插补任务需要 pred_len == seq_len")

        self.save_hyperparameters({"cfg": cfg})

    def _step(self, batch, stage: str):
        if len(batch) == 3:
            x, y, mask = batch
            x_mark = None
        elif len(batch) == 4:
            x, y, x_mark, mask = batch
        else:
            raise ValueError("不支持的 batch 格式")

        inp = x.masked_fill(mask > 0.5, 0.0)
        pred, y = self.model.forward_imputation(inp, x_mark, y)
        features = str((self.dataset_cfg.get("schema") or {}).get("features") or "")
        if features == "MS":
            pred = pred[..., -1:]
            y = y[..., -1:]
            mask = mask[..., -1:]
        mask = mask > 0.5
        loss = self.criterion(pred[mask], y[mask])
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae(pred[mask], y[mask]), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse(pred[mask], y[mask]), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_rmse", rmse(pred[mask], y[mask]), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if len(batch) == 3:
            x, _, mask = batch
            x_mark = None
        elif len(batch) == 4:
            x, _, x_mark, mask = batch
        else:
            raise ValueError("不支持的 batch 格式")
        inp = x.masked_fill(mask > 0.5, 0.0)
        pred, _ = self.model.forward_imputation(inp, x_mark, None)
        return pred

    def configure_optimizers(self):
        return build_optimizer(self.optim_cfg, self.parameters())


__all__ = ["ExpImputation"]
