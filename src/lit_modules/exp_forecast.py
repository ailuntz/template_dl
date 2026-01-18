from typing import Any, Dict

import lightning as L

from src.losses import build_loss
from src.metrics import mae, mse, rmse
from src.models import create_model
from src.optimizers import build_optimizer


class ExpForecast(L.LightningModule):
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
        self.save_hyperparameters({"cfg": cfg})

    def _step(self, batch, stage: str):
        use_time_features = bool(self.task_cfg.get("time_features", False))
        mask = None
        if len(batch) == 2:
            x, y = batch
            x_mark = y_mark = None
        elif len(batch) == 4:
            x, y, x_mark, y_mark = batch
            if not use_time_features:
                mask = y_mark
                x_mark = y_mark = None
        else:
            raise ValueError("不支持的 batch 格式")
        pred, y = self.model.forward_forecast(x, x_mark, y, y_mark)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.size(1) != y.size(1):
                mask = mask[:, -y.size(1):, :]
            if mask.size(-1) != y.size(-1):
                mask = mask[..., :y.size(-1)]
            mask = mask > 0.5
            loss = self.criterion(pred[mask], y[mask])
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_mae", mae(pred[mask], y[mask]), on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse", mse(pred[mask], y[mask]), on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_rmse", rmse(pred[mask], y[mask]), on_step=False, on_epoch=True, prog_bar=False)
            return loss
        loss = self.criterion(pred, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_rmse", rmse(pred, y), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        use_time_features = bool(self.task_cfg.get("time_features", False))
        if len(batch) == 2:
            x, y = batch
            x_mark = y_mark = None
        elif len(batch) == 4:
            x, y, x_mark, y_mark = batch
            if not use_time_features:
                x_mark = y_mark = None
        else:
            raise ValueError("不支持的 batch 格式")
        pred, _ = self.model.forward_forecast(x, x_mark, y, y_mark)
        return pred

    def configure_optimizers(self):
        return build_optimizer(self.optim_cfg, self.parameters())


__all__ = ["ExpForecast"]
