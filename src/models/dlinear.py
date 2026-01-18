import torch
import torch.nn as nn

from src.layers.decomposition import SeriesDecomp


class DLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = str(configs.task_name).lower()
        self.seq_len = int(configs.seq_len)
        if self.task_name in {"classification", "anomaly", "imputation"}:
            self.pred_len = self.seq_len
        else:
            self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(configs.individual)
        self.decomp = SeriesDecomp(int(configs.moving_avg))
        self.target_index = int(getattr(configs, "target_index", 0))

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.channels):
                self.linear_seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.linear_trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        if self.task_name == "classification":
            if int(configs.num_class) <= 0:
                raise ValueError("DLinear 分类需要 num_class")
            self.projection = nn.Linear(self.channels * self.seq_len, int(configs.num_class))

        self._init_weights()

    def _init_weights(self) -> None:
        weight = (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        if self.individual:
            for idx in range(self.channels):
                self.linear_seasonal[idx].weight = nn.Parameter(weight.clone())
                self.linear_trend[idx].weight = nn.Parameter(weight.clone())
        else:
            self.linear_seasonal.weight = nn.Parameter(weight.clone())
            self.linear_trend.weight = nn.Parameter(weight.clone())

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(
                [seasonal.size(0), seasonal.size(1), self.pred_len],
                dtype=seasonal.dtype,
                device=seasonal.device,
            )
            trend_out = torch.zeros(
                [trend.size(0), trend.size(1), self.pred_len],
                dtype=trend.dtype,
                device=trend.device,
            )
            for idx in range(self.channels):
                seasonal_out[:, idx, :] = self.linear_seasonal[idx](seasonal[:, idx, :])
                trend_out[:, idx, :] = self.linear_trend[idx](trend[:, idx, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        out = seasonal_out + trend_out
        return out.permute(0, 2, 1)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_enc)

    def imputation(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_enc)

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "forecast":
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc)
        if self.task_name == "anomaly":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc)
        raise ValueError(f"不支持的任务类型: {self.task_name}")

    def _align_pred(self, pred: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        if y is None:
            return pred
        if pred.size(1) != y.size(1):
            pred = pred[:, -y.size(1):, :]
        if pred.size(-1) != y.size(-1):
            pred = pred[..., self.target_index:self.target_index + 1]
        return pred

    def forward_forecast(self, x_enc, x_mark_enc=None, y=None, y_mark_dec=None):
        pred = self.forecast(x_enc)
        pred = pred[:, -self.pred_len:, :]
        if y is not None and self.pred_len > 0:
            y = y[:, -self.pred_len:, :]
        pred = self._align_pred(pred, y)
        return pred, y

    def forward_imputation(self, x_enc, x_mark_enc=None, y=None):
        pred = self.imputation(x_enc)
        pred = self._align_pred(pred, y)
        return pred, y

    def forward_anomaly(self, x_enc, y=None):
        pred = self.anomaly_detection(x_enc)
        pred = self._align_pred(pred, y)
        return pred, y

    def forward_classification(self, x_enc, x_mark_enc=None, y=None):
        logits = self.classification(x_enc)
        return logits, y
