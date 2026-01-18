import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, seq_len: int, channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x


class TSMixer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = str(configs.task_name).lower()
        if self.task_name != "forecast":
            raise ValueError("TSMixer 仅支持 forecast 任务")
        self.layer = int(configs.e_layers)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        hidden_dim = int(configs.d_model)
        channels = int(configs.enc_in)
        dropout = float(configs.dropout)
        self.target_index = int(getattr(configs, "target_index", 0))
        self.blocks = nn.ModuleList(
            [ResBlock(self.seq_len, channels, hidden_dim, dropout) for _ in range(self.layer)]
        )
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forecast(self, x_enc):
        for block in self.blocks:
            x_enc = block(x_enc)
        return self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]

    def _align_pred(self, pred, y):
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

    def forward_classification(self, x_enc, x_mark_enc=None, y=None):
        raise ValueError("TSMixer 暂不支持 classification 任务")
