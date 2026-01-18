import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.autoformer_autocorr import AutoCorrelation, AutoCorrelationLayer
from src.layers.autoformer_embed import DataEmbeddingWoPos
from src.layers.autoformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer, MyLayerNorm
from src.layers.decomposition import SeriesDecomp


class Autoformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = str(configs.task_name).lower()
        self.seq_len = int(configs.seq_len)
        self.label_len = int(configs.label_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.dec_in = int(configs.dec_in)
        self.c_out = int(configs.c_out)
        self.decomp = SeriesDecomp(int(configs.moving_avg))
        self.target_index = int(getattr(configs, "target_index", 0))

        self.enc_embedding = DataEmbeddingWoPos(
            self.enc_in,
            int(configs.d_model),
            configs.embed,
            configs.freq,
            float(configs.dropout),
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, float(configs.factor), attention_dropout=float(configs.dropout), output_attention=False),
                        int(configs.d_model),
                        int(configs.n_heads),
                    ),
                    int(configs.d_model),
                    int(configs.d_ff),
                    moving_avg=int(configs.moving_avg),
                    dropout=float(configs.dropout),
                    activation=str(configs.activation),
                )
                for _ in range(int(configs.e_layers))
            ],
            norm_layer=MyLayerNorm(int(configs.d_model)),
        )

        if self.task_name == "forecast":
            self.dec_embedding = DataEmbeddingWoPos(
                self.dec_in,
                int(configs.d_model),
                configs.embed,
                configs.freq,
                float(configs.dropout),
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, float(configs.factor), attention_dropout=float(configs.dropout), output_attention=False),
                            int(configs.d_model),
                            int(configs.n_heads),
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, float(configs.factor), attention_dropout=float(configs.dropout), output_attention=False),
                            int(configs.d_model),
                            int(configs.n_heads),
                        ),
                        int(configs.d_model),
                        self.c_out,
                        int(configs.d_ff),
                        moving_avg=int(configs.moving_avg),
                        dropout=float(configs.dropout),
                        activation=str(configs.activation),
                    )
                    for _ in range(int(configs.d_layers))
                ],
                norm_layer=MyLayerNorm(int(configs.d_model)),
                projection=nn.Linear(int(configs.d_model), self.c_out, bias=True),
            )
        elif self.task_name in {"imputation", "anomaly"}:
            self.projection = nn.Linear(int(configs.d_model), self.c_out, bias=True)
        elif self.task_name == "classification":
            if int(configs.num_class) <= 0:
                raise ValueError("Autoformer 分类需要 num_class")
            self.act = F.gelu
            self.dropout = nn.Dropout(float(configs.dropout))
            self.projection = nn.Linear(int(configs.d_model) * self.seq_len, int(configs.num_class))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            (x_dec.shape[0], self.pred_len, x_dec.shape[2]),
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        if self.label_len > 0:
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        else:
            trend_init = mean
            seasonal_init = zeros

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        return trend_part + seasonal_part

    def imputation(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        if x_mark_enc is not None:
            if x_mark_enc.dim() == 2:
                x_mark_enc = x_mark_enc.unsqueeze(-1)
            output = output * x_mark_enc
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc)
        if self.task_name == "anomaly":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise ValueError(f"不支持的任务类型: {self.task_name}")

    def _align_pred(self, pred: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        if y is None:
            return pred
        if pred.size(1) != y.size(1):
            pred = pred[:, -y.size(1):, :]
        if pred.size(-1) != y.size(-1):
            pred = pred[..., self.target_index:self.target_index + 1]
        return pred

    def _build_dec_inp(self, x_enc: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        if y is not None:
            past = y[:, :self.label_len, :] if self.label_len > 0 else y[:, :0, :]
        else:
            past = x_enc[:, -self.label_len:, :] if self.label_len > 0 else x_enc[:, :0, :]
        zeros = torch.zeros(
            (x_enc.size(0), self.pred_len, x_enc.size(-1)),
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        return torch.cat([past, zeros], dim=1)

    def forward_forecast(self, x_enc, x_mark_enc=None, y=None, y_mark_dec=None):
        dec_inp = self._build_dec_inp(x_enc, y)
        pred = self.forecast(x_enc, x_mark_enc, dec_inp, y_mark_dec)
        pred = pred[:, -self.pred_len:, :]
        if y is not None and self.pred_len > 0:
            y = y[:, -self.pred_len:, :]
        pred = self._align_pred(pred, y)
        return pred, y

    def forward_imputation(self, x_enc, x_mark_enc=None, y=None):
        pred = self.imputation(x_enc, x_mark_enc)
        pred = self._align_pred(pred, y)
        return pred, y

    def forward_anomaly(self, x_enc, y=None):
        pred = self.anomaly_detection(x_enc)
        pred = self._align_pred(pred, y)
        return pred, y

    def forward_classification(self, x_enc, x_mark_enc=None, y=None):
        logits = self.classification(x_enc, x_mark_enc)
        return logits, y
