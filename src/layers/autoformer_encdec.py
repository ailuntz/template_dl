import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.decomposition import SeriesDecomp
from src.layers.autoformer_conv import SafeCircularConv1d, SafePointwiseConv1d


class MyLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class SeriesDecompLayer(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)

    def forward(self, x: torch.Tensor):
        return self.decomp(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model: int, d_ff: int = None, moving_avg: int = 25,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = SafePointwiseConv1d(d_model, d_ff, bias=False)
        self.conv2 = SafePointwiseConv1d(d_ff, d_model, bias=False)
        self.decomp1 = SeriesDecompLayer(moving_avg)
        self.decomp2 = SeriesDecompLayer(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1).contiguous())))
        y = self.dropout(self.conv2(y).transpose(-1, 1).contiguous())
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model: int, c_out: int, d_ff: int = None,
                 moving_avg: int = 25, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = SafePointwiseConv1d(d_model, d_ff, bias=False)
        self.conv2 = SafePointwiseConv1d(d_ff, d_model, bias=False)
        self.decomp1 = SeriesDecompLayer(moving_avg)
        self.decomp2 = SeriesDecompLayer(moving_avg)
        self.decomp3 = SeriesDecompLayer(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = SafeCircularConv1d(d_model, c_out, kernel_size=3, bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask=None, cross_mask=None, trend=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1).contiguous())))
        y = self.dropout(self.conv2(y).transpose(-1, 1).contiguous())
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = residual_trend.permute(0, 2, 1).contiguous()
        residual_trend = self.projection(residual_trend).transpose(1, 2).contiguous()
        return x, residual_trend


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend
