import math

import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag: bool = True, factor: float = 1.0, scale=None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = min(length, max(1, int(self.factor * math.log(length))))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        init_index = init_index.to(values.device)
        top_k = min(length, max(1, int(self.factor * math.log(length))))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                1, head, channel, length
            )
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            )
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        bsz, length, heads, _ = queries.shape
        _, src_len, _, _ = values.shape
        if length > src_len:
            zeros = torch.zeros_like(queries[:, :(length - src_len), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :length, :, :]
            keys = keys[:, :length, :, :]

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        if self.training:
            out = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr)
        else:
            out = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr)
        out = out.permute(0, 3, 1, 2).contiguous()

        if self.output_attention:
            return out.contiguous(), corr.permute(0, 3, 1, 2)
        return out.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model: int, n_heads: int, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        bsz, length, _ = queries.shape
        _, src_len, _ = keys.shape
        heads = self.n_heads
        queries = self.query_projection(queries).reshape(bsz, length, heads, -1)
        keys = self.key_projection(keys).reshape(bsz, src_len, heads, -1)
        values = self.value_projection(values).reshape(bsz, src_len, heads, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.reshape(bsz, length, -1)
        return self.out_projection(out), attn
