import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码（Transformer标准）"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEmbedding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    """Token嵌入层"""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class DataEmbedding(nn.Module):
    """数据嵌入（特征投影 + 位置编码）"""

    def __init__(self, input_dim: int, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)
