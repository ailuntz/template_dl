import torch.nn as nn

from src.layers.attention import MultiHeadAttention
from src.layers.embed import DataEmbedding


class TransformerEncoder(nn.Module):
    """简单的Transformer编码器示例（演示如何使用layers）"""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        num_classes: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = DataEmbedding(input_dim, d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)
        x = self.fc(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
