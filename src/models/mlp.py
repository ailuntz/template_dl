import torch.nn as nn


class MLP(nn.Module):
    """简单MLP模型"""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
