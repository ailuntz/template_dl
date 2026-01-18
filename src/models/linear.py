import torch.nn as nn


class Linear(nn.Module):
    """线性模型。"""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.head(x)
