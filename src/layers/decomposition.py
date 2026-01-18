import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需要为奇数")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.avg(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
