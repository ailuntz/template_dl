import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeCircularConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        init: str = "kaiming_uniform",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 需要为奇数")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        weight = torch.empty(out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        init = init.lower()
        if init == "kaiming_normal":
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="leaky_relu")
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def _conv_cpu(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad), mode="circular")
        return F.conv1d(x, self.weight, self.bias, stride=1, padding=0)

    def _conv_mps(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        shifts = [pad - k for k in range(self.kernel_size)]
        xs = [torch.roll(x, shifts=shift, dims=-1) for shift in shifts]
        stacked = torch.stack(xs, dim=-1)
        out = torch.einsum("bclk,ock->bol", stacked, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("SafeCircularConv1d 需要输入 (B, C, L)")
        if x.is_mps:
            return self._conv_mps(x)
        return self._conv_cpu(x)


class SafePointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_mps:
            weight = self.conv.weight.squeeze(-1)
            out = torch.einsum("bcl,oc->bol", x, weight)
            if self.conv.bias is not None:
                out = out + self.conv.bias.view(1, -1, 1)
            return out
        return self.conv(x)
