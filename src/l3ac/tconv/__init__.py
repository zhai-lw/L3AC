import torch
from torch import nn

from ..layers import ChannelNorm, Conv1d, Linear, GRN, Snake1d
from . import base


class V3FirstBlock(base.BaseBlock):  # (1, 5, 11, 21, 45)
    def __init__(self, target_dim, conv_kernels=(7, 7, 7, 7, 7), pool_kernels=(1, 5, 11, 21, 45), dilation_rate=7):
        h_dim = len(pool_kernels) * 4
        super().__init__(h_dim, conv_kernels, pool_kernels, dilation_rate=dilation_rate)
        self.conv_1 = Conv1d(h_dim, h_dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.conv_2 = Conv1d(h_dim * 4 + 1, target_dim, kernel_size=1)

    def forward(self, x):
        h = super().forward(x)
        h = self.conv_1(h)
        h = self.act(h)
        y = torch.cat([h, x], dim=1)
        y = self.conv_2(y)
        return y


FirstBlock = lambda dim: (
    V3FirstBlock(dim, conv_kernels=(7, 7, 7, 7, 7), pool_kernels=(1, 5, 11, 21, 45), dilation_rate=99)
)


class EnhanceBlock(base.BaseBlock):
    def __init__(self, dim):
        super().__init__(4, conv_kernels=(7, 7, 7, 7), pool_kernels=(1, 3, 5, 9))
        self.dim = dim
        self.merge_layer = nn.Sequential(
            # nn.LeakyReLU(),  # ! if active or use InstanceNorm1d
            nn.InstanceNorm1d(4, affine=True),
            nn.Conv1d(4, self.dim, kernel_size=1)
        )

    def forward(self, x):
        xi = x[:, :1, :]
        yi = super().forward(xi)
        y = self.merge_layer(yi)
        return x + y * x
