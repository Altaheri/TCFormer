# Standard library
import numpy as np

# Third-party libraries
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# Local application-specific imports
from .classification_module import ClassificationModule
from .modules import CausalConv1d, Conv2dWithConstraint, LinearWithConstraint
from utils.weight_initialization import glorot_weight_zero_bias


class _ConvBlock(nn.Module):
    def __init__(self, F1: int = 16, kernel_length: int = 64, pool_length: int = 8,
                 D: int = 2, in_channels: int = 22, dropout: float = 0.3):
        super(_ConvBlock, self).__init__()
        self.rearrange_input = Rearrange("b c seq -> b 1 c seq")
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length),
                                       padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.spat_conv = Conv2dWithConstraint(F1, F1 * D, (in_channels, 1), bias=False,
                                              groups=F1, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.nonlinearity1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_length))
        self.drop1 = nn.Dropout(dropout)

        self.conv = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.nonlinearity2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 7))
        self.drop2 = nn.Dropout(dropout)

        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.spat_conv(x)
        x = self.bn2(x)
        x = self.nonlinearity1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv(x)
        x = self.bn3(x)
        x = self.nonlinearity2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x


class _AttentionBlock(nn.Module):
    def __init__(self, d_model, key_dim=8, n_head=2, dropout=0.5):
        super(_AttentionBlock, self).__init__()
        self.n_head = n_head

        self.w_qs = nn.Linear(d_model, n_head * key_dim)
        self.w_ks = nn.Linear(d_model, n_head * key_dim)
        self.w_vs = nn.Linear(d_model, n_head * key_dim)

        self.fc = nn.Linear(n_head * key_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        glorot_weight_zero_bias(self)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        q = rearrange(self.w_qs(x), 'b l (head k) -> head b l k', head=self.n_head)
        k = rearrange(self.w_ks(x), 'b t (head k) -> head b t k', head=self.n_head)
        v = rearrange(self.w_vs(x), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk, hbtk -> hblt', [q, k]) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=3)

        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = output + residual

        return output


class TCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 dropout: float = 0.3):
        super(TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.nonlinearity3 = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, input):
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        x = self.nonlinearity3(input + x)
        return x


class TCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 dropout: float = 0.3):
        super(TCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(kernel_length, n_filters, dilation, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ATCBlock(nn.Module):
    def __init__(self, d_model: int = 32, key_dim: int = 8, n_head: int = 2,
                 dropout_attn: float = 0.3, tcn_depth: int = 2, kernel_length: int = 4,
                 dropout_tcn: float = 0.3, n_classes: int = 4):
        super(ATCBlock, self).__init__()
        self.attention_block = _AttentionBlock(d_model, key_dim, n_head, dropout_attn)
        self.rearrange = Rearrange("b seq c -> b c seq")
        self.tcn = TCN(tcn_depth, kernel_length, d_model, dropout_tcn)
        self.linear = LinearWithConstraint(d_model, n_classes, max_norm=0.25)

    def forward(self, x):
        x = self.attention_block(x)
        x = self.rearrange(x)
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x


class ATCNetModule(nn.Module):
    def __init__(
            self,
            n_channels: int = 22,
            n_classes: int = 4,
            F1: int = 16,
            kernel_length_conv: int = 64,
            pool_length: int = 8,
            D: int = 2,
            dropout_conv: float = 0.3,
            d_model: int = 32,
            key_dim: int = 8,
            n_head: int = 2,
            dropout_attn: float = 0.5,
            tcn_depth: int = 2,
            kernel_length_tcn: int = 4,
            dropout_tcn: float = 0.3,
            n_windows: int = 5,
    ):
        super().__init__()
        self.conv_block = _ConvBlock(F1, kernel_length_conv, pool_length, D,
                                     n_channels, dropout_conv)
        self.rearrange = Rearrange("b c 1 seq -> b seq c")

        self.atc_blocks = nn.ModuleList([
            ATCBlock(d_model, key_dim, n_head, dropout_attn, tcn_depth,
                     kernel_length_tcn, dropout_tcn, n_classes)
            for _ in range(n_windows)
        ])
        self.n_windows = n_windows
        self.n_classes = n_classes

    def forward(self, x):
        x = self.conv_block(x)
        x = self.rearrange(x)

        bs, seq_len, _ = x.shape
        blk_output = torch.zeros(bs, self.n_classes, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.atc_blocks):
            blk_output = blk_output + blk(x[:, i:(seq_len-self.n_windows+i+1), :])

        blk_output = blk_output / self.n_windows

        return blk_output


class ATCNet(ClassificationModule):
    def __init__(
            self,
            n_channels: int ,
            n_classes: int,
            F1: int = 16,
            kernel_length_conv: int = 64,
            pool_length: int = 8,
            D: int = 2,
            dropout_conv: float = 0.3,
            d_model: int = 32,
            key_dim: int = 8,
            n_head: int = 2,
            dropout_attn: float = 0.5,
            tcn_depth: int = 2,
            kernel_length_tcn: int = 4,
            dropout_tcn: float = 0.3,
            n_windows: int = 5,
            **kwargs
    ):
        model = ATCNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1,
            kernel_length_conv=kernel_length_conv,
            pool_length=pool_length,
            D=D,
            dropout_conv=dropout_conv,
            d_model=d_model,
            key_dim=key_dim,
            n_head=n_head,
            dropout_attn=dropout_attn,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            n_windows=n_windows
        )
        super(ATCNet, self).__init__(model, n_classes, **kwargs)
