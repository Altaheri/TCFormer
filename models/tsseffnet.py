import os
from pathlib import Path

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from scipy import io
import torch
from torch import nn

from .classification_module import ClassificationModule


class _TempConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_length: int = 11,
                 conv_stride: int = 1, batch_norm: bool = True,
                 batch_norm_alpha: float = 0.1, pool_stride: int = 3,
                 drop_prob: float = 0.5):
        super(_TempConvUnit, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_length, 1),
                              stride=(conv_stride, 1), bias=not batch_norm)
        self.bn = nn.BatchNorm2d(out_channels, momentum=batch_norm_alpha, affine=True,
                                 eps=1e-5)
        self.conv_nonlinear = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(pool_stride, 1))

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.conv_nonlinear(x)
        x = self.pool(x)
        return x


class _SELayer(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super(_SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.gap(x).squeeze(-1).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class _WaveletTransform(nn.Module):
    def __init__(self, channel: int):

        super(_WaveletTransform, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel * 2,
                              kernel_size=(1, 8), stride=(1, 2), padding=0,
                              groups=channel, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                path = os.path.join(
                    Path(__file__).resolve().parents[0], "ts-seffnet_filter.mat")
                f = io.loadmat(path)
                Lo_D = np.flip(f['Lo_D'], axis=1).astype('float32')
                Hi_D = np.flip(f['Hi_D'], axis=1).astype('float32')
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis=0)).\
                    unsqueeze(1).unsqueeze(1).repeat(channel, 1, 1, 1)
                m.weight.requires_grad = False

    def forward(self, x):
        out = self.conv(self.self_padding(x))
        return out[:, 0::2, :, :], out[:, 1::2, :, :]

    @staticmethod
    def self_padding(x):
        return torch.cat((x[:, :, :, -3:], x, x[:, :, :, 0:3]), 3)


class TSSEFFNetModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            reduction_ratio: int = 8,
            conv_stride: int = 1,
            pool_stride: int = 3,
            batch_norm: bool = True,
            batch_norm_alpha: float = 0.1,
            drop_prob: float = 0.5
    ):
        super(TSSEFFNetModule, self).__init__()
        self.input_block = nn.Sequential(
            Rearrange("b c t -> b 1 t c"),
            nn.Conv2d(1, 25, (11, 1)),
            nn.Conv2d(25, 25, (1, n_channels), bias=not batch_norm),
            nn.BatchNorm2d(25, momentum=batch_norm_alpha, affine=True),
            nn.ELU(),
        )

        self.first_pool = nn.MaxPool2d((3, 1), (pool_stride, 1))
        self.temp_conv_units = nn.ModuleList(
            _TempConvUnit(in_ch, out_ch, conv_stride=conv_stride, batch_norm=batch_norm,
                          batch_norm_alpha=batch_norm_alpha, pool_stride=pool_stride,
                          drop_prob=drop_prob)
            for (in_ch, out_ch) in zip([25, 100, 100], [100, 100, 100])
        )

        self.conv_spectral = nn.Conv2d(25, 10, (1, 1))
        self.wavelet_transform = _WaveletTransform(10)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 69))
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)

        self.SEBlock1 = nn.Sequential(
            _SELayer(50, reduction_ratio),
            nn.Conv2d(50, 100, kernel_size=(1, 7)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3))
        )

        self.SEBlock2 = nn.Sequential(
            _SELayer(100, reduction_ratio),
            nn.Conv2d(100, 100, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1),
                      groups=1, bias=True),
            nn.BatchNorm2d(100),
            nn.ELU()
        )

        self.conv_classifier = nn.Sequential(
            nn.Conv2d(100, n_classes, (9, 1)),
            Rearrange("b n_classes 1 1 -> b n_classes")
        )

    def forward(self, x):
        x = self.input_block(x)

        out = self.conv_spectral(x)
        out, gamma = self.wavelet_transform(rearrange(out, "b c t 1 -> b c 1 t"))
        out, beta = self.wavelet_transform(out)
        out, alpha = self.wavelet_transform(out)
        delta, theta = self.wavelet_transform(out)
        x_freq_feature = torch.cat((
            delta,
            theta,
            self.avg_pool1(alpha),
            self.avg_pool1(beta),
            self.avg_pool1(gamma)), 1)

        x_freq_feature = self.SEBlock1(x_freq_feature)
        x_freq_feature = self.avg_pool2(x_freq_feature)

        x = self.first_pool(x)
        for blk in self.temp_conv_units:
            x = blk(x)

        x = self.SEBlock2(x)

        x = torch.cat([x, x_freq_feature], dim=2)
        x = self.conv_classifier(x)
        return x


class TSSEFFNet(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            reduction_ratio: int = 8,
            conv_stride: int = 1,
            pool_stride: int = 3,
            batch_norm: bool = True,
            batch_norm_alpha: float = 0.1,
            drop_prob: float = 0.5,
            **kwargs
    ):
        model = TSSEFFNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            reduction_ratio=reduction_ratio,
            conv_stride=conv_stride,
            pool_stride=pool_stride,
            batch_norm=batch_norm,
            batch_norm_alpha=batch_norm_alpha,
            drop_prob=drop_prob
        )
        super(TSSEFFNet, self).__init__(model, n_classes, **kwargs)
