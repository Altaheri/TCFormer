from einops.layers.torch import Rearrange
import torch
from torch import nn

from .classification_module import ClassificationModule
from channel_attention.utils.weight_initialization import glorot_weight_zero_bias


class ShallowNetModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            n_filters_time: int = 40,
            filter_time_length: int = 25,
            pool_time_length: int = 75,
            pool_time_stride: int = 15,
            drop_prob: float = 0.5,
    ):
        super(ShallowNetModule, self).__init__()
        self.rearrange_input = Rearrange("b c t -> b 1 t c")
        self.conv_time = nn.Conv2d(1, n_filters_time, (filter_time_length, 1),
                                   bias=True)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time, (1, n_channels),
                                   bias=False)
        self.bnorm = nn.BatchNorm2d(n_filters_time)

        self.pool = nn.AvgPool2d((pool_time_length, 1), (pool_time_stride, 1))
        self.dropout = nn.Dropout(drop_prob)
        out = input_window_samples - filter_time_length + 1
        out = int((out - pool_time_length) / pool_time_stride + 1)

        self.classifier = nn.Sequential(nn.Conv2d(n_filters_time, n_classes, (out, 1)),
                                        Rearrange("b c 1 1 -> b c"))
        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ShallowNet(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            n_filters_time: int = 40,
            filter_time_length: int = 25,
            pool_time_length: int = 75,
            pool_time_stride: int = 15,
            drop_prob: float = 0.5,
            **kwargs
    ):
        model = ShallowNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob
        )
        super(ShallowNet, self).__init__(model, n_classes, **kwargs)
