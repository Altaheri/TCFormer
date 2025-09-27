from einops.layers.torch import Rearrange

from torch import nn

from .classification_module import ClassificationModule
from .modules import Conv2dWithConstraint
from channel_attention.utils.weight_initialization import glorot_weight_zero_bias


class EEGNetModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16
    ):
        super(EEGNetModule, self).__init__()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]


        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), bias=False,
                                       padding=(0, kernel_length // 2))
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(F1, F1 * D, (n_channels, 1),
                                                 bias=False, groups=F1, max_norm=1)
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu_1 = nn.ELU()
        self.pool_1 = pool_class((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop1 = nn.Dropout(drop_prob)

        self.dw_sep_conv = nn.Sequential(
            nn.Conv2d(F2, F2, (1, kernel_length_dw_sep), bias=False, groups=F2,
                      padding=(0, kernel_length_dw_sep // 2)),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3),
            nn.ELU(),
            pool_class((1, 8)),
            nn.Dropout(drop_prob),
        )

        out = input_window_samples + 2 * (kernel_length // 2) - kernel_length + 1
        out = int((out - pool_time_length) / pool_time_stride + 1)
        out = out + 2 * (kernel_length_dw_sep // 2) - kernel_length_dw_sep + 1
        out = int((out - 8) / 8 + 1)

        self.classifier = nn.Sequential(
            nn.Conv2d(F2, n_classes, (1, out)),
            Rearrange("b n_classes 1 1 -> b n_classes"))

        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        x = self.pool_1(x)
        x = self.drop1(x)
        x = self.dw_sep_conv(x)
        x = self.classifier(x)
        return x


class EEGNet(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            **kwargs
    ):
        model = EEGNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            drop_prob=drop_prob,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            kernel_length_dw_sep=kernel_length_dw_sep,
        )
        super(EEGNet, self).__init__(model, n_classes, **kwargs)
