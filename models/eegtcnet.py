from einops.layers.torch import Rearrange
from torch import nn

from .modules import CausalConv1d, Conv2dWithConstraint, LinearWithConstraint
from .classification_module import ClassificationModule
from channel_attention.utils.weight_initialization import glorot_weight_zero_bias

nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float, activation: str = "relu"):
        super(_TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nonlinearity_dict[activation]
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nonlinearity_dict[activation]
        self.drop2 = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.project_channels = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.project_channels = nn.Identity()
        self.final_nonlinearity = nonlinearity_dict[activation]

    def forward(self, x):
        residual = self.project_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        return self.final_nonlinearity(out + residual)


class EEGTCNetModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            layers: int = 2,
            kernel_s: int = 4,
            filt: int = 12,
            dropout: float = 0.3,
            activation: str = 'relu',
            F1: int = 8,
            D: int = 2,
            kernLength: int = 32,
            dropout_eeg: float = 0.2
    ):
        super(EEGTCNetModule, self).__init__()
        regRate = 0.25
        numFilters = F1
        F2 = numFilters * D

        self.eegnet = nn.Sequential(
            Rearrange("b c t -> b 1 c t"),
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=0.001),
            Conv2dWithConstraint(F1, F2, (n_channels, 1), bias=False, groups=F1,
                                 max_norm=1),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            Rearrange("b c 1 t -> b c t")
        )

        in_channels = [F2] + (layers - 1) * [filt]
        dilations = [2 ** i for i in range(layers)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])

        self.classifier = LinearWithConstraint(filt, n_classes, max_norm=regRate)
        glorot_weight_zero_bias(self.eegnet)
        glorot_weight_zero_bias(self.classifier)

    def forward(self, x):
        x = self.eegnet(x)
        for blk in self.tcn_blocks:
            x = blk(x)
        x = self.classifier(x[:, :, -1])
        return x


class EEGTCNet(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            layers: int = 2,
            kernel_s: int = 4,
            filt: int = 12,
            dropout: float = 0.3,
            activation: str = 'relu',
            F1: int = 8,
            D: int = 2,
            kernLength: int = 32,
            dropout_eeg: float = 0.2,
            **kwargs
    ):
        model = EEGTCNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            layers=layers,
            kernel_s=kernel_s,
            filt=filt,
            dropout=dropout,
            activation=activation,
            F1=F1,
            D=D,
            kernLength=kernLength,
            dropout_eeg=dropout_eeg
        )
        super(EEGTCNet, self).__init__(model, n_classes, **kwargs)
