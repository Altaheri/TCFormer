from .classification_module import ClassificationModule
import math

import torch
from torch import nn

from .sst_dpn_model_utils import SSA, LightweightConv1d, Mixer1D


class Efficient_Encoder(nn.Module):

    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=75,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x): # x: (batch_size, chans, samples) .Size([64, 22, 1000])

        x = self.time_conv(x) # torch.Size([64, 198, 1000])
        x, _ = self.ssa(x) # torch.Size([64, 198, 1000])
        x_chan = self.chanConv(x) # torch.Size([64, 48, 1000])

        feature = self.mixer(x_chan) # torch.Size([64, 1072])

        return feature


class SST_DPN_Module(nn.Module):

    def __init__(
        self,
        chans,
        samples,
        num_classes=4,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=[50, 100, 200],
    ):
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, chans, samples)) # torch.Size([1, 22, 1000])
        out = self.encoder(x) # torch.Size([1, 1072])
        feat_dim = out.shape[-1] # 1072

        # *Inter-class Separation Prototype(ISP)
        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True) # torch.Size([4, 1072])
        # *Intra-class Compactness(ICP)
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True) # torch.Size([4, 1072])
        nn.init.kaiming_normal_(self.isp)

        # self.linear = nn.Linear(feat_dim, num_classes)

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x): # torch.Size([64, 22, 1000])

        features = self.encoder(x) # torch.Size([64, 1072])
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1) # torch.Size([4, 1072])
        logits = torch.einsum("bd,cd->bc", features, self.isp) # torch.Size([64, 4])

        # logits = self.linear(features) # torch.Size([64, 4])
        return logits
        # return features

class SST_DPN(ClassificationModule):
    def __init__(
            self,
            F1 = 9,
            F2 = 48,
            time_kernel1 = 75,
            pool_kernels = [50, 100, 200],

            n_channels: int = 22,
            n_classes: int = 4,
            **kwargs
    ):
        model = SST_DPN_Module(
            chans=n_channels, 
            samples=1000, 
            num_classes=n_classes,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        super(SST_DPN, self).__init__(model, n_classes, **kwargs)
    
if __name__ == "__main__":
    # a simple test
    model = SST_DPN_Module(chans=22, samples=1000, num_classes=4)
    inp = torch.rand(64, 22, 1000)
    out = model(inp)
    print(out.shape)
