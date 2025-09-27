import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LightweightConv1d(nn.Module):

    def __init__(
        self,
        in_channels,
        num_heads=1,
        depth_multiplier=1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp): # torch.Size([64, 22, 1000])
        B, C, T = inp.size()
        H = self.num_heads # 1

        weight = self.weight # torch.Size([9, 1, 75])``
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        # input = input.view(-1, H, T)
        inp = rearrange(inp, "b (h c) t ->(b c) h t", h=H) # torch.Size([1408, 1, 1000])
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d( # torch.Size([1408, 9, 1000])
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t ->b (h c) t", b=B) # torch.Size([64, 198, 1000])

        return output


class VarMaxPool1D(nn.Module):
    def __init__(self, T, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x): # torch.Size([64, 198, 1000])
        mean_of_squares = F.avg_pool1d( # torch.Size([64, 198, 4])
            x**2, self.kernel_size, self.stride, self.padding
        )
        # Compute the square of the mean (E[x])^2
        square_of_mean = ( # torch.Size([64, 198, 4])
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean # torch.Size([64, 198, 4])
        # out = self.time_agg(variance)
        out = F.avg_pool1d(variance, variance.shape[-1])    # torch.Size([64, 198, 1])

        return out


class VarPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x): # torch.Size([64, 16, 1000])
        # Calculate the size of the result tensor after pooling

        # Compute the mean of the squares (E[x^2])
        mean_of_squares = F.avg_pool1d(     # torch.Size([64, 16, 39/19/9])
            x**2, self.kernel_size, self.stride, self.padding
        )

        # Compute the square of the mean (E[x])^2
        square_of_mean = (      # torch.Size([64, 16, 39/19/9])
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean # torch.Size([64, 16, 39/19/9])

        return variance


class SSA(nn.Module):
    # Spatial-Spectral Attention
    def __init__(self, T, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        self.GP = VarMaxPool1D(T, 250)

    def forward(self, x): # torch.Size([64, 198, 1000])
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        elif self.mode == "var":

            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha   # torch.Size([64, 198, 1])
            norm = (self.gamma) / ( # torch.Size([64, 198, 1])
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta) # torch.Size([64, 198, 1])

        return x * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[50, 100, 250]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):   # torch.Size([64, 48, 1000])
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)    # 3 * torch.Size([64, 16, 1000])
        out = [] # 3 * torch.Size([64, 624/304/144])
        for i in range(len(x_split)):
            x = self.var_layers[i](x_split[i])  # torch.Size([64, 624])
            out.append(x)
        y = torch.concat(out, dim=1)    # torch.Size([64, 1072])
        return y

class Mixer1D_1(nn.Module):
    def __init__(self, kernel_sizes=[100, 150, 200]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    #nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):   # torch.Size([64, 48, 1000])
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)    # 3 * torch.Size([64, 16, 1000])
        out = [] # 3 * torch.Size([64, 16, 39/19/9])
        for i in range(len(x_split)):
            x = self.var_layers[i](x_split[i])  # torch.Size([64, 624])
            x = x.permute(0, 2, 1)     # => [B, T, C]  (like [B, seq_len, d_model]) torch.Size([64, 20, 32])
            out.append(x)
        
        # y = torch.stack(out, dim=1)  # [B, n_windows, n_classes] torch.Size([64, 5, 4])
        # y = torch.concat(out, dim=1)    # torch.Size([64, 1072])
        return out