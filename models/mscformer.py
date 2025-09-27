"""
Zhao, W., Zhang, B., Zhou, H. et al. Multi-scale convolutional transformer network for motor imagery brain-computer interface. Sci Rep 15, 12935 (2025). 
https://doi.org/10.1038/s41598-025-96611-5

@author: Wei Zhao
"""

import numpy as np
import pandas as pd
import random
import datetime
import time
import os
import math
import torch
from torchsummary import summary
from torch.backends import cudnn
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from torch.autograd import Variable
cudnn.benchmark = False
cudnn.deterministic = True

class Parameters():
    def __init__(self, dropout_rate=0.5):
        # The number of heads in the multi-head self-attention mechanism
        self.heads = 8
        # Transformer encoder depth
        self.depth = 5
        # The total number of feature channels of the multi-scale convolution module
        self.emb_size = 16*3
        # The number of feature channels at each time scale of the convolution module
        self.f1 = 16 
        # average pooling size in Convolution module, 44 for BCI IV-2a and 52 for BCI IV-2b
        self.pooling_size = 52  
        # drop out rate in Convolution module, 0.5 for subject-specific, and 0.25 for corss-subject
        self.dropout_rate = dropout_rate

# def numberClassChannel(database_type):
#     if database_type=='A': # BCI IV-2a dataset
#         number_class = 4
#         number_channel = 22
#     elif database_type=='B': # BCI IV-2b dataset
#         number_class = 2
#         number_channel = 3
#     return number_class, number_channel

class PatchEmbeddingCNN(nn.Module):
    '''
    Multi-scale convolutional module
    '''
    def __init__(self, 
                 f1=16, 
                 pooling_size=52, 
                 dropout_rate=0.5, 
                 number_channel=22):
        super().__init__()
        self.rearrange_input = Rearrange("b c seq -> b 1 c seq")

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 85), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 65), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )        
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 45), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.rearrange_input(x) 
        b, _, _, _ = x.shape
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        # Concatenate along feature channel dimension.
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.projection(x)
        return x    
    
  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out

# add & LayerNorm
class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res)+x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        # x = x + self.encoding[:, :x.shape[1], :].cuda()
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)        
        
class MSCFormer_model(nn.Module):
    def __init__(self, 
                 parameters,
                 n_channels,
                 n_classes,
                #  database_type='A', 
                 **kwargs):
        super().__init__()
        # self.number_class, self.number_channel = numberClassChannel(database_type)
        self.number_class = n_classes
        self.number_channel = n_channels

        self.emb_size = parameters.emb_size
        parameters.number_channel = self.number_channel
        self.cnn = PatchEmbeddingCNN(f1=parameters.f1, 
                                 pooling_size=parameters.pooling_size, 
                                 dropout_rate=parameters.dropout_rate,
                                 number_channel=parameters.number_channel,
                                 )
        self.position = PositioinalEncoding(parameters.emb_size, dropout=0.1)
        self.trans = TransformerEncoder(parameters.heads, 
                                        parameters.depth, 
                                        parameters.emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.emb_size , self.number_class) 
    def forward(self, x):
        # x.shape = [batch, feature channels, electrodes channels, length], such as [batch, 1, 22, 1000] in BCI IV-2a dataset.
        x = self.cnn(x)
        b, l, e = x.shape
        
        # Add class token like BERT
        # x = torch.cat((torch.zeros((b, 1, e),requires_grad=True).cuda(),x), 1)
        x = torch.cat((torch.zeros((b, 1, e), requires_grad=True, device=x.device), x), 1)
        x = x * math.sqrt(self.emb_size)
        x = self.position(x)
        trans = self.trans(x)

        # Take the class token as the final features for classifying.
        features = trans[:, 0, :]
        
        out = self.classification(features)
        # return features, out
        return out

from .modules import CausalConv1d, Conv1dWithConstraint
class TCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 n_groups: int = 1, dropout: float = 0.3):
        super(TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        # self.conv1wn = weight_norm(CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
        #                           dilation=dilation, groups=n_groups))
        # self.bn1 = nn.BatchNorm1d(n_filters, momentum=0.01, eps=0.001)
        self.bn1 = nn.BatchNorm1d(n_filters)
        # self.ln1 = nn.LayerNorm(n_filters)
        # self.gn1 = nn.GroupNorm(num_groups=n_groups, num_channels=n_filters)
        # self.gn1 = nn.GroupNorm(num_groups=1, num_channels=n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        # self.conv2wn = weight_norm(CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
        #                           dilation=dilation, groups=n_groups))        
        # self.bn2 = nn.BatchNorm1d(n_filters, momentum=0.01, eps=0.001)
        self.bn2 = nn.BatchNorm1d(n_filters)
        # self.gn2 = nn.GroupNorm(num_groups=n_groups, num_channels=n_filters)
        # self.gn2 = nn.GroupNorm(num_groups=1, num_channels=n_filters)
        # self.ln2 = nn.LayerNorm(n_filters)

        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.nonlinearity3 = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)
        # nn.init.constant_(self.conv1wn.bias, 0.0)
        # nn.init.constant_(self.conv2wn.bias, 0.0)

    def forward(self, input):
        # x = self.drop1(self.nonlinearity1(self.ln1(self.conv1(input).permute(0, 2, 1)).permute(0, 2, 1) ))
        # x = self.drop2(self.nonlinearity2(self.ln2(self.conv2(x).permute(0, 2, 1)).permute(0, 2, 1) ))
        # x = self.drop1(self.nonlinearity1(self.gn1(self.conv1(input))))
        # x = self.drop2(self.nonlinearity2(self.gn2(self.conv2(x))))
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        # x = self.drop1(self.nonlinearity1((self.conv1(input))))
        # x = self.drop2(self.nonlinearity2((self.conv2(x))))
        # x = self.drop1(self.nonlinearity1((self.conv1wn(input))))
        # x = self.drop2(self.nonlinearity2((self.conv2wn(x))))
        x = self.nonlinearity3(input + x)
        return x


class TCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 n_groups: int = 1, dropout: float = 0.3):
        super(TCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(kernel_length, n_filters, dilation, n_groups, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

from channel_attention.utils.weight_initialization import glorot_weight_zero_bias
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
from .channel_group_attention import ChannelGroupAttention
class ATCBlock(nn.Module):
    def __init__(self, d_model: int = 32, n_groups: int = 1, key_dim: int = 8, n_head: int = 2,
                 dropout_attn: float = 0.3, tcn_depth: int = 2, kernel_length: int = 4,
                 dropout_tcn: float = 0.3, n_classes: int = 4, use_group_attn: str = None):

        super().__init__()
        self.use_group_attn = use_group_attn
        if use_group_attn != None:
            self.group_attn = ChannelGroupAttention(in_channels=d_model, num_groups=n_groups,
                                                     use_group_attn=use_group_attn)
        self.attention_block = _AttentionBlock(d_model, key_dim, n_head, dropout_attn)
        
        self.rearrange = Rearrange("b seq c -> b c seq")
        self.rearrange1 = Rearrange("b c seq -> b seq c")
        self.tcn = TCN(tcn_depth, kernel_length, d_model, n_groups, dropout_tcn)
        # self.linear = LinearWithConstraint(d_model, n_classes, max_norm=0.25)
        # self.linear = LinearWithConstraint(d_model, n_classes)
        # self.linear1 = nn.Conv1d(d_model, n_classes*n_groups, kernel_size=1, groups=n_groups)
        self.linear1 = Conv1dWithConstraint(d_model, n_classes*n_groups, kernel_size=1, 
                                               groups=n_groups, max_norm=0.25)
        # self.linear = LinearWithConstraint(d_model*20, n_classes, max_norm=0.25)
        # self.linear = LinearWithConstraint(d_model*6, n_classes, max_norm=0.25)
        # self.linear = LinearWithConstraint(17*d_model, n_classes)
        
        self.flatten = nn.Flatten()
        # self.classification = ClassificationHead(d_model*17, n_classes) 
        # self.classification = ClassificationHead(16*3*17, n_classes)
        # self.classification = ClassificationHead(d_model*4, n_classes) 

    def forward(self, x):
        x = self.attention_block(x)

        x = self.rearrange(x).unsqueeze(2)   # [batch, feature channels, length]
        if self.use_group_attn != None:            # group attention (optional) 
            x = x + self.group_attn(x)             # residual attention

        tcn_out = self.tcn(x.squeeze(2))  # [batch, feature channels, length]
        # tcn_out = tcn_out[:, :, -1]
        # tcn_out = self.linear(self.flatten(tcn_out))
        tcn_out = self.rearrange1(tcn_out[:, :, -1:])
        # tcn_out = self.linear1(tcn_out).squeeze(-1)
        return tcn_out


class MSCFormer_TCN_model(nn.Module):
    def __init__(self, 
                 parameters,
                 in_channels = 22,
                 n_classes = 4,
                 use_group_attn: str = None,
                #  database_type='A', 
                 **kwargs):
        super().__init__()
        # self.number_class, self.number_channel = numberClassChannel(database_type)
        self.number_class = n_classes
        self.number_channel = in_channels

        self.emb_size = parameters.emb_size
        parameters.number_channel = self.number_channel
        self.cnn = PatchEmbeddingCNN(f1=parameters.f1, 
                                 pooling_size=parameters.pooling_size, 
                                 dropout_rate=parameters.dropout_rate,
                                 number_channel=parameters.number_channel,
                                 )
        
        d_model = 16
        self.n_groups = 3
        n_windows = 1
        key_dim = 8
        n_head = 2
        dropout_attn = 0.3
        tcn_depth = 2
        kernel_length_tcn = 4
        dropout_tcn = 0.3
        self.atc_block = ATCBlock(d_model*self.n_groups*n_windows, self.n_groups*n_windows, key_dim, n_head, 
                                  dropout_attn, tcn_depth, kernel_length_tcn, dropout_tcn, n_classes,
                                  use_group_attn = use_group_attn)

        self.position = PositioinalEncoding(parameters.emb_size, dropout=0.1)
        self.trans = TransformerEncoder(parameters.heads, 
                                        parameters.depth, 
                                        parameters.emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.emb_size , self.number_class) 
    def forward(self, x):
        # x.shape = [batch, feature channels, electrodes channels, length], such as [batch, 1, 22, 1000] in BCI IV-2a dataset.
        x = self.cnn(x)
        b, l, e = x.shape
        
        # Add class token like BERT
        blk_output = self.atc_block(x)
        x = torch.cat((blk_output,x), 1)
        # x = torch.cat((torch.zeros((b, 1, e),requires_grad=True).cuda(),x), 1)
        x = x * math.sqrt(self.emb_size)
        x = self.position(x)
        trans = self.trans(x)

        # Take the class token as the final features for classifying.
        features = trans[:, 0, :]
        
        out = self.classification(features)
        # return features, out
        return out
    
from .classification_module import ClassificationModule

class MSCFormer(ClassificationModule):
    def __init__(self,
            n_channels: int,
            n_classes: int,
            **kwargs
        ):
        model = MSCFormer_model(
            parameters=Parameters(),
            n_classes=n_classes,
            n_channels=n_channels,
        )
        super().__init__(model, n_classes, **kwargs)
