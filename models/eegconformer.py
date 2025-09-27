from einops import rearrange
import torch
from torch import nn

from .classification_module import ClassificationModule


class _PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 22, embedding_size: int = 40):
        super(_PatchEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (in_channels, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5)
        )

        self.projection = nn.Conv2d(40, self.embedding_size, (1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.shallownet(x)
        x = self.projection(x)
        # x = torch.permute(torch.squeeze(x), (0, 2, 1))
        x = torch.permute(torch.squeeze(x, dim=2), (0, 2, 1))
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = torch.nn.functional.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(_ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int, dropout: float):
        self.embedding_size = embedding_size
        self.expansion = expansion
        self.dropout = dropout
        super(_FeedForwardBlock, self).__init__(
            nn.Linear(self.embedding_size, self.expansion * self.embedding_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.expansion * self.embedding_size, self.embedding_size)
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embedding_size: int, num_heads: int = 10, dropout: float = 0.5,
                 forward_expansion: int = 4, forward_dropout: float = 0.5):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        super(_TransformerEncoderBlock, self).__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(self.embedding_size),
                _MultiHeadAttention(self.embedding_size, self.num_heads, self.dropout),
                nn.Dropout(self.dropout)
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(self.embedding_size),
                _FeedForwardBlock(self.embedding_size, expansion=self.forward_expansion,
                                  dropout=self.forward_dropout),
                nn.Dropout(self.dropout)
            ))
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, embedding_size: int):
        self.depth = depth
        self.embedding_size = embedding_size
        super(_TransformerEncoder, self).__init__(
            *[_TransformerEncoderBlock(self.embedding_size) for _ in range(self.depth)])


class _ClassificationHead(nn.Module):
    def __init__(self, embedding_size: int, n_classes: int, input_size_cls: int = 2440):
        super(_ClassificationHead, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.head = nn.Sequential(
            nn.Linear(input_size_cls, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.head(x)
        return x, out


class EEGConformerModule(nn.Module):
    def __init__(self, 
                 n_channels: int, 
                 n_classes: int, 
                 embedding_size: int = 40, 
                 depth: int = 6,
                 input_size_cls: int = 2440
            ):
        super(EEGConformerModule, self).__init__()
        self.embedding_size = embedding_size
        self.depth = depth
        self.n_classes = n_classes
        self.patch_embedding = _PatchEmbedding(n_channels, self.embedding_size)
        self.transformer_encoder = _TransformerEncoder(self.depth, self.embedding_size)
        self.classification_head = _ClassificationHead(self.embedding_size,
                                                       self.n_classes,
                                                       input_size_cls=input_size_cls)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        _, x = self.classification_head(x)
        return x

def inspect_model_parameters(model: torch.nn.Module, top_k: int = 5):
    """
    Prints the total number of trainable parameters and the top-k layers 
    with the most parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
        top_k (int): Number of top layers to display based on parameter count.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Total trainable parameters: {total_params / 1e3:.2f}K\n")

    layer_param_counts = [
        (name, param.numel()) 
        for name, param in model.named_parameters() 
        if param.requires_grad
    ]

    # Sort layers by parameter count descending
    layer_param_counts.sort(key=lambda x: x[1], reverse=True)

    print(f"🔍 Top {top_k} layers by parameter count:")
    for name, count in layer_param_counts[:top_k]:
        print(f"{name:<40} {count:,} params")

    max_layer = layer_param_counts[0]
    print(f"\n🔥 Layer with the most parameters: {max_layer[0]} ({max_layer[1]:,} params)")

class EEGConformer(ClassificationModule):
    def __init__(self, 
                 n_channels: int, 
                 n_classes: int, 
                 embedding_size: int = 40, 
                 depth: int = 6,
                 input_size_cls: int = 2440, 
                 **kwargs
            ):
        model = EEGConformerModule(
            n_channels=n_channels, 
            n_classes=n_classes,
            embedding_size=embedding_size, 
            depth=depth, 
            input_size_cls=input_size_cls
        )
        inspect_model_parameters(model)
        
        super().__init__(model, n_classes, **kwargs)


