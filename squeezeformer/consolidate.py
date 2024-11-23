import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class Swish(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class DepthwiseConv2dSubsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DepthwiseConv2dSubsampling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.out_channels = out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_p: float) -> None:
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(inputs))))

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float) -> None:
        super(RelativeMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_p)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Tensor) -> Tensor:
        batch_size, seq_length, _ = query.size()
        qkv = self.qkv_proj(query).reshape(batch_size, seq_length, self.num_heads, 3 * self.d_model // self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = (attn @ v).transpose(1, 2).reshape(batch_size, seq_length, self.d_model)
        return self.out_proj(context)

class SqueezeformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_p: float) -> None:
        super(SqueezeformerBlock, self).__init__()
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.feed_forward = FeedForwardModule(d_model, d_ff, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: Tensor, pos_embedding: Tensor, mask: Tensor) -> Tensor:
        x = self.norm1(inputs)
        x = self.attention(x, x, x, pos_embedding, mask)
        x = self.dropout(x) + inputs
        y = self.norm2(x)
        y = self.feed_forward(y)
        return self.dropout(y) + x

class Squeezeformer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_model: int, num_heads: int, d_ff: int, num_blocks: int, dropout_p: float) -> None:
        super(Squeezeformer, self).__init__()
        self.conv_subsample = DepthwiseConv2dSubsampling(in_channels, out_channels)
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(d_model, num_heads, d_ff, dropout_p) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(d_model, out_channels)

    def forward(self, inputs: Tensor, pos_embedding: Tensor, mask: Tensor) -> Tensor:
        x = self.conv_subsample(inputs)
        for block in self.blocks:
            x = block(x, pos_embedding, mask)
        return self.fc(x)