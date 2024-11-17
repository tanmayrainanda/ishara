import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        residual = x
        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch, dim) 
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # Revert shape back to (batch, seq_len, dim)
        return self.layer_norm(x + residual)

class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)  # Change shape to (batch, dim, seq_len) for conv layers
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # Revert shape to (batch, seq_len, dim)
        return self.layer_norm(x + residual)

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, expansion_factor=4, kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(dim, expansion_factor, dropout)
        self.attention = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(dim, expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        x = self.ffn1(x)
        x = self.attention(x, attn_mask=attn_mask)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.layer_norm(x)

# Create a stack of Conformer blocks
class ConformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=7, num_heads=8, expansion_factor=4, kernel_size=31, dropout=0.1):
        super(ConformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(dim, num_heads, expansion_factor, kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x
