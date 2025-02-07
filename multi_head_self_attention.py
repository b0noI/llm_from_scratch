from self_attention import SelfAttention_v1

from torch import nn

import torch


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, *, d_in, d_out, context_length, num_heads, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention_v1(
                d_in, d_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
