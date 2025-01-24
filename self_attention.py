import torch
from torch import nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_query = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_keys = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_values = torch.nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_keys
        values = x @ self.W_values

        atten_scores = queries @ keys.T
        atten_weights = torch.softmax(
            atten_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        return atten_weights @ values
