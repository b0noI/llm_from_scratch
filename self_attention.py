import torch
from torch import nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        context_length = self._get_context_length(x)
        atten_scores = queries @ keys.T

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        atten_masked = atten_scores.masked_fill(mask.bool(), -torch.inf)
        atten_weights = torch.softmax(
            atten_masked / keys.shape[-1] ** 0.5, dim=-1
        )

        atten_weights_with_dropout = self.dropout(atten_weights)

        return atten_weights_with_dropout @ values
    
    def _get_context_length(self, x):
        return x.shape[0]

