import torch
from torch import nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # dimentions: [batch, tokens, embeddings]
        # context length and d_in we have saved in constructor
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        atten_scores = queries @ keys.transpose(1, 2)
        
        atten_scores.masked_fill_(
            self.mask.bool(), -torch.inf
        )

        atten_weights = torch.softmax(
            atten_scores / self.d_in ** 0.5, dim=-1
        )
        
        atten_weights = self.dropout(atten_weights)
        return atten_weights @ values

