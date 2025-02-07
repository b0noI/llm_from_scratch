from self_attention import SelfAttention_v1

from torch import nn

import torch


class MultiHeadSelfAttention_2(nn.Module):

    def __init__(self, *, d_in, d_out, context_length, num_heads, dropout=0.5, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out MUST be divisible by num_heads"
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), 
                       diagonal=1)
        )

    def forward(self, x):
        # dimentions: [batch, tokens, embeddings]
        # context length and d_in we have saved in constructor
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(
            x.shape[0], self.context_length, 
            self.num_heads, self.head_dim)
        queries = queries.view(
            x.shape[0], self.context_length, 
            self.num_heads, self.head_dim)
        values = values.view(
            x.shape[0], self.context_length, 
            self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**-.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            x.shape[0], self.context_length, self.d_out
        )
        return self.out_proj(context_vector)

