from self_attention import SelfAttention_v1
from multi_head_self_attention import MultiHeadSelfAttention
from multi_head_self_attention_2 import MultiHeadSelfAttention_2
from torch_based_data import load_verdict_txt_dataloader


import torch


torch.manual_seed(123)


vocab_size = 50257
embedding_size = 768
context_length = 1024
batch_size = 5
num_heads = 12

embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

dataloader = load_verdict_txt_dataloader(batch_size=batch_size, max_length=context_length, shuffle=True, stride=context_length)
data_iter = iter(dataloader)
inputs, tagets = next(data_iter)
token_embeddings = embedding_layer(inputs)

pos_embedding_layer = torch.nn.Embedding(context_length, embedding_size)
range_tensor = torch.arange(context_length)
pos_embeddings = pos_embedding_layer(range_tensor)
input_embeddings = token_embeddings + pos_embeddings

d_in = embedding_size
d_out = embedding_size

llm = MultiHeadSelfAttention_2(d_in=d_in, d_out=d_out, context_length=context_length, num_heads=num_heads)
result = llm(input_embeddings)
print(result)
print(result.shape)
