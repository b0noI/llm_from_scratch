import torch

from torch_based_data import load_verdict_txt_dataloader


vocab_size = 50257
embedding_size = 3

embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

context_length = 10
batch_size = 8

dataloader = load_verdict_txt_dataloader(batch_size=batch_size, max_length=context_length, shuffle=True, stride=context_length)
data_iter = iter(dataloader)
inputs, tagets = next(data_iter)
token_embeddings = embedding_layer(inputs)

pos_embedding_layer = torch.nn.Embedding(context_length, embedding_size)
range_tensor = torch.arange(context_length)
pos_embeddings = pos_embedding_layer(range_tensor)
input_embeddings = token_embeddings + pos_embeddings

d_in = embedding_size
d_out = embedding_size + 1

torch.manual_seed(123)
q_w = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
k_w = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
v_w = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)


def compute_context_for_token_i_no_batch(input_embeddings, i):
    query_input = input_embeddings[i]
    query = query_input @ q_w
    keys = input_embeddings @ k_w
    values = input_embeddings @ v_w

    attention_scores = query @ keys.T
    attention_scores = torch.softmax(attention_scores / (query_input.shape[-1] ** 0.5), dim=-1)

    context = attention_scores @ values
    return context


print(compute_context_for_token_i_no_batch(input_embeddings[0], 3))
