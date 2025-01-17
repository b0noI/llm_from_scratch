import torch

from torch_based_data import load_verdict_txt_dataloader


vocab_size = 50257
embedding_size = 256

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


def calculate_attention_score_no_batch(inputs, query_index):
    query = inputs[query_index]
    attention_scores = torch.empty(inputs.shape[0])
    for i in range(len(inputs)):
        attention_scores[i] = torch.dot(query, inputs[i])
    attention_scores = torch.softmax(attention_scores, dim=0)
    return attention_scores


def calculate_context_vctor_no_batch(inputs, i):
    attention_score = calculate_attention_score_no_batch(inputs, i)
    contex_vector = torch.zeros(inputs[0].shape)
    for i in range(len(inputs)):
        contex_vector += attention_score[i] * inputs[i]
    return contex_vector


def calculate_attention_score(inputs):
    return torch.softmax(inputs @ inputs.T, dim=-1)


def calculate_context_vctor(inputs):
    return calculate_attention_score(inputs) @ inputs

# print(torch.sum(calculate_attention_score(input_embeddings[0]), dim=-1))
# print(calculate_attention_score(input_embeddings[0]))
print(calculate_context_vctor(input_embeddings[0]))
# print(calculate_context_vctor_no_batch(input_embeddings[0], 0))

