import torch

from torch_based_data import load_verdict_txt_dataloader


vocab_size = 50257
embedding_size = 256

embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

context_length = 6

dataloader = load_verdict_txt_dataloader(batch_size=8, max_length=context_length, shuffle=True, stride=context_length)

data_iter = iter(dataloader)
inputs, tagets = next(data_iter)

token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)

#TODO: why not to have additional dimension for the orgiinal embedding and just be saving position there?
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_size)
range_tensor = torch.arange(context_length)
print(range_tensor)
pos_embeddings = pos_embedding_layer(range_tensor)
print(pos_embeddings.shape)
