import torch
import torch.nn as nn

vocab_size = 30

embedding_dim = 8

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

token_ids = torch.tensor([3, 15, 27])

vectors = embedding_layer(token_ids)

print(vectors.shape)

print(vectors)