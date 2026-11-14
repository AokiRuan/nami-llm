import torch
import math


vocab = {"猫": 0, "吃": 1, "鱼": 2}
d_model = 8
max_len = 10 # 预先算好足够长度的位置编码，用多少截多少


def get_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = get_positional_encoding(max_len, d_model)


torch.manual_seed(42)
embedding = torch.nn.Embedding(len(vocab), d_model)


def encode(sentence_tokens):
    ids = torch.tensor([vocab[t] for t in sentence_tokens])
    emb = embedding(ids)
    pos_enc = pe[: len(sentence_tokens)]
    return emb + pos_enc


cat_eat_fish = encode(["猫", "吃", "鱼"])
fish_eat_cat = encode(["鱼", "吃", "猫"])

print("猫吃鱼:\n", cat_eat_fish)
print("鱼吃猫:\n", fish_eat_cat)
