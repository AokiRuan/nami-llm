import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 矩阵乘法前必须先transpose，matmul是对最后两维起作用
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # contiguous() 是因为 transpose 之后张量在内存里不再连续存储，直接 view 会报错，需要先重新整理内存布局。
        # view把多头注意力再拼回8维
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(attn_output)
        return output, attn_weights


if __name__ == "__main__":
    vocab = {"猫": 0, "吃": 1, "鱼": 2}
    d_model = 8
    num_heads = 2

    torch.manual_seed(42)
    embedding = nn.Embedding(len(vocab), d_model)

    sentence = ["猫", "吃", "鱼"]
    # 外层list代表batch维度
    token_ids = torch.tensor([[vocab[w] for w in sentence]])
    print("tensor shape: ", token_ids.shape) # （1, 3）
    x = embedding(token_ids)  # (1, 3, 8)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, attn_weights = mha(x)

    print("输入 shape:", x.shape)              # torch.Size([1, 3, 8])
    print("输出 shape:", output.shape)          # torch.Size([1, 3, 8])
    print("注意力权重 shape:", attn_weights.shape)  # torch.Size([1, 2, 3, 3])
