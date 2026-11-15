import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x_norm1 = self.ln1(x)
        attn_out, _ = self.attn(x_norm1)
        x1 = x + attn_out

        x_norm2 = self.ln2(x1)
        ffn_out = self.ffn(x_norm2)
        x2 = x1 + ffn_out

        return x2


if __name__ == "__main__":
    # 猫吃鱼验证
    d_model, n_heads, d_ff = 4, 2, 16
    block = TransformerBlock(d_model, n_heads, d_ff)

    x = torch.randn(1, 3, d_model)  # 3个token(猫/吃/鱼),4维embedding
    out = block(x)

    print(out.shape)  # torch.Size([1, 3, 4]) —— 和输入一模一样
