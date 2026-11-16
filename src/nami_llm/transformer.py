import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True)
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


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        d_ff = 4 * d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享（weight tying）：输出头和 token embedding 用同一个矩阵，GPT-2 就是这样
        self.head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device) # 给序列加上位置编号，张量生成位置要和idx一样
        x = self.token_emb(idx) + self.pos_emb(pos) # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits



if __name__ == "__main__":
    # 猫吃鱼验证
    d_model, n_heads, d_ff = 4, 2, 16
    block = TransformerBlock(d_model, n_heads, d_ff)

    x = torch.randn(1, 3, d_model)  # 3个token(猫/吃/鱼),4维embedding
    out = block(x)

    print(out.shape)  # torch.Size([1, 3, 4]) —— 和输入一模一样


    model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
    x = torch.tensor([[464, 3797, 25365]])
    logits = model(x)
    print(logits.shape) # torch.Size([1, 3, 50257])