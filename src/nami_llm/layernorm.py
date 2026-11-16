import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # dim参数代表运算作用在张量的哪一个维度
        # keepdim默认False，运算后会从张量中把这个维度删掉
        # unbiased表示无偏方差N-1，这里不需要
        # x: [batch, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)   # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # [batch, seq_len, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # 防止某个词向量的方差为0或极小导致数值爆炸
        return self.gamma * x_norm + self.beta


if __name__ == "__main__":
    torch.manual_seed(42)
    x1 = torch.randn(1, 3, 8)

    ln = LayerNorm(d_model=8)
    out = ln(x1)

    official_ln = nn.LayerNorm(8)
    out_official = official_ln(x1)

    print("手写实现输出 shape:", out.shape)        # [1, 3, 8]
    print("官方实现输出 shape:", out_official.shape)  # [1, 3, 8]

    # 验证核心结论:去掉 gamma/beta 之后,每一行的均值应约等于0,方差应约等于1
    x_norm_only = (x1 - x1.mean(dim=-1, keepdim=True)) / torch.sqrt(
        x1.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
    )
    print("每行均值(应≈0):", x_norm_only.mean(dim=-1))
    print("每行方差(应≈1):", x_norm_only.var(dim=-1, unbiased=False))

    # 验证模长:方差=1 意味着模长应该是 sqrt(d_model) = sqrt(8) ≈ 2.828
    norm_length = x_norm_only.norm(dim=-1)
    print("每行模长(应≈√8≈2.828):", norm_length)
