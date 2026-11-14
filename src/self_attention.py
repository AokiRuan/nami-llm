import torch
import torch.nn.functional as F

X = torch.randn(3, 8)

d_model = 8
d_k = 8

W_q = torch.nn.Linear(d_model, d_k, bias=False)
W_k = torch.nn.Linear(d_model, d_k, bias=False)
W_v = torch.nn.Linear(d_model, d_k, bias=False)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

# 相似度
scores = Q @ K.T

scores = scores / (d_k ** 0.5)

attn_weights = F.softmax(scores, dim=-1)

output = attn_weights @ V

print("注意力权重矩阵：\n", attn_weights)
print("输出：\n", output)