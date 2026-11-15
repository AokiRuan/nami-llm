import torch
import torch.nn as nn


x = torch.randn(3, 4)

W1 = nn.Linear(4, 16)
W2 = nn.Linear(16, 4)

relu = nn.ReLU()
gelu = nn.GELU()

hidden = W1(x)

hidden_relu = relu(hidden)
hidden_gelu = gelu(hidden)

out_relu = W2(hidden_relu)
out_gelu = W2(hidden_gelu)

print("ReLU 激活后有多少维度被清零：", (hidden_relu == 0).sum().item(), "/", hidden_relu.numel())
print("GELU 激活后有多少维度接近但不等于0：", ((hidden_gelu.abs() < 0.01) & (hidden_gelu != 0)).sum().item())