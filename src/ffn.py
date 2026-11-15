import torch
import torch.nn as nn



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.W1(x)
        x = self.act(x)
        x = self.W2(x)
        return x


if __name__ == "__main__":
    d_model = 4
    d_ff = 16

    ffn = FeedForward(d_model, d_ff)

    x = torch.randn(3, 4)
    out = ffn(x)
    print(out.shape)
