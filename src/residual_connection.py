import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sublayer(x)


if __name__ == "__main__":
    X = torch.randn(3, 8)

    attn_output = torch.randn(3, 8)

    output = X + attn_output
