import torch
import torch.nn as nn


class Simplemodel(nn.Module):
    """Generated model for: SimpleModel"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
