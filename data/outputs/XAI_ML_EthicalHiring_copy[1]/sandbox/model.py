import torch
import torch.nn as nn


class HiringModel(nn.Module):
    def __init__(self, input_dim: int):
        super(HiringModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
