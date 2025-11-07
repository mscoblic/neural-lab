import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Standard 2-layer feedforward block used inside transformer layers.
    Behavior identical to your original implementation.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
