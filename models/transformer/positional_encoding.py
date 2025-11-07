import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Behavior unchanged from your original implementation.
    """

    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Compute the div_term (same formula you used)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_seq, d_model)

    def forward(self, x):
        """
        x: (B, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]
