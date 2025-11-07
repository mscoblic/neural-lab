import torch.nn as nn

from models.transformer.attention import MultiHeadAttention
from models.transformer.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    One Transformer encoder block.

    Structure:
        x → SelfAttention → Add+Norm →
            FeedForward → Add+Norm → output

    Identical behavior to your original implementation.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Self-attention ---
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # --- Feedforward ---
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
