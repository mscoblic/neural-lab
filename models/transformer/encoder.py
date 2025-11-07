import torch.nn as nn

from models.transformer.positional_encoding import PositionalEncoding
from models.transformer.encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):
    """
    Full Transformer encoder stack.
    Behavior is identical to your original implementation.
    """

    def __init__(
        self,
        input_dim,      # raw input dimension (3)
        d_model,        # model width
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        output_dim=None   # None → return embeddings only
    ):
        super().__init__()

        # Project tokens into model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Add positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_seq_length)

        # Stack encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

        # Optional output projection
        self.head = nn.Linear(d_model, output_dim) if output_dim is not None else None

    def forward(self, x, mask=None):
        """
        x: (B, seq_len, input_dim)
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)

        for layer in self.layers:
            h = layer(h, mask=mask)

        h = self.norm(h)

        return self.head(h) if self.head is not None else h
