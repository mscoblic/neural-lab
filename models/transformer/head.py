import torch.nn as nn


class FlattenMLPHead(nn.Module):
    """
    After the encoder outputs (B, seq_len, d_model),
    this MLP flattens the sequence and predicts
    T_out × 3 control points.

    Behavior is identical to your original implementation.
    """

    def __init__(self, seq_len, d_model, hidden, t_out, out_dim):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.t_out = t_out
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(seq_len * d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, t_out * out_dim)
        )

    def forward(self, h):  # h: (B, seq_len, d_model)
        B, L, D = h.shape
        flat = h.reshape(B, L * D)
        out = self.net(flat)                  # (B, t_out*out_dim)
        return out.view(B, self.t_out, self.out_dim)  # (B, T_out, 3)


class TrajModel(nn.Module):
    """
    Simple wrapper: encoder + MLP head.

    model(x) → predict trajectory control points.
    """

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        h = self.encoder(x)     # (B, seq_len, d_model)
        return self.head(h)     # (B, T_out, 3)
