import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.
    Behavior is unchanged from your original code.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        q, k, v: (B, heads, seq, d_k)
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out

    def split_heads(self, x):
        """
        x: (B, seq, d_model)
        → (B, heads, seq, d_k)
        """
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        x: (B, heads, seq, d_k)
        → (B, seq, d_model)
        """
        B, _, T, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        attn_out = self.scaled_dot_product_attention(Q, K, V, mask)
        out = self.W_o(self.combine_heads(attn_out))
        return out
