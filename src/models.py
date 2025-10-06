from __future__ import annotations
from typing import Callable, Dict, Literal, Optional
import torch
import torch.nn as nn

# -----------------------------
# Feedforward Network
# -----------------------------
class FeedForwardNet(nn.Module):
    """
    A simple MLP that maps (batch, input_dim) -> (batch, output_dim).
    Depth/width/activation are configurable; last layer is linear.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        activation: Literal["relu", "gelu", "tanh", "silu", "swish", "leaky_relu", "lrelu"] = "relu",
        dropout: float = 0.0,
        activation_kwargs: Optional[dict] = None,  # NEW: for negative_slope, etc.
    ):
        super().__init__()
        assert n_hidden >= 1, "n_hidden must be >= 1"
        activation_kwargs = activation_kwargs or {}

        # Builders allow passing kwargs like negative_slope for LeakyReLU
        act_builders: Dict[str, Callable[..., nn.Module]] = {
            "relu":        nn.ReLU,
            "gelu":        nn.GELU,
            "tanh":        nn.Tanh,
            "silu":        nn.SiLU,     # Swish
            "swish":       nn.SiLU,     # alias
            "leaky_relu":  nn.LeakyReLU,
            "lrelu":       nn.LeakyReLU # alias
        }
        if activation not in act_builders:
            raise ValueError(f"Unsupported activation: {activation}")

        def make_act():
            # Sensible default for LeakyReLU
            if activation in {"leaky_relu", "lrelu"} and "negative_slope" not in activation_kwargs:
                activation_kwargs["negative_slope"] = 0.01
            return act_builders[activation](**activation_kwargs)

        layers = []
        # First hidden block
        layers += [nn.Linear(input_dim, hidden_dim), make_act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Additional hidden blocks
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), make_act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output head (no activation)
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------
# Factory helpers (unchanged)
# -----------------------------
def make_model(
    kind: Literal["ffn"],
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> nn.Module:
    if kind == "ffn":
        return FeedForwardNet(input_dim, output_dim, **kwargs)
    raise ValueError(f"Unknown model kind: {kind}")


def build_model_from_config(
    model_cfg: dict,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    kind = model_cfg.get("kind")
    params = model_cfg.get(kind, {}) or {}
    return make_model(kind, input_dim, output_dim, **params)
