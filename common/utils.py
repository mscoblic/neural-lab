import logging
import torch

def setup_logging(verbosity: int = 1):
    """Configure logging level."""
    level = (
        logging.DEBUG if verbosity > 1
        else logging.INFO if verbosity == 1
        else logging.WARNING
    )
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def to_np(t):
    """Convert a PyTorch tensor to a numpy array."""
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
