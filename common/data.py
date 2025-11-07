import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrajDataset(Dataset):
    """Simple dataset wrapper for (X, Y) tensors."""
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def build_XY_from_excel(df, output_cols):
    """
    Build X (tokens) and Y (trajectory control points) from the DataFrame.

    X shape → (N, 4, 3):
        - start (x0,y0,z0)
        - end (xf,yf,zf)
        - obstacle (ox,oy,oz)
        - control / initial velocity (vxinit,vyinit,vzinit)

    Y shape → (N, T_out, 3)
    """
    # Input tokens
    start    = df[["x0","y0","z0"]].to_numpy(np.float32)
    end      = df[["xf","yf","zf"]].to_numpy(np.float32)
    obstacle = df[["ox","oy","oz"]].to_numpy(np.float32)
    control  = df[["vxinit","vyinit","vzinit"]].to_numpy(np.float32)

    # Stack 4 tokens
    X = np.stack([start, end, obstacle, control], axis=1)  # (N,4,3)

    # Outputs
    Y_raw = df[output_cols].to_numpy(np.float32)           # (N, 3*T_out)
    T_out = len(output_cols) // 3
    N = len(df)

    # reshape (N, 3*T_out) → (N, T_out, 3)
    Y = Y_raw.reshape(N, 3, T_out).transpose(0, 2, 1)

    return X, Y, T_out


def split_indices(N, frac=0.8, seed=42):
    """Random train/test split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(frac * N)
    return perm[:n_train], perm[n_train:]


def compute_norm_stats(X_train, Y_train, eps=1e-8):
    """
    Compute per-token mean/std for X and Y.
    Shapes:
        X_mean, X_std → (1, 4, 3)
        Y_mean, Y_std → (1, T_out, 3)
    """
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std  = X_train.std(axis=0, keepdims=True) + eps

    Y_mean = Y_train.mean(axis=0, keepdims=True)
    Y_std  = Y_train.std(axis=0, keepdims=True) + eps

    return X_mean, X_std, Y_mean, Y_std


def apply_norm(X, mean, std):
    return (X - mean) / std


def apply_denorm(X, mean, std):
    return X * std + mean
