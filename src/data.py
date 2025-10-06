"""
File: data.py
Author: Mia Scoblic
Date: 2025-08-15
Description:
This module manages datasets and normalization. It loads raw data from Excel,
applies schemas to select input/output columns, and prepares train/val/test splits.
It also handles normalization so models train on standardized values, and provides
utilities to wrap datasets with the correct transforms.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any
from torch.utils.data import Dataset, random_split, Subset
import numpy as np
import pandas as pd
import torch

# ----------------------------------------------------------------------------------------------------------------------
# Defining data classes
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class DataSchema:
    header: Optional[int]
    input_cols: Sequence[int]
    output_slices: Sequence[Tuple[int, int]]
    infer_T_from_outputs: bool
    K: int


@dataclass
class DataConfig:
    excel_path: str
    schema: DataSchema
    normalization: Dict[str, str]
    task_name: str

@dataclass
class NormStats:
    mean: torch.Tensor
    std: torch.Tensor

# ----------------------------------------------------------------------------------------------------------------------
# Converts excel data into tensors
# ----------------------------------------------------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """
    A simple dataset for trajectory data stored in Excel.

    It reads the file, pulls out the input and output columns defined
    in the schema, and stores them as PyTorch tensors.

    """
    def __init__(self, excel_path: str, schema: DataSchema, dtype=torch.float32, task_name: str = ""):
        self.task_name = task_name

        # Read Excel
        df = pd.read_excel(excel_path, header=schema.header, engine="openpyxl")

        # Inputs
        X = (
            df.iloc[:, list(schema.input_cols)]
              .to_numpy(dtype=np.float32)
        )
        # Outputs
        parts = []
        total_out = 0
        for start, stop in schema.output_slices:
            arr = (
                df.iloc[:, start:stop]
                  .to_numpy(dtype=np.float32)
            )
            parts.append(arr)
            total_out += (stop - start)
        Y = np.hstack(parts).astype(np.float32, copy=False)

        # ----------------------------------------------------------------------------------
        # Validation: no NaNs or infinities or uneven outputs allowed
        # ----------------------------------------------------------------------------------
        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            raise ValueError(
                f"Invalid numeric values in inputs at positions {bad[:10]} "
                f"(showing first 10)."
            )
        if not np.isfinite(Y).all():
            bad = np.argwhere(~np.isfinite(Y))
            raise ValueError(
                f"Invalid numeric values in outputs at positions {bad[:10]} "
                f"(showing first 10)."
            )

        K = schema.K
        assert total_out % K == 0, f"Total output columns must be divisible by {K} (got {total_out})."
        T = total_out // K

        # Save tensors
        self.inputs = torch.tensor(X, dtype=dtype)
        self.outputs = torch.tensor(Y, dtype=dtype)
        self.K = K
        self.T = T
        self.input_dim = self.inputs.shape[1]
        self.output_dim = self.outputs.shape[1]

        # Summary print (useful during bring-up)
        print(f"[DATA] {Path(excel_path).name}: inputs {self.inputs.shape} -> outputs {self.outputs.shape}  (T={self.T})")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def _compute_stats(t: torch.Tensor) -> NormStats:
    mean = t.mean(dim=0)
    std = t.std(dim=0, unbiased=False)
    # avoid divide-by-zero
    std = torch.where(std == 0, torch.ones_like(std), std)
    return NormStats(mean=mean, std=std)

# ------------------------------v---------------------------------------------------------------------------------------

# ------------------------------v---------------------------------------------------------------------------------------
class Normalizer:
    """Column-wise standardization or identity (optionally on a column subset)."""
    def __init__(self, mode: str, cols: Optional[Sequence[int]] = None):
        mode = str(mode).lower()
        assert mode in ("standardize", "none")      # add other normalization methods here
        self.mode = mode
        self.cols = None if cols is None else list(cols)
        self.fitted = False
        self.stats: Optional[NormStats] = None

    def fit(self, t: torch.Tensor):
        if self.mode == "standardize":
            if self.cols is None:
                # standardize all columns
                self.stats = _compute_stats(t)
            else:
                # standardize only selected columns
                mean = torch.zeros(t.shape[1], dtype=t.dtype)       # initialize mean vector
                std = torch.ones(t.shape[1], dtype=t.dtype)         # initialize std vector
                sub = t[:, self.cols]
                sub_stats = _compute_stats(sub)         # compute mean/std
                mean[self.cols] = sub_stats.mean        # populate mean
                std[self.cols] = sub_stats.std          # populate std
                self.stats = NormStats(mean=mean, std=std)      # store together
        self.fitted = True      # flag verifying the data has been normalized

    # Apply normalization to all data sets
    def transform(self, t: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return t
        assert self.fitted and self.stats is not None, "Normalizer not fit"
        return (t - self.stats.mean) / self.stats.std

    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return t
        assert self.fitted and self.stats is not None, "Normalizer not fit"
        return t * self.stats.std + self.stats.mean

    # Creates a dictionary capturing attributes of normalization for saving
    def state_dict(self) -> Dict[str, Any]:
        if self.mode == "none" or not self.fitted or self.stats is None:
            return {"mode": self.mode, "cols": self.cols, "fitted": self.fitted, "stats": None}
        return {
            "mode": self.mode,
            "cols": self.cols,
            "fitted": self.fitted,
            "stats": {
                "mean": self.stats.mean.tolist(),
                "std": self.stats.std.tolist(),
            },
        }

class NormalizedDataset(Dataset):
    def __init__(self, base: TrajectoryDataset,
                 x_norm: Normalizer, y_norm: Normalizer):
        self.base = base
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.T = base.T
        self.input_dim = base.input_dim
        self.output_dim = base.output_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.x_norm.transform(x), self.y_norm.transform(y)

# left off
def prepare_datasets(raw: TrajectoryDataset,
                     norm_inputs: str, norm_outputs: str,
                     val_ratio: float = 0.1, test_ratio: float = 0.0,
                     seed: int = 42,
                     split_indices: Optional[Dict[str, Sequence[int]]] = None
                     ) -> Tuple[NormalizedDataset, NormalizedDataset, Optional[NormalizedDataset], Dict[str, Any]]:
    """
    Returns (train_ds, val_ds, test_ds_or_None, norm_state_dict)

    If split_indices is provided, it must be a dict with keys 'train','val', and
    optionally 'test', each mapping to a list of integer indices into `raw`.

    If split_indices is None, we create splits using val_ratio/test_ratio and the seed.
    """

    # If repeated split indices requested
    if split_indices is not None:
        train_idx = list(split_indices["train"])
        val_idx   = list(split_indices["val"])
        test_idx  = list(split_indices.get("test", [])) # Test split can be empty
        train_raw = Subset(raw, train_idx)
        val_raw   = Subset(raw, val_idx)
        test_raw  = Subset(raw, test_idx) if test_idx else None
    # If generating a new split
    else:
        n_total = len(raw)
        n_test = int(round(n_total * test_ratio))
        n_val  = int(round(n_total * val_ratio))
        n_train = n_total - n_val - n_test
        assert n_train > 0, "Train split must be > 0."
        g = torch.Generator().manual_seed(seed)

        # Test split can be empty
        if n_test > 0:
            train_raw, val_raw, test_raw = random_split(raw, [n_train, n_val, n_test], generator=g)
        else:
            train_raw, val_raw = random_split(raw, [n_train, n_val], generator=g)
            test_raw = None
        # Used for normalizing only on training data
        train_idx = train_raw.indices if hasattr(train_raw, "indices") else list(range(len(train_raw)))

    # ------------------------------------------------------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------------------------------------------------------

    # Grab the normalization method from YAML
    def _mode_from(x):
        if isinstance(x, dict):
            if "mode" not in x:
                raise ValueError("Normalizer config dict must include a 'mode' key (e.g., 'standardize' or 'none').")
            return str(x["mode"]).lower()
        return str(x).lower()

    # For inputs/ouputs
    mode_in  = _mode_from(norm_inputs)
    mode_out = _mode_from(norm_outputs)

    # Include only desired input features
    subset_cols = None
    if isinstance(norm_inputs, dict):       # always true at this point
        subset_cols = norm_inputs.get("include_indices")

    # left off
    x_norm = Normalizer(mode_in, cols=subset_cols)
    y_norm = Normalizer(mode_out)

    x_list, y_list = [], []
    for i in train_idx:
        x_i, y_i = raw[i]
        x_list.append(x_i)
        y_list.append(y_i)
    x_train = torch.stack(x_list, dim=0)
    y_train = torch.stack(y_list, dim=0)

    # mean and std are stored in the x/y_norm instance
    x_norm.fit(x_train)
    y_norm.fit(y_train)

    # Wrap subsets
    def _wrap(subset):
        if subset is None: return None
        return NormalizedDataset(_SubsetView(raw, subset), x_norm, y_norm)

    # returns normalized data, normalized off of the training data set
    train_ds = _wrap(train_raw)
    val_ds   = _wrap(val_raw)
    test_ds  = _wrap(test_raw)

    # Keeps tracks of all normalization related data
    norm_state = {
        "inputs": x_norm.state_dict(),
        "outputs": y_norm.state_dict(),
        "T": raw.T,
        "input_dim": raw.input_dim,
        "output_dim": raw.output_dim,
        "K": raw.K,
    }
    return train_ds, val_ds, test_ds, norm_state


class _SubsetView(Dataset):
    def __init__(self, base: TrajectoryDataset, subset: Optional[torch.utils.data.Subset]):
        self.base = base
        if subset is None:
            self.indices = list(range(len(base)))
        else:
            self.indices = subset.indices

        # >>> add these three lines <<<
        self.T = base.T
        self.input_dim = base.input_dim
        self.output_dim = base.output_dim


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
