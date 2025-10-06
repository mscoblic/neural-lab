"""
File: utils.py
Author: Mia Scoblic
Date: 2025-08-15
Description:
This module contains helper functions for training and experiments. It includes
seeding for reproducibility, lightweight logging to CSV and JSON, and automatic
run name generation so experiments can be tracked and compared easily.
"""

from __future__ import annotations
import json, csv, time, random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import hashlib
from datetime import datetime

# Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Select computation device (CPU or CUDA) based on user preference or availability
def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save a Python dictionary as a formatted JSON file at the given path
def save_json(obj: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

# Lightweight CSV logger that appends rows to a file and writes headers if missing
class CSVLogger:
    def __init__(self, path: str | Path, header: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists()
        self.f = self.path.open("a", newline="")
        self.w = csv.writer(self.f)
        if write_header:
            self.w.writerow(header)

    def log(self, row: list[Any]):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

# Generate a unique run name using task/model info, date, seed, and a hash of config contents
def default_run_name(
    task_name: str,
    model_kind: str,
    seed: int,
    data_cfg_path: Path,
    model_cfg_path: Path,
    run_cfg_path: Path,
) -> str:
    """
    Run name: {task}_{modelkind}_{MMDDYY}_s{seed}_{hash6}
    where hash6 = first 6 hex chars of SHA1 over the *contents* of
    data/model/run YAMLs (guarantees uniqueness when small params change).
    """
    # Date stamp
    date_str = datetime.now().strftime("%m%d%y")

    # Read YAML contents and hash
    def _read(p: Path) -> str:
        return Path(p).read_text(encoding="utf-8")

    blob = _read(data_cfg_path) + _read(model_cfg_path) + _read(run_cfg_path)
    hash6 = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:6]

    return f"{task_name}_{model_kind}_{date_str}_s{seed}_{hash6}"

