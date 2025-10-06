"""
File: config.py
Author: Mia Scoblic
Date: 2025-08-15
Description:
This module provides utilities for loading YAML configuration files. It ensures
consistent handling of settings across training and plotting, making it easy to
define experiments without changing the core code.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

# Load a YAML configuration file into a dictionary
def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

# Make sure the file path is absolute
def resolve_path(root: str | Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else Path(root) / p
