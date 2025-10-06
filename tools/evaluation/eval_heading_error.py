#!/usr/bin/env python3
"""
check_heading_error_test.py

Computes heading error between predicted and ground-truth trajectories
on the test split.

- Reuses project code: DataSchema, TrajectoryDataset, Normalizer, NormStats
- Reads configs_used.json, norm.json, split_indices.json, meta.json from the run dir
- Uses YAML-driven config (no CLI defaults except --run-dir)

Heading definition:
  heading_t = atan2(y[t+1] - y[t], x[t+1] - x[t])
Error is absolute angular difference in radians, mean over timesteps and batches.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config


def _restore_normalizer(state: dict) -> Normalizer:
    norm = Normalizer(state.get("mode", "none"), cols=state.get("cols", None))
    norm.fitted = bool(state.get("fitted", False))
    stats = state.get("stats", None)
    if stats:
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        norm.stats = NormStats(mean=mean, std=std)
    return norm


def compute_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x, y: [N, T] trajectories
    returns: [N, T-1] heading angles in radians
    """
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    return np.arctan2(dy, dx)


def angular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute minimal angular difference |a-b| in radians in [-pi, pi].
    """
    diff = a - b
    return np.abs((diff + np.pi) % (2 * np.pi) - np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory with model_best.pt and configs_used.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfgs_path = run_dir / "configs_used.json"
    norm_path = run_dir / "norm.json"
    split_path = run_dir / "split_indices.json"
    meta_path = run_dir / "meta.json"
    ckpt_path = run_dir / "model_best.pt"

    for p in [cfgs_path, norm_path, split_path, meta_path, ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    with cfgs_path.open() as f:
        cfgs = json.load(f)
    data_cfg, model_cfg, run_cfg = cfgs["data_config"], cfgs["model_config"], cfgs["run_config"]

    with norm_path.open() as f:
        norm_state = json.load(f)
    with split_path.open() as f:
        splits = json.load(f)
    test_idx = splits.get("test", [])
    if not test_idx:
        raise RuntimeError("No 'test' indices in split_indices.json")

    with meta_path.open() as f:
        meta = json.load(f)["meta"]

    # Dataset
    excel_path = Path(meta.get("excel_path", data_cfg.get("excel_path", ""))).resolve()
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[tuple(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 2)),
    )
    raw = TrajectoryDataset(str(excel_path), schema, dtype=torch.float32, task_name=data_cfg.get("task_name", ""))
    test_idx = [i for i in test_idx if 0 <= i < len(raw)]
    test_ds = Subset(raw, test_idx)

    # Normalizers
    x_norm = _restore_normalizer(norm_state["inputs"])
    y_norm = _restore_normalizer(norm_state["outputs"])

    # Device
    device_str = run_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Model
    model = build_model_from_config(model_cfg, input_dim=raw.input_dim, output_dim=raw.output_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Inference
    loader = DataLoader(test_ds, batch_size=int(run_cfg.get("batch_size", 64)), shuffle=False)
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb_n = x_norm.transform(xb)
            y_pred_n = model(xb_n.to(device)).cpu()
            y_pred = y_norm.inverse(y_pred_n)
            preds.append(y_pred)
            gts.append(yb)
    Y_pred = torch.cat(preds, dim=0).numpy()
    Y_gt = torch.cat(gts, dim=0).numpy()

    # Split into trajectories
    K = raw.K
    T = Y_pred.shape[1] // K
    x_pred, y_pred = Y_pred[:, :T], Y_pred[:, T:2*T]
    x_gt,   y_gt   = Y_gt[:,   :T], Y_gt[:,   T:2*T]

    # Headings
    heading_pred = compute_heading(x_pred, y_pred)
    heading_gt   = compute_heading(x_gt,   y_gt)

    # Errors
    err = angular_error(heading_pred, heading_gt)   # [N, T-1]
    mean_err_per_traj = err.mean(axis=1)
    overall_mean = mean_err_per_traj.mean()
    overall_median = np.median(mean_err_per_traj)

    # ---- Save one-row summary instead of per-row CSV ----
    n = len(test_idx)
    overall_mean_deg = float(np.degrees(overall_mean))
    overall_median_deg = float(np.degrees(overall_median))
    overall_std = float(mean_err_per_traj.std())
    overall_std_deg = float(np.degrees(overall_std))

    summary = {
        "n_test": n,
        "mean_heading_error_rad": float(overall_mean),
        "mean_heading_error_deg": overall_mean_deg,
        "median_heading_error_rad": float(overall_median),
        "median_heading_error_deg": overall_median_deg,
        "std_heading_error_rad": overall_std,
        "std_heading_error_deg": overall_std_deg,
    }

    out_path = run_dir / "heading_summary_test.csv"  # name it as you like
    pd.DataFrame([summary]).to_csv(out_path, index=False)

    # (optional) also save a tiny .txt with just the mean in radians for easy parsing
    (out_path.with_suffix(".txt")).write_text(f"{overall_mean:.6f}\n")

    print(f"[DONE] Mean heading error = {overall_mean:.6f} rad "
          f"({overall_mean_deg:.3f} deg), "
          f"Median = {overall_median:.6f} rad "
          f"({overall_median_deg:.3f} deg)")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
