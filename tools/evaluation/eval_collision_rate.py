#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# >>> reuse your project code <<<
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config

def _require(run_cfg: dict, key: str):
    if key not in run_cfg:
        raise KeyError(f"Missing '{key}' in run_config saved to configs_used.json")
    return run_cfg[key]

def _restore_normalizer(state: dict) -> Normalizer:
    """Rebuild your Normalizer from norm.json using the real NormStats dataclass."""
    mode = state.get("mode", "none")
    cols = state.get("cols", None)
    norm = Normalizer(mode, cols=cols)
    norm.fitted = bool(state.get("fitted", False))
    stats = state.get("stats", None)
    if stats is not None:
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std  = torch.tensor(stats["std"],  dtype=torch.float32)
        norm.stats = NormStats(mean=mean, std=std)
    else:
        norm.stats = None
    return norm

def main():
    # YAML-only: only --run-dir is accepted
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory with model_best.pt & configs_used.json")
    ap.add_argument("--check-gt", action="store_true",
                    help="Also evaluate ground-truth collisions on the test split")
    args = ap.parse_args()

    run_dir    = Path(args.run_dir).resolve()
    cfgs_path  = run_dir / "configs_used.json"
    norm_path  = run_dir / "norm.json"
    split_path = run_dir / "split_indices.json"
    meta_path  = run_dir / "meta.json"
    ckpt_path  = run_dir / "model_best.pt"

    for p in [cfgs_path, norm_path, split_path, meta_path, ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    with cfgs_path.open() as f:
        cfgs = json.load(f)
    data_cfg  = cfgs["data_config"]
    model_cfg = cfgs["model_config"]
    run_cfg   = cfgs["run_config"]

    # ---- Require params from YAML (no CLI defaults, no eps) ----
    device_str = str(_require(run_cfg, "device"))          # "cpu" | "cuda" | "auto"
    batch_size = int(_require(run_cfg, "batch_size"))
    radius = 0.099

    out_csv = "collisions_pred_test.csv"
    #out_csv    = str(_require(run_cfg, "out_csv"))

    with norm_path.open() as f:
        norm_state = json.load(f)
    with split_path.open() as f:
        splits = json.load(f)
    test_idx = splits.get("test", [])
    if not test_idx:
        raise RuntimeError("No 'test' indices in split_indices.json")

    with meta_path.open() as f:
        meta = json.load(f)["meta"]

    # ---- Resolve Excel path ----
    excel_path = Path(meta.get("excel_path", data_cfg.get("excel_path", ""))).resolve()
    if not excel_path.exists():
        # fallback to repo-root relative (configs/../../data/...)
        excel_maybe = (Path(__file__).resolve().parents[2] / data_cfg["excel_path"]).resolve()
        if excel_maybe.exists():
            excel_path = excel_maybe
        else:
            raise FileNotFoundError(f"Excel not found: {excel_path}")

    # ---- Build dataset via your schema & dataset ----
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[tuple(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 2)),
    )
    raw = TrajectoryDataset(str(excel_path), schema, dtype=torch.float32, task_name=data_cfg.get("task_name", ""))

    # Clamp test indices to current Excel length
    test_idx = [i for i in test_idx if 0 <= i < len(raw)]
    if not test_idx:
        raise RuntimeError("After clamping, no valid test indices remain.")
    test_ds = Subset(raw, test_idx)

    # ---- Restore normalizers saved during training ----
    x_norm = _restore_normalizer(norm_state["inputs"])
    y_norm = _restore_normalizer(norm_state["outputs"])
    if not (x_norm.fitted and y_norm.fitted):
        raise RuntimeError("Normalizers in norm.json are not marked 'fitted'")

    # ---- Device from YAML ----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # ---- Build exact model from saved model_config, then load weights ----
    model = build_model_from_config(model_cfg, input_dim=raw.input_dim, output_dim=raw.output_dim).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # ---- Inference on test (normalize -> predict -> denormalize) ----
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    preds, X_rows = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb_n     = x_norm.transform(xb)
            y_pred_n = model(xb_n.to(device)).cpu()
            y_pred   = y_norm.inverse(y_pred_n)  # back to original units
            preds.append(y_pred)
            X_rows.append(xb)                    # raw inputs for ox, oy

    Y_pred = torch.cat(preds,  dim=0).numpy()   # [N, Dout]
    X_raw  = torch.cat(X_rows, dim=0).numpy()   # [N, Din]

    # ---- Split predictions into (x_traj, y_traj) using K ----
    K = raw.K
    assert Y_pred.shape[1] % K == 0, f"D_out must be divisible by K={K}"
    T = Y_pred.shape[1] // K
    x_traj = Y_pred[:, 0:T]
    y_traj = Y_pred[:, T:2*T]

    # ---- Ox/Oy are last two inputs per your schema ----
    ox = X_raw[:, -2].reshape(-1, 1)
    oy = X_raw[:, -1].reshape(-1, 1)

    # ---- Euclidean check: PREDICTIONS ----
    dx = x_traj - ox
    dy = y_traj - oy
    dist = np.sqrt(dx*dx + dy*dy)         # [N, T]
    min_distance_pred = dist.min(axis=1)  # [N]
    t_argmin_pred = dist.argmin(axis=1)   # [N]
    collided_pred = min_distance_pred <= radius

    # Collect results into dict (predictions only by default)
    results = {
        "row_index": test_idx,
        "min_distance_pred": min_distance_pred,
        "collided_pred": collided_pred.astype(bool),
        "t_argmin_pred": t_argmin_pred,
    }

    # ---- Optional: Euclidean check on GROUND TRUTH (enable with --check-gt) ----
    if args.check_gt:
        Y_gt = raw.outputs[test_idx].numpy()   # [N, Dout] raw ground-truth outputs
        T_gt = Y_gt.shape[1] // raw.K
        x_gt = Y_gt[:, 0:T_gt]
        y_gt = Y_gt[:, T_gt:2*T_gt]

        dx_gt = x_gt - ox
        dy_gt = y_gt - oy
        dist_gt = np.sqrt(dx_gt*dx_gt + dy_gt*dy_gt)   # [N, T_gt]
        min_distance_gt = dist_gt.min(axis=1)
        t_argmin_gt = dist_gt.argmin(axis=1)
        collided_gt = min_distance_gt <= radius

        results.update({
            "min_distance_gt": min_distance_gt,
            "collided_gt": collided_gt.astype(bool),
            "t_argmin_gt": t_argmin_gt,
        })

    # ---- Save CSV (path from YAML; relative -> inside run_dir) ----
    # ---- Save summary (collision %) instead of all rows ----
    out_path = Path(out_csv)
    if not out_path.is_absolute():
        out_path = run_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(test_idx)
    pred_collisions = int(collided_pred.sum())
    pred_rate = 100.0 * pred_collisions / n

    summary = {
        "n_test": n,
        "pred_collisions": pred_collisions,
        "pred_collision_rate": pred_rate,
    }

    if args.check_gt:
        gt_collisions = int(collided_gt.sum())
        gt_rate = 100.0 * gt_collisions / n
        summary.update({
            "gt_collisions": gt_collisions,
            "gt_collision_rate": gt_rate,
        })

    # write one-row CSV
    pd.DataFrame([summary]).to_csv(out_path, index=False)

    # (optional) also write a plain text file with just the % for easy parsing
    (rate_path := out_path.with_suffix(".txt")).write_text(f"{pred_rate:.6f}\n")

    # Console summary
    if args.check_gt:
        print(f"[DONE] Test rows: {n} | Pred collisions: {pred_collisions} ({pred_rate:.2f}%)"
              f" | GT collisions: {gt_collisions} ({gt_rate:.2f}%) | T={T}")
    else:
        print(f"[DONE] Test rows: {n} | Pred collisions: {pred_collisions} ({pred_rate:.2f}%) | T={T}")

    print(f"[SAVED] {out_path}")
    print(f"[SAVED] {rate_path}")  # if you keep the .txt too


if __name__ == "__main__":
    main()
