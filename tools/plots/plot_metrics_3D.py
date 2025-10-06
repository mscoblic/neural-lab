#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def smooth(values, window: int = 1):
    """Simple moving average."""
    if window <= 1:
        return values
    return values.rolling(window, min_periods=1, center=True).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to run folder (under runs/)")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window size")
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")

    # Load metrics
    df = pd.read_csv(csv_path)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], smooth(df["train_mse"], args.smooth),
             label="train MSE", color="tab:blue")
    plt.plot(df["epoch"], smooth(df["val_mse"], args.smooth),
             label="val MSE", color="tab:orange")

    # Handle 3D case
    if "val_mse_x" in df.columns and "val_mse_y" in df.columns and "val_mse_z" in df.columns:
        plt.plot(df["epoch"], smooth(df["val_mse_x"], args.smooth),
                 label="val MSE x", color="tab:green", linestyle="--")
        plt.plot(df["epoch"], smooth(df["val_mse_y"], args.smooth),
                 label="val MSE y", color="tab:red", linestyle="--")
        plt.plot(df["epoch"], smooth(df["val_mse_z"], args.smooth),
                 label="val MSE z", color="tab:purple", linestyle="--")
    elif "val_mse_x" in df.columns and "val_mse_y" in df.columns:
        # fallback for 2D
        plt.plot(df["epoch"], smooth(df["val_mse_x"], args.smooth),
                 label="val MSE x", color="tab:green", linestyle="--")
        plt.plot(df["epoch"], smooth(df["val_mse_y"], args.smooth),
                 label="val MSE y", color="tab:red", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(run_dir.name)
    if args.logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_png = run_dir / "metrics_3d.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
