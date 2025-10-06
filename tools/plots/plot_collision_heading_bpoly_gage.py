from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config import load_yaml
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config


# ---------- Restore normalizers ----------
def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        norm = json.load(f)
    x_state = norm["inputs"]; y_state = norm["outputs"]
    x_norm = Normalizer(x_state["mode"]); y_norm = Normalizer(y_state["mode"])

    def _restore(nstate, norm_obj):
        if nstate["stats"] is None:
            norm_obj.fitted = True
            norm_obj.stats = None
        else:
            mean = torch.tensor(nstate["stats"]["mean"], dtype=torch.float32)
            std  = torch.tensor(nstate["stats"]["std"],  dtype=torch.float32)
            norm_obj.fitted = True
            norm_obj.stats = NormStats(mean=mean, std=std)

    _restore(x_state, x_norm)
    _restore(y_state, y_norm)
    return x_norm, y_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a specific run folder under runs/")
    # default to your *gage* config with raw heading angle
    ap.add_argument("--data-config",  default="configs/collision_heading_bpoly_gage/data_collision_heading_bpoly_gage.yaml")
    ap.add_argument("--model-config", default="configs/collision_heading_bpoly_gage/model_collision_heading_bpoly_gage.yaml")
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=1)

    # If dataset GT is stored normalized, invert before plotting
    ap.add_argument("--outputs-are-normalized", action="store_true")

    # Obstacle radius (center comes from the sample: ox, oy)
    ap.add_argument("--radius", type=float, default=0.10)

    args = ap.parse_args()

    # Project root (this file lives under tools/plots/)
    root = Path(__file__).resolve().parent.parent.parent

    run_dir   = (root / args.run_dir).resolve()
    ckpt_path = run_dir / "model_best.pt"
    norm_path = run_dir / "norm.json"

    # Load split indices produced during training
    split_path = run_dir / "split_indices.json"
    with split_path.open("r") as f:
        splits = json.load(f)
    indices = splits[args.subset]

    # Load normalizers
    x_norm, y_norm = load_norm(norm_path)

    # Dataset and schema
    data_cfg   = load_yaml(root / args.data_config)
    excel_path = (root / data_cfg["excel_path"]).resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],         # [x0, y0, xf, yf, theta(rad), ox, oy]
        output_slices=[tuple(s) for s in data_cfg["schema"]["output_slices"]],  # [[7,17],[17,27]]
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 1)),
    )
    ds = TrajectoryDataset(excel_path, schema)
    T  = ds.T  # number of control points per axis

    # Model
    model_cfg = load_yaml(root / args.model_config)
    model     = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state     = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    # ---- Plot samples ----
    for j, idx in enumerate(indices[:args.num_samples]):
        # Fetch one sample
        x_row, y_true_row = ds[idx]   # x_row: [x0, y0, xf, yf, theta(rad), ox, oy]
        x0, y0, xf, yf, theta, ox, oy = [x_row[k].item() for k in range(7)]

        # Prepare model input (normalize then predict)
        x_in = x_norm.transform(x_row.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)

        # Denormalize predictions to plot in original units
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()  # shape (2T,)
        pred_2xT = y_pred.view(2, T).numpy()

        # Ground truth control points (optionally inverse if dataset stored normalized)
        y_gt = y_true_row
        if args.outputs_are_normalized:
            with torch.no_grad():
                y_gt = y_norm.inverse(y_gt)
        gt_2xT = y_gt.detach().cpu().view(2, T).numpy()

        # -------------------- Plot --------------------
        fig, ax = plt.subplots(figsize=(6, 6))

        # Ground truth CPs: black crosses, slightly larger, behind predictions
        ax.scatter(
            gt_2xT[0], gt_2xT[1],
            marker="x",
            s=48,  # larger than preds
            color="black",
            linewidths=2,
            alpha=0.9,  # opaque enough to peek through
            label="ground truth",
            zorder=1
        )

        # Predicted CPs: circles on top
        ax.scatter(
            pred_2xT[0], pred_2xT[1],
            marker="o",
            s=36,
            alpha=0.95,
            label="prediction",
            zorder=2
        )

        # Start/end markers
        ax.scatter([x0], [y0], marker="o", s=80, label="start", zorder=3)
        ax.scatter([xf], [yf], marker="*", s=130, label="end", zorder=3)

        # Heading arrow (theta in radians from +x)
        L = 0.12
        ax.quiver(
            x0, y0, L * np.cos(theta), L * np.sin(theta),
            angles="xy", scale_units="xy", scale=1, width=0.004,
            zorder=3, label="heading"
        )

        # Obstacle
        circle = patches.Circle((ox, oy), radius=float(args.radius),
                                facecolor="red", edgecolor="black",
                                alpha=0.7, label="obstacle")
        ax.add_patch(circle)

        # Axes / cosmetics
        ax.set_xlabel("x");
        ax.set_ylabel("y")
        ax.axis("equal");
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{run_dir.name} | {args.subset} idx={idx}")
        ax.legend(loc="best", fontsize=9)

        out_png = run_dir / f"cp_only_gage_{args.subset}_{j:03d}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
