from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config import load_yaml
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config
from tools.extra import *
import numpy as np

from tools.extra.BeBOT import BernsteinPoly, PiecewiseBernsteinPoly


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
    ap.add_argument("--data-config",  default="configs/collision_heading/data_collision_heading.yaml")
    ap.add_argument("--model-config", default="configs/collision_heading/model_collision_heading.yaml")
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=1)

    # Optional test-time injections; if omitted, use dataset values
    ap.add_argument("--heading", type=float, help="Radians relative to baseline (0.1,0.1)->(0.9,0.9)")
    ap.add_argument("--ox", type=float, help="Obstacle x")
    ap.add_argument("--oy", type=float, help="Obstacle y")
    ap.add_argument("--radius", type=float, default=0.1, help="Obstacle radius")

    # Optional overrides for start/end; otherwise dataset defaults
    ap.add_argument("--sx", type=float)
    ap.add_argument("--sy", type=float)
    ap.add_argument("--ex", type=float)
    ap.add_argument("--ey", type=float)

    # Normalization + plotting options
    ap.add_argument("--outputs-are-normalized", action="store_true",
                    help="If dataset stores normalized outputs, inverse-normalize GT before plotting.")
    ap.add_argument("--plot-gt", action="store_true",
                    help="Overlay ground-truth trajectory from the dataset index (points + faint line).")
    args = ap.parse_args()

    # From tools/plots -> project root
    root = Path(__file__).resolve().parent.parent.parent
    run_dir = (root / args.run_dir).resolve()
    ckpt_path = run_dir / "model_best.pt"
    norm_path = run_dir / "norm.json"

    # Load fixed split indices
    split_path = run_dir / "split_indices.json"
    with split_path.open("r") as f:
        splits = json.load(f)
    indices = splits[args.subset]

    # 1) Load normalizers
    x_norm, y_norm = load_norm(norm_path)

    # 2) Rebuild dataset (raw, unnormalized)
    data_cfg = load_yaml(root / args.data_config)
    excel_path = (root / data_cfg["excel_path"]).resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[tuple(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K")),
    )
    ds = TrajectoryDataset(excel_path, schema)
    T = ds.T

    # 3) Rebuild model and load checkpoint
    model_cfg = load_yaml(root / args.model_config)
    model = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    # 4) Plot per sample
    for j, idx in enumerate(indices[:args.num_samples]):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Pull a row to get dataset defaults (assumed order: [sx, sy, ex, ey, heading, ox, oy])
        x_row, y_true_row = ds[idx]
        sx0, sy0, ex0, ey0, heading0, ox0, oy0 = [x_row[k].item() for k in range(7)]

        # Use CLI overrides where provided; otherwise dataset defaults
        sx = args.sx if args.sx is not None else sx0
        sy = args.sy if args.sy is not None else sy0
        ex = args.ex if args.ex is not None else ex0
        ey = args.ey if args.ey is not None else ey0
        heading = args.heading if args.heading is not None else heading0
        ox = args.ox if args.ox is not None else ox0
        oy = args.oy if args.oy is not None else oy0
        R  = float(args.radius)

        # Rebuild the exact model input order
        x_raw = torch.tensor([sx, sy, ex, ey, heading, ox, oy], dtype=torch.float32)

        # Model prediction in original units
        x_in = x_norm.transform(x_raw.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()
        pred_2xT = y_pred.view(2, T)

        # === Optional ground truth overlay (points only, black crosses) ===
        if args.plot_gt:
            y_gt = y_true_row
            if args.outputs_are_normalized:
                with torch.no_grad():
                    y_gt = y_norm.inverse(y_gt)
            y_gt = y_gt.detach().cpu().view(2, T)

            # black crosses, slightly larger than prediction markers
            ax.scatter(
                y_gt[0], y_gt[1],
                marker="x", s=48,  # larger size than before
                color="black", linewidths=2,
                label="ground truth"
            )

        # Predicted trajectory (points)
        ax.scatter(pred_2xT[0], pred_2xT[1], marker="o", s=28, label="prediction")

        # Start / end markers
        ax.scatter([sx], [sy], marker="o", s=80, label="start")
        ax.scatter([ex], [ey], marker="*", s=120, label="end")

        # Obstacle as a circle with chosen radius  (fixed: facecolor typo)
        circle = patches.Circle((ox, oy), radius=R, facecolor="red",
                                edgecolor="black", alpha=0.7, label="obstacle")
        ax.add_patch(circle)

        # BPoly
        t = np.linspace(0, 100, 100)
        tknots = np.linspace(0, 100, 10)

        x_col = pred_2xT[0].unsqueeze(1)  # (T,1)
        y_col = pred_2xT[1].unsqueeze(1)  # (T,1)
        x = PiecewiseBernsteinPoly(x_col, tknots, t)
        y = PiecewiseBernsteinPoly(y_col, tknots, t)

        print(x)
        print(y)



        # === Heading arrow relative to baseline (0.1,0.1)->(0.9,0.9) ===
        # Baseline (45°) unit vector
        bx, by = 0.8, 0.8
        bnorm = math.hypot(bx, by)
        ux, uy = bx / bnorm, by / bnorm

        # Rotate baseline by 'heading' radians (CCW)
        c, s = math.cos(heading), math.sin(heading)
        dx = c * ux - s * uy
        dy = s * ux + c * uy

        # Arrow length
        L = 0.12
        ax.quiver(sx, sy, L * dx, L * dy, angles="xy", scale_units="xy",
                  scale=1, width=0.004, label="heading")
        ax.quiver(sx, sy, L * ux, L * uy, angles="xy", scale_units="xy",
                  scale=1, width=0.002, alpha=0.4)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True)
        title_bits = [run_dir.name, f"{args.subset} sample {j}", f"R={R:.3f}"]
        if any(v is not None for v in [args.sx, args.sy, args.ex, args.ey, args.heading, args.ox, args.oy]):
            title_bits.append("overrides")
        ax.set_title(" | ".join(title_bits))
        ax.legend()

        out_png = run_dir / f"collision_heading_sample_{j:03d}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
