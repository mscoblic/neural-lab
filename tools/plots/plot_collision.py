from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config import load_yaml
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config


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


# ----------------------------
# Collision utilities
# ----------------------------
def _point_hits_circle(x: float, y: float, ox: float, oy: float, R: float) -> bool:
    dx, dy = x - ox, y - oy
    return dx * dx + dy * dy <= R * R

def _segment_circle_intersect(p1, p2, c, R) -> bool:
    (x1, y1), (x2, y2) = p1, p2
    (cx, cy) = c
    vx, vy = x2 - x1, y2 - y1
    wx, wy = cx - x1, cy - y1

    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0.0:
        return _point_hits_circle(x1, y1, cx, cy, R)

    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    qx, qy = x1 + t * vx, y1 + t * vy
    dx, dy = qx - cx, qy - cy
    return dx * dx + dy * dy <= R * R

def count_collisions(traj_2xT: torch.Tensor, ox: float, oy: float, R: float):
    """
    traj_2xT: tensor shaped (2, T) in original units.
    Returns (num_point_hits, num_segment_crosses, collides_bool).
    """
    xs = traj_2xT[0].tolist()
    ys = traj_2xT[1].tolist()
    T = len(xs)

    point_hits = sum(1 for t in range(T) if _point_hits_circle(xs[t], ys[t], ox, oy, R))
    seg_hits = 0
    for t in range(T - 1):
        if _segment_circle_intersect((xs[t], ys[t]), (xs[t+1], ys[t+1]), (ox, oy), R):
            seg_hits += 1

    collides = (point_hits > 0) or (seg_hits > 0)
    return point_hits, seg_hits, collides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a specific run folder under runs/")
    ap.add_argument("--data-config",  default="configs/collision/data_collision.yaml")
    ap.add_argument("--model-config", default="configs/collision/model_collision.yaml")
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=1)

    # Optional overrides and obstacle radius (no heading in this task)
    ap.add_argument("--sx", type=float)
    ap.add_argument("--sy", type=float)
    ap.add_argument("--ex", type=float)
    ap.add_argument("--ey", type=float)
    ap.add_argument("--ox", type=float)
    ap.add_argument("--oy", type=float)
    ap.add_argument("--radius", type=float, default=0.05, help="Obstacle radius")

    # Normalization + plotting options
    ap.add_argument("--outputs-are-normalized", action="store_true",
                    help="If dataset stores normalized outputs, inverse-normalize GT before plotting.")
    ap.add_argument("--plot-gt", action="store_true",
                    help="Overlay ground-truth trajectory (points + faint line).")
    ap.add_argument("--highlight-collisions", action="store_true",
                    help="Visually mark colliding prediction/GT points.")
    args = ap.parse_args()

    # From tools/plots -> project root (go up three levels like the heading tool)
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

    total_pred_collides = total_gt_collides = 0

    # 4) One figure per sample: predicted trajectory + start/end/obstacle
    for j, idx in enumerate(indices[:args.num_samples]):
        fig, ax = plt.subplots(figsize=(6, 6))

        x_row, y_true_row = ds[idx]

        # unpack inputs: [sx, sy, ex, ey, ox, oy]
        sx0, sy0, ex0, ey0, ox0, oy0 = [x_row[k].item() for k in range(6)]

        # apply CLI overrides if provided
        sx = args.sx if args.sx is not None else sx0
        sy = args.sy if args.sy is not None else sy0
        ex = args.ex if args.ex is not None else ex0
        ey = args.ey if args.ey is not None else ey0
        ox = args.ox if args.ox is not None else ox0
        oy = args.oy if args.oy is not None else oy0
        R  = float(args.radius)

        # rebuild model input order
        x_raw = torch.tensor([sx, sy, ex, ey, ox, oy], dtype=torch.float32)

        # model prediction in original units
        x_in = x_norm.transform(x_raw.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()
        pred_2xT = y_pred.view(2, T)

        # ---- optional GT overlay ----
        if args.plot_gt:
            y_gt = y_true_row
            if args.outputs_are_normalized:
                with torch.no_grad():
                    y_gt = y_norm.inverse(y_gt)
            y_gt = y_gt.detach().cpu().view(2, T)
            ax.scatter(y_gt[0], y_gt[1], marker="x", s=36, label="ground truth")
            ax.plot(y_gt[0], y_gt[1], linestyle="--", linewidth=0.8, alpha=0.5)

        # predicted trajectory
        ax.scatter(pred_2xT[0], pred_2xT[1], marker="o", s=28, label="prediction")

        # start / end markers
        ax.scatter([sx], [sy], marker="o", s=80, label="start")
        ax.scatter([ex], [ey], marker="*", s=120, label="end")

        # obstacle as a circle with chosen radius
        circle = patches.Circle((ox, oy), radius=R, facecolor="red", edgecolor="black", alpha=0.7, label="obstacle")
        ax.add_patch(circle)

        # ---- collision stats (prediction) ----
        p_hits, p_cross, p_collides = count_collisions(pred_2xT, ox, oy, R)
        total_pred_collides += int(p_collides)

        if args.highlight_collisions:
            xs, ys = pred_2xT[0].numpy(), pred_2xT[1].numpy()
            hit_mask = ((xs - ox) ** 2 + (ys - oy) ** 2) <= (R ** 2)
            ax.scatter(xs[hit_mask], ys[hit_mask], s=60, marker="o",
                       facecolors="none", edgecolors="black", linewidths=1.5)

        # ---- collision stats (GT) if present ----
        if args.plot_gt:
            g_hits, g_cross, g_collides = count_collisions(y_gt, ox, oy, R)
            total_gt_collides += int(g_collides)
            print(f"[SAMPLE {j}] pred: {p_hits} point hits, {p_cross} segment crosses → "
                  f"{'COLLIDES' if p_collides else 'no collision'}")
            print(f"            gt  : {g_hits} point hits, {g_cross} segment crosses → "
                  f"{'COLLIDES' if g_collides else 'no collision'}")
        else:
            print(f"[SAMPLE {j}] pred: {p_hits} point hits, {p_cross} segment crosses → "
                  f"{'COLLIDES' if p_collides else 'no collision'}")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True)
        title_bits = [run_dir.name, f"{args.subset} sample {j}", f"R={R:.3f}"]
        if any(v is not None for v in [args.sx, args.sy, args.ex, args.ey, args.ox, args.oy]):
            title_bits.append("overrides")
        ax.set_title(" | ".join(title_bits))
        ax.legend()

        out_png = run_dir / f"collision_sample_{j:03d}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")

    # Summary
    if args.plot_gt:
        print(f"[SUMMARY] {total_pred_collides}/{args.num_samples} pred samples collided; "
              f"{total_gt_collides}/{args.num_samples} GT samples collided.")
    else:
        print(f"[SUMMARY] {total_pred_collides}/{args.num_samples} pred samples collided.")


if __name__ == "__main__":
    main()
