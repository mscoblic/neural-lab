from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa:
from matplotlib.animation import FuncAnimation, PillowWriter


from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config


def animate_radius_sweep(model, x_norm, y_norm, manual_input: dict, K: int, T: int,
                         run_dir: Path, args):
    """
    Create animation sweeping through different obstacle radii
    """
    # Hardcoded parameters
    radius_min = 0.01
    radius_max = 0.20
    num_frames = 30

    radii = np.linspace(radius_min, radius_max, num_frames)

    # Setup figure once
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    sx, sy, sz = manual_input['x0'], manual_input['y0'], manual_input['z0']
    ex, ey, ez = manual_input['xf'], manual_input['yf'], manual_input['zf']
    ox, oy, oz = manual_input['ox'], manual_input['oy'], manual_input['oz']

    def update(frame):
        ax.clear()
        ax.view_init(elev=args.elev, azim=args.azim)

        # Update radius
        current_radius = radii[frame]
        manual_input['r'] = current_radius

        # Predict trajectory with new radius
        pred_3xT = predict_manual_input(model, x_norm, y_norm, manual_input, K, T)

        # Check collision
        collides, min_dist, _ = check_collision(
            pred_3xT, (ox, oy, oz), radius=current_radius
        )

        # Plot trajectory
        color = 'red' if collides else 'green'
        ax.scatter(pred_3xT[0].numpy(), pred_3xT[1].numpy(), pred_3xT[2].numpy(),
                   s=28, marker="o", c=color, label="trajectory", depthshade=False)

        # Start/end points
        ax.scatter([sx], [sy], [sz], s=80, marker="o", label="start", depthshade=False)
        ax.scatter([ex], [ey], [ez], s=120, marker="*", label="end", depthshade=False)

        # Obstacle sphere
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        xs = ox + current_radius * np.cos(u) * np.sin(v)
        ys = oy + current_radius * np.sin(u) * np.sin(v)
        zs = oz + current_radius * np.cos(v)
        ax.plot_surface(xs, ys, zs, alpha=0.3, linewidth=0)

        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.set_zlabel("z")
        ax.set_xlim(0, 1);
        ax.set_ylim(0, 1);
        ax.set_zlim(0, 1)
        ax.set_title(f"Radius: {current_radius:.3f} | {'COLLISION' if collides else 'SAFE'}")
        ax.legend(loc="best")
        ax.grid(True)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=True)

    # Save as GIF
    gif_path = run_dir / "radius_sweep.gif"
    anim.save(gif_path, writer=PillowWriter(fps=10))
    print(f"[SAVED] {gif_path}")

    plt.close(fig)

def check_collision(pred_3xT, obstacle_pos, radius=0.1):
    """
    Check if predicted trajectory collides with obstacle

    Returns:
        collides (bool): True if collision detected
        min_distance (float): Minimum distance to obstacle
        collision_points (list): Indices of colliding points
    """
    ox, oy, oz = obstacle_pos
    obstacle_center = np.array([ox, oy, oz])

    # Get trajectory points as (T, 3)
    traj_points = pred_3xT.T.numpy()  # (T, 3)

    # Compute distances to obstacle
    distances = np.linalg.norm(traj_points - obstacle_center, axis=1)

    # Check collisions
    collision_mask = distances < radius
    collision_points = np.where(collision_mask)[0].tolist()
    min_distance = np.min(distances)
    collides = len(collision_points) > 0

    return collides, min_distance, collision_points

def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        norm = json.load(f)
    x_state = norm["inputs"];
    y_state = norm["outputs"]
    x_norm = Normalizer(x_state["mode"]);
    y_norm = Normalizer(y_state["mode"])

    def _restore(nstate, norm_obj):
        if nstate["stats"] is None:
            norm_obj.fitted = True;
            norm_obj.stats = None
        else:
            mean = torch.tensor(nstate["stats"]["mean"], dtype=torch.float32)
            std = torch.tensor(nstate["stats"]["std"], dtype=torch.float32)
            norm_obj.fitted = True;
            norm_obj.stats = NormStats(mean=mean, std=std)

    _restore(x_state, x_norm);
    _restore(y_state, y_norm)
    return x_norm, y_norm

def plot_training_curves(run_dir: Path):
    """Plot MSE training curves from metrics.csv"""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"[WARN] No metrics.csv found at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot train and val MSE
    ax.plot(df["epoch"], df["train_mse"], label="Train MSE", linewidth=2)
    ax.plot(df["epoch"], df["val_mse"], label="Val MSE", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"Training Curves - {run_dir.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = run_dir / "training_curves.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def predict_manual_input(model, x_norm, y_norm, manual_input: dict, K: int, T: int):
    """
    Predict trajectory from manual input

    manual_input dict should contain:
        'x0', 'y0', 'z0': start position
        'xf', 'yf', 'zf': goal position
        'ox', 'oy', 'oz': obstacle position
        'vx', 'vy', 'vz': initial velocity
    """
    # Build input tensor in same order as training data
    x_raw = torch.tensor([
        manual_input['x0'], manual_input['y0'], manual_input['z0'],
        manual_input['xf'], manual_input['yf'], manual_input['zf'],
        manual_input['ox'], manual_input['oy'], manual_input['oz'],
        manual_input['vx'], manual_input['vy'], manual_input['vz'],
        manual_input['r']
    ], dtype=torch.float32)

    # Normalize and predict
    x_in = x_norm.transform(x_raw.unsqueeze(0))
    with torch.no_grad():
        y_pred_norm = model(x_in).squeeze(0)
    y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

    pred_3xT = y_pred.view(K, T)
    return pred_3xT

def plot_manual_trajectory(pred_3xT, manual_input: dict, run_dir: Path, args):
    """Plot trajectory from manual input"""
    sx, sy, sz = manual_input['x0'], manual_input['y0'], manual_input['z0']
    ex, ey, ez = manual_input['xf'], manual_input['yf'], manual_input['zf']
    ox, oy, oz = manual_input['ox'], manual_input['oy'], manual_input['oz']

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect([1, 1, 1])

    # predicted points
    ax.scatter(pred_3xT[0].numpy(), pred_3xT[1].numpy(), pred_3xT[2].numpy(),
               s=28, marker="o", label="prediction", depthshade=False)

    # start / end
    ax.scatter([sx], [sy], [sz], s=80, marker="o", label="start", depthshade=False)
    ax.scatter([ex], [ey], [ez], s=120, marker="*", label="end", depthshade=False)

    # obstacle
    if args.show_obs:
        R = float(args.R)
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        xs = ox + R * np.cos(u) * np.sin(v)
        ys = oy + R * np.sin(u) * np.sin(v)
        zs = oz + R * np.cos(v)
        ax.plot_surface(xs, ys, zs, alpha=0.25, linewidth=0)

    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")
    ax.set_title(f"Manual Input Prediction - {run_dir.name}")
    ax.legend(loc="best");
    ax.grid(True)

    out_png = run_dir / "manual_input_prediction.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--show-gt", action="store_true")
    ap.add_argument("--show-obs", action="store_true")
    ap.add_argument("--elev", type=float, default=45)
    ap.add_argument("--azim", type=float, default=-70)
    ap.add_argument("--animate-radius", action="store_true", help="Create radius sweep animation")
    ap.add_argument("--plot-curves", action="store_true", help="Plot training curves")
    ap.add_argument("--manual", action="store_true", help="Use manual input")
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--xf", type=float, default=1.0)
    ap.add_argument("--yf", type=float, default=1.0)
    ap.add_argument("--zf", type=float, default=1.0)
    ap.add_argument("--ox", type=float, default=0.5)
    ap.add_argument("--oy", type=float, default=0.5)
    ap.add_argument("--oz", type=float, default=0.4)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vz", type=float, default=0.0)
    ap.add_argument("--R", type=float, default=0.10, help="obstacle radius")

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_dir = (root / args.run_dir).resolve()

    # Plot training curves if requested
    if args.plot_curves:
        plot_training_curves(run_dir)
        return

    # Load configs and model
    with (run_dir / "configs_used.json").open() as f:
        configs = json.load(f)
    data_cfg = configs["data_config"];
    model_cfg = configs["model_config"]

    x_norm, y_norm = load_norm(run_dir / "norm.json")

    # Load schema for K and T
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[list(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 3)),
    )

    # Build model
    excel_path = (root / data_cfg["excel_path"]).resolve()
    ds = TrajectoryDataset(excel_path, schema)
    T, K = ds.T, schema.K

    ckpt_path = run_dir / args.ckpt
    model = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"]);
    model.eval()

    # Manual input mode
    if args.manual:
        manual_input = {
            'x0': args.x0, 'y0': args.y0, 'z0': args.z0,
            'xf': args.xf, 'yf': args.yf, 'zf': args.zf,
            'ox': args.ox, 'oy': args.oy, 'oz': args.oz,
            'vx': args.vx, 'vy': args.vy, 'vz': args.vz,
            'r': args.R
        }

        if args.animate_radius:
            animate_radius_sweep(model, x_norm, y_norm, manual_input, K, T, run_dir, args)
            return

        pred_3xT = predict_manual_input(model, x_norm, y_norm, manual_input, K, T)

        # Check collision
        collides, min_dist, coll_points = check_collision(
            pred_3xT,
            (manual_input['ox'], manual_input['oy'], manual_input['oz']),
            radius=args.R
        )

        # Print results
        print("\n" + "=" * 60)
        print("COLLISION DETECTION RESULTS")
        print("=" * 60)
        print(f"Obstacle: ({manual_input['ox']:.3f}, {manual_input['oy']:.3f}, {manual_input['oz']:.3f})")
        print(f"Obstacle radius: {args.R:.3f}")
        print(f"Minimum distance to obstacle: {min_dist:.6f}")
        print(f"Collision detected: {'YES ⚠️' if collides else 'NO ✓'}")
        if collides:
            print(f"Collision at {len(coll_points)}/{T} predicted points")
            print(f"Collision point indices: {coll_points}")
        print("=" * 60 + "\n")

        plot_manual_trajectory(pred_3xT, manual_input, run_dir, args)
        return

    # Regular mode - plot test samples
    with (run_dir / "split_indices.json").open() as f:
        splits = json.load(f)
    indices = splits[args.subset]

    for j, idx in enumerate(indices[:args.num_samples]):
        x_row, y_true_row = ds[idx]

        sx, sy, sz = x_row[0].item(), x_row[1].item(), x_row[2].item()
        ex, ey, ez = x_row[3].item(), x_row[4].item(), x_row[5].item()
        ox, oy, oz = x_row[6].item(), x_row[7].item(), x_row[8].item()
        r = x_row[9].item()

        x_in = x_norm.transform(x_row.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

        pred_3xT = y_pred.view(K, T)
        gt_3xT = y_true_row.view(K, T)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_box_aspect([1, 1, 1])

        ax.scatter(pred_3xT[0].numpy(), pred_3xT[1].numpy(), pred_3xT[2].numpy(),
                   s=28, marker="o", label="prediction", depthshade=False)

        if args.show_gt:
            ax.scatter(gt_3xT[0].numpy(), gt_3xT[1].numpy(), gt_3xT[2].numpy(),
                       s=36, marker="x", label="ground truth", depthshade=False)

        ax.scatter([sx], [sy], [sz], s=80, marker="o", label="start", depthshade=False)
        ax.scatter([ex], [ey], [ez], s=120, marker="*", label="end", depthshade=False)

        if args.show_obs:
            u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
            xs = ox + r * np.cos(u) * np.sin(v)
            ys = oy + r * np.sin(u) * np.sin(v)
            zs = oz + r * np.cos(v)
            ax.plot_surface(xs, ys, zs, alpha=0.25, linewidth=0)

        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.set_zlabel("z")
        ax.set_title(f"{run_dir.name} | {args.subset} sample {j}")
        ax.legend(loc="best");
        ax.grid(True)

        out_png = run_dir / f"traj3d_scatter_{j:03d}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")

if __name__ == "__main__":
    main()