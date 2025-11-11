from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os  # Add this to your imports
import sys  # ← ADD THIS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa:
from matplotlib.animation import FuncAnimation, PillowWriter
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config
import time

tools_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(tools_dir / 'extra'))
from BeBOT import PiecewiseBernsteinPoly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Add this function to plot_fnn.py after your imports
@torch.no_grad()
def time_inference(model, x_norm, sample_input, n_warmup=20, n_iterations=100):
    """
    Time model inference with proper GPU synchronization

    Args:
        model: trained model
        x_norm: normalizer object
        sample_input: raw input tensor (13,)
        n_warmup: number of warmup iterations
        n_iterations: number of timed iterations

    Returns:
        dict with timing statistics
    """
    print("\n" + "=" * 60)
    print("TIMING INFERENCE")
    print("=" * 60)

    # Prepare input
    x_in = x_norm.transform(sample_input.unsqueeze(0))

    # Warmup
    print(f"Running {n_warmup} warmup iterations...")
    for _ in range(n_warmup):
        _ = model(x_in)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Timed iterations
    print(f"Running {n_iterations} timed iterations...")
    times = []

    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model(x_in)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    # Print statistics
    print(f"\nSingle sample inference time: {times.mean():.3f}ms ± {times.std():.3f}ms")
    print(f"Min: {times.min():.3f}ms, Max: {times.max():.3f}ms")
    print(f"Median: {np.median(times):.3f}ms")
    print(f"Frequency: {1000 / times.mean():.0f} Hz")
    print("=" * 60 + "\n")

    return {
        'mean_ms': times.mean(),
        'std_ms': times.std(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'median_ms': np.median(times),
        'frequency_hz': 1000 / times.mean()
    }

def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        norm = json.load(f)
    x_state = norm["inputs"]
    y_state = norm["outputs"]
    x_norm = Normalizer(x_state["mode"])
    y_norm = Normalizer(y_state["mode"])

    def _restore(nstate, norm_obj):
        if nstate["stats"] is None:
            norm_obj.fitted = True
            norm_obj.stats = None
        else:
            mean = torch.tensor(nstate["stats"]["mean"], dtype=torch.float32)
            std = torch.tensor(nstate["stats"]["std"], dtype=torch.float32)
            norm_obj.fitted = True
            norm_obj.stats = NormStats(mean=mean, std=std)

    _restore(x_state, x_norm)
    _restore(y_state, y_norm)

    # Extract numpy arrays for global use
    X_mean = np.array(x_state["stats"]["mean"], dtype=np.float32).reshape(1, -1)
    X_std = np.array(x_state["stats"]["std"], dtype=np.float32).reshape(1, -1)
    Y_mean = np.array(y_state["stats"]["mean"], dtype=np.float32).reshape(1, -1)
    Y_std = np.array(y_state["stats"]["std"], dtype=np.float32).reshape(1, -1)

    return x_norm, y_norm, X_mean, X_std, Y_mean, Y_std


def animate_radius_sweep(model, test_input_base, save_path="radius_sweep.gif",
                         elev=45, azim=-70, n_eval=50):
    """
    Create animation sweeping through different obstacle radii

    Args:
        model: trained model
        test_input_base: base test input array (1, 13) flat FNN format
        save_path: where to save the GIF
        elev, azim: camera angles
        n_eval: number of points for Bernstein polynomial evaluation
    """
    # Hardcoded parameters
    radius_min = 0.05
    radius_max = 0.20
    num_frames = 30

    radii = np.linspace(radius_min, radius_max, num_frames)

    # Extract obstacle position for diagnostics
    ox, oy, oz = test_input_base[0, 6], test_input_base[0, 7], test_input_base[0, 8]
    obstacle_center = np.array([ox, oy, oz])

    # DIAGNOSTIC: Check if predictions actually change
    print("\n=== Checking if model responds to radius changes ===")
    print(f"Obstacle center: ({ox:.3f}, {oy:.3f}, {oz:.3f})")
    for i, r in enumerate([radii[0], radii[num_frames // 2], radii[-1]]):
        test_input = test_input_base.copy()
        test_input[0, 9] = r
        test_input_norm = (test_input - X_mean) / X_std
        test_input_tensor = torch.from_numpy(test_input_norm).to(device)

        with torch.no_grad():
            output_norm = model(test_input_tensor)
            output = output_norm.cpu().numpy() * Y_std + Y_mean

        # In animate_radius_sweep, after normalization:
        print(f"\nRadius {r:.3f}:")
        print(f"  Raw input radius index [9]: {test_input[0, 9]:.6f}")
        print(f"  Normalized radius: {test_input_norm[0, 9]:.6f}")
        print(f"  First 3 outputs (raw): {output[0, :3]}")

        # Reshape to see actual control points
        output_reshaped = output[0].reshape(3, 7)
        pred_points = output_reshaped.T  # (7, 3)

        # Calculate distances for first few control points
        cp2_dist = np.linalg.norm(pred_points[0] - obstacle_center)
        cp3_dist = np.linalg.norm(pred_points[1] - obstacle_center)
        cp4_dist = np.linalg.norm(pred_points[2] - obstacle_center)

        print(f"\nRadius {r:.3f}:")
        print(
            f"  CP2: [{pred_points[0, 0]:.4f}, {pred_points[0, 1]:.4f}, {pred_points[0, 2]:.4f}] | dist to obs: {cp2_dist:.4f}")
        print(
            f"  CP3: [{pred_points[1, 0]:.4f}, {pred_points[1, 1]:.4f}, {pred_points[1, 2]:.4f}] | dist to obs: {cp3_dist:.4f}")
        print(
            f"  CP4: [{pred_points[2, 0]:.4f}, {pred_points[2, 1]:.4f}, {pred_points[2, 2]:.4f}] | dist to obs: {cp4_dist:.4f}")
        print(f"  Clearance needed: {r:.3f}")

    print("\n" + "=" * 60)

    # Extract fixed values from base input
    sx, sy, sz = test_input_base[0, 0], test_input_base[0, 1], test_input_base[0, 2]
    ex, ey, ez = test_input_base[0, 3], test_input_base[0, 4], test_input_base[0, 5]
    cpx, cpy, cpz = test_input_base[0, 10], test_input_base[0, 11], test_input_base[0, 12]

    print(f"\nModel in eval mode: {not model.training}")

    # Setup figure once
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    def update(frame):
        ax.clear()
        ax.view_init(elev=elev, azim=azim)

        # Update radius in test input
        current_radius = radii[frame]
        test_input = test_input_base.copy()
        test_input[0, 9] = current_radius

        # Normalize and predict
        test_input_norm = (test_input - X_mean) / X_std
        test_input_tensor = torch.from_numpy(test_input_norm).to(device)

        with torch.no_grad():
            output_norm = model(test_input_tensor)
            output = output_norm.cpu().numpy() * Y_std + Y_mean

        # Reshape: FNN outputs (21,) as [x2...x8, y2...y8, z2...z8]
        output_reshaped = output[0].reshape(3, 7)  # (3, 7) -> [x_coords, y_coords, z_coords]
        pred_points = output_reshaped.T  # (7, 3) -> each row is [x, y, z]

        # Build Bernstein polynomial trajectory
        cp2, cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

        control_points_x = np.array([sx, cp2[0], cp3[0], cp4[0], cp5[0],
                                     cp5[0], cp6[0], cp7[0], cp8[0], ex])
        control_points_y = np.array([sy, cp2[1], cp3[1], cp4[1], cp5[1],
                                     cp5[1], cp6[1], cp7[1], cp8[1], ey])
        control_points_z = np.array([sz, cp2[2], cp3[2], cp4[2], cp5[2],
                                     cp5[2], cp6[2], cp7[2], cp8[2], ez])

        tknots = np.array([0, 0.5, 1.0])
        t_eval = np.linspace(0, 1, n_eval)

        traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
        traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
        traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

        # Check collision
        trajectory = np.column_stack([traj_x, traj_y, traj_z])
        distances = np.linalg.norm(trajectory - obstacle_center, axis=1)
        min_dist = distances.min()
        collides = np.any(distances < current_radius)

        # Plot trajectory
        color = 'red' if collides else 'green'
        ax.plot(traj_x, traj_y, traj_z, color=color, linewidth=2,
                label='trajectory', alpha=0.7)

        # Start/end points
        ax.scatter([sx], [sy], [sz], s=80, marker="o", label="start", depthshade=False)
        ax.scatter([ex], [ey], [ez], s=120, marker="*", label="end", depthshade=False)
        ax.scatter([ox], [oy], [oz], s=120, color='red', label='obstacle', depthshade=False)

        # Control point (initial velocity)
        ax.scatter([cpx], [cpy], [cpz], s=80, marker='o', label='heading', depthshade=False)

        # Predicted control points
        for i, cp in enumerate(pred_points):
            ax.scatter([cp[0]], [cp[1]], [cp[2]], s=28, marker='o',
                       alpha=0.6, depthshade=False)

        # Obstacle sphere
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        xs = ox + current_radius * np.cos(u) * np.sin(v)
        ys = oy + current_radius * np.sin(u) * np.sin(v)
        zs = oz + current_radius * np.cos(v)
        ax.plot_surface(xs, ys, zs, color='red', alpha=0.3, linewidth=0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f"Radius: {current_radius:.3f} | Min Dist: {min_dist:.3f} | {'COLLISION' if collides else 'SAFE'}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=True)

    print(f"\nStarting to save animation to {save_path}...")
    anim.save(save_path, writer=PillowWriter(fps=10))
    print(f"[SAVED] {save_path}")
    print(f"File exists: {os.path.exists(save_path)}")

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
    ap.add_argument("--time-eval", action="store_true", help="Run timing benchmark")
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
    global X_mean, X_std, Y_mean, Y_std
    x_norm, y_norm, X_mean, X_std, Y_mean, Y_std = load_norm(run_dir / "norm.json")


    # Plot training curves if requested
    if args.plot_curves:
        plot_training_curves(run_dir)
        return

    # Load configs and model
    with (run_dir / "configs_used.json").open() as f:
        configs = json.load(f)
    data_cfg = configs["data_config"];
    model_cfg = configs["model_config"]

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

    # Timing inference
    if args.time_eval:
        sample_input, _ = ds[0]
        timing_stats = time_inference(model, x_norm, sample_input)

    print("\n=== Testing on actual training sample ===")
    # Get a random training sample
    train_idx = 100  # or any index
    x_sample, y_true = ds[train_idx]
    print(f"Sample {train_idx} inputs: {x_sample[:13]}")
    print(f"Sample radius: {x_sample[12].item():.3f}")

    x_in = x_norm.transform(x_sample.unsqueeze(0))
    with torch.no_grad():
        y_pred_norm = model(x_in).squeeze(0)
    y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

    print(f"Ground truth first 3 outputs: {y_true[:3]}")
    print(f"Predicted first 3 outputs: {y_pred[:3]}")
    print(f"Error: {torch.abs(y_pred - y_true).mean().item():.6f}")



    # Manual input mode
    if args.manual:
        manual_input = {
            'x0': args.x0, 'y0': args.y0, 'z0': args.z0,
            'xf': args.xf, 'yf': args.yf, 'zf': args.zf,
            'ox': args.ox, 'oy': args.oy, 'oz': args.oz,
            'vx': args.vx, 'vy': args.vy, 'vz': args.vz,
            'r': args.R
        }

        test_input = np.array([[
            0.0, 0.0, 0.0,  # x0, y0, z0 (start)
            1.0, 1.0, 1.0,  # xf, yf, zf (goal)
            0.5, 0.5, 0.4,  # ox, oy, oz (obstacle)
            0.1,  # r (radius - will be varied in animation)
            0.0, 0.0, 0.0  # vx, vy, vz (initial velocity)

        ]], dtype=np.float32)

        # Add this debugging right after test_input definition:
        print("\n=== Input Format Verification ===")
        print(f"test_input shape: {test_input.shape}")  # Should be (1, 13)
        print(f"test_input[0]: {test_input[0]}")
        print(f"Expected order: [x0, y0, z0, xf, yf, zf, ox, oy, oz, vx, vy, vz, r]")

        # Add this debugging in your main() before the animation call:
        print("\n=== Normalization Stats Check ===")
        print(f"X_mean shape: {X_mean.shape}")
        print(f"X_std shape: {X_std.shape}")
        print(f"Radius mean: {X_mean[0, 9]:.6f}")
        print(f"Radius std: {X_std[0, 9]:.6f}")
        print("\nTest input (before norm):")
        print(f"Radius value: {test_input[0, 9]:.3f}")
        test_input_norm = (test_input - X_mean) / X_std
        print(f"Radius value (after norm): {test_input_norm[0, 9]:.3f}")
        print("=" * 60)

        if args.animate_radius:
            animate_radius_sweep(model, test_input, save_path="tools/extra/radius_sweep_fnn.gif", elev=45, azim=-70, n_eval=50)
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