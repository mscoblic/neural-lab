from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa:
from matplotlib.animation import FuncAnimation, PillowWriter
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config
import time
from scipy import stats
import plotly.graph_objects as go

tools_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(tools_dir / 'extra'))
from BeBOT import PiecewiseBernsteinPoly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def time_inference_comparison(model, x_norm, y_norm, dataset, n_samples=100, n_warmup=10):
    """
    Compare model-only vs full pipeline timing for FNN
    Uses INDEPENDENT samples for unbiased measurement
    """
    import time

    device = next(model.parameters()).device
    model.eval()

    # Warmup
    x, _ = dataset[0]
    x_in = x_norm.transform(x.unsqueeze(0)).to(device)
    for _ in range(n_warmup):
        y_out = model(x_in)
        _ = y_norm.inverse(y_out)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Sample indices - need 2x for independent measurements
    indices_pipeline = np.random.choice(len(dataset), size=n_samples, replace=False)
    indices_model = np.random.choice(len(dataset), size=n_samples, replace=False)

    # === MEASURE FULL PIPELINE ===
    pipeline_times = []
    for idx in indices_pipeline:
        x_raw, _ = dataset[idx]

        t0 = time.perf_counter()
        x_in = x_norm.transform(x_raw.unsqueeze(0)).to(device)
        y_norm_out = model(x_in)
        y_out = y_norm.inverse(y_norm_out)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pipeline_times.append((t1 - t0) * 1000)

    # === MEASURE MODEL ONLY (separate loop, independent samples) ===
    model_times = []
    for idx in indices_model:
        x_raw, _ = dataset[idx]

        # Pre-normalize (NOT timed)
        x_in = x_norm.transform(x_raw.unsqueeze(0)).to(device)

        # Time only forward pass
        t0 = time.perf_counter()
        _ = model(x_in)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        model_times.append((t1 - t0) * 1000)

    model_times = np.array(model_times)
    pipeline_times = np.array(pipeline_times)
    overhead = pipeline_times.mean() - model_times.mean()

    print(f"\n{'=' * 60}")
    print("FNN INFERENCE TIMING")
    print(f"{'=' * 60}")
    print(f"\nModel Only (forward pass):")
    print(f"  {model_times.mean():.3f} ± {model_times.std():.3f} ms")
    print(f"  Min: {model_times.min():.3f} ms, Max: {model_times.max():.3f} ms")
    print(f"  → {1000.0 / model_times.mean():.0f} Hz")

    print(f"\nFull Pipeline (norm + forward + denorm):")
    print(f"  {pipeline_times.mean():.3f} ± {pipeline_times.std():.3f} ms")
    print(f"  Min: {pipeline_times.min():.3f} ms, Max: {pipeline_times.max():.3f} ms")
    print(f"  → {1000.0 / pipeline_times.mean():.0f} Hz")

    print(f"{'=' * 60}\n")

    return {
        'model_mean': model_times.mean(),
        'pipeline_mean': pipeline_times.mean(),
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

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot val and train MSE (val first)
    xs = np.arange(1, len(df) + 1)
    ax.plot(xs, df["val_mse"], linestyle='-', linewidth=2, label='Val MSE')
    ax.plot(xs, df["train_mse"], linestyle='-', linewidth=2, label='Train MSE')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("MSE over epochs")
    ax.legend()
    ax.grid(True)

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
        'obstacles': list of (ox, oy, oz) tuples
        'r': obstacle radius
        'vx', 'vy', 'vz': initial velocity
    """
    # Build input tensor: [start(3), goal(3), obstacles(numObs*3), radius(1), velocity(3)]
    x_raw_list = [
        manual_input['x0'], manual_input['y0'], manual_input['z0'],
        manual_input['xf'], manual_input['yf'], manual_input['zf']
    ]

    # Add all obstacles
    for obs in manual_input['obstacles']:
        x_raw_list.extend(obs)

    # Add radius
    x_raw_list.append(manual_input['r'])

    # Add velocity
    x_raw_list.extend([manual_input['vx'], manual_input['vy'], manual_input['vz']])

    x_raw = torch.tensor(x_raw_list, dtype=torch.float32)

    # Normalize and predict
    x_in = x_norm.transform(x_raw.unsqueeze(0))
    with torch.no_grad():
        y_pred_norm = model(x_in).squeeze(0)
    y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

    pred_3xT = y_pred.view(K, T)
    return pred_3xT


def plot_manual_trajectory(pred_3xT, manual_input: dict, run_dir: Path, args):
    """Plot trajectory from manual input using BeBOT - both matplotlib and Plotly"""
    sx, sy, sz = manual_input['x0'], manual_input['y0'], manual_input['z0']
    ex, ey, ez = manual_input['xf'], manual_input['yf'], manual_input['zf']
    obstacles = manual_input['obstacles']
    r = manual_input['r']

    # Build Bernstein trajectory from predicted control points
    pred_points = pred_3xT.T.numpy()  # (T, 3)

    # Assuming T=7 predicted points (cp2 through cp8)
    cp2, cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

    control_points_x = np.array([sx, cp2[0], cp3[0], cp4[0], cp5[0],
                                 cp5[0], cp6[0], cp7[0], cp8[0], ex])
    control_points_y = np.array([sy, cp2[1], cp3[1], cp4[1], cp5[1],
                                 cp5[1], cp6[1], cp7[1], cp8[1], ey])
    control_points_z = np.array([sz, cp2[2], cp3[2], cp4[2], cp5[2],
                                 cp5[2], cp6[2], cp7[2], cp8[2], ez])

    tknots = np.array([0, 0.5, 1.0])
    t_eval = np.linspace(0, 1, 100)

    traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
    traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
    traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

    # Create Plotly figure
    fig = go.Figure()

    # Trajectory
    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Trajectory'
    ))

    # Control points
    fig.add_trace(go.Scatter3d(
        x=pred_points[:, 0], y=pred_points[:, 1], z=pred_points[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='green'),
        text=[f'CP{i + 2}' for i in range(len(pred_points))],
        textposition='top center',
        name='Control Points'
    ))

    # Start/End
    fig.add_trace(go.Scatter3d(
        x=[sx, ex], y=[sy, ey], z=[sz, ez],
        mode='markers',
        marker=dict(size=10, color=['green', 'orange'], symbol='diamond'),
        name='Start/End'
    ))

    # Obstacle spheres
    if args.show_obs:
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        for i, (ox, oy, oz) in enumerate(obstacles):
            xs = ox + r * np.cos(u) * np.sin(v)
            ys = oy + r * np.sin(u) * np.sin(v)
            zs = oz + r * np.cos(v)

            fig.add_trace(go.Surface(
                x=xs, y=ys, z=zs,
                opacity=0.3,
                colorscale='Reds',
                showscale=False,
                name=f'Obstacle {i + 1}'
            ))

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[0, 1], title='X'),
            yaxis=dict(range=[0, 1], title='Y'),
            zaxis=dict(range=[0, 1], title='Z'),
            camera=dict(
                eye=dict(x=1.6, y=-1.3, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            )
        ),
        title=f'Manual Input Prediction - {run_dir.name}'
    )

    out_html = run_dir / "manual_input_prediction.html"
    out_png = run_dir / "manual_input_prediction.png"
    fig.write_html(out_html)
    fig.write_image(out_png, width=800, height=800)
    print(f"[SAVED] {out_html} (interactive)")
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
    ap.add_argument("--plot-curves", action="store_true", help="Plot training curves")
    ap.add_argument("--manual", action="store_true", help="Use manual input")
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--xf", type=float, default=1.0)
    ap.add_argument("--yf", type=float, default=1.0)
    ap.add_argument("--zf", type=float, default=1.0)
    ap.add_argument("--R", type=float, default=0.05, help="obstacle radius")
    # Support multiple obstacles via repeated --ox, --oy, --oz arguments
    ap.add_argument("--ox", type=float, action='append', help="obstacle x positions (can be repeated)")
    ap.add_argument("--oy", type=float, action='append', help="obstacle y positions (can be repeated)")
    ap.add_argument("--oz", type=float, action='append', help="obstacle z positions (can be repeated)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vz", type=float, default=0.0)

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
    data_cfg = configs["data_config"]
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
    model.load_state_dict(state["state_dict"])
    model.eval()

    model = model.to(device)

    # Timing inference
    if args.time_eval:
        with (run_dir / "split_indices.json").open() as f:
            splits = json.load(f)
        test_indices = splits["test"]

        class SubsetDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                return self.dataset[self.indices[i]]

        test_subset = SubsetDataset(ds, test_indices)
        timing_stats = time_inference_comparison(model, x_norm, y_norm, test_subset, n_samples=100)

        fnn_results = {
            'model_only_ms': timing_stats['model_mean'],
            'full_pipeline_ms': timing_stats['pipeline_mean'],
            'model_only_hz': 1000.0 / timing_stats['model_mean'],
            'full_pipeline_hz': 1000.0 / timing_stats['pipeline_mean']
        }

        with open('fnn_timing_results.json', 'w') as f:
            json.dump(fnn_results, f, indent=2)

        print("[SAVED] fnn_timing_results.json")
        return

    # Manual input mode
    if args.manual:
        # Parse obstacles
        if args.ox is None or args.oy is None or args.oz is None:
            print("Error: Must provide at least one obstacle with --ox, --oy, --oz")
            return

        if not (len(args.ox) == len(args.oy) == len(args.oz)):
            print("Error: Number of --ox, --oy, --oz arguments must match")
            return

        obstacles = list(zip(args.ox, args.oy, args.oz))

        manual_input = {
            'x0': args.x0, 'y0': args.y0, 'z0': args.z0,
            'xf': args.xf, 'yf': args.yf, 'zf': args.zf,
            'obstacles': obstacles,
            'r': args.R,
            'vx': args.vx, 'vy': args.vy, 'vz': args.vz
        }

        pred_3xT = predict_manual_input(model, x_norm, y_norm, manual_input, K, T)

        # Check collision with first obstacle
        collides, min_dist, coll_points = check_collision(
            pred_3xT,
            obstacles[0],
            radius=args.R
        )

        print("\n" + "=" * 60)
        print("COLLISION DETECTION RESULTS")
        print("=" * 60)
        print(f"Number of obstacles: {len(obstacles)}")
        for i, obs in enumerate(obstacles):
            print(f"Obstacle {i + 1}: ({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f})")
        print(f"Obstacle radius: {args.R:.3f}")
        print(f"Minimum distance to first obstacle: {min_dist:.6f}")
        print(f"Collision detected: {'YES ⚠️' if collides else 'NO ✓'}")
        if collides:
            print(f"Collision at {len(coll_points)}/{T} predicted points")
        print("=" * 60 + "\n")

        plot_manual_trajectory(pred_3xT, manual_input, run_dir, args)
        return

    # Regular mode - plot test samples with BeBOT
    with (run_dir / "split_indices.json").open() as f:
        splits = json.load(f)
    indices = splits[args.subset]

    for j, idx in enumerate(indices[:args.num_samples]):
        x_row, y_true_row = ds[idx]

        sx, sy, sz = x_row[0].item(), x_row[1].item(), x_row[2].item()
        ex, ey, ez = x_row[3].item(), x_row[4].item(), x_row[5].item()

        ox, oy, oz = x_row[6].item(), x_row[7].item(), x_row[8].item()
        r = 0.1

        print(f"\n[DEBUG] Sample {j}: Obstacle at ({ox:.3f}, {oy:.3f}, {oz:.3f}), radius={r:.3f}")

        x_in = x_norm.transform(x_row.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

        pred_3xT = y_pred.view(K, T)
        gt_3xT = y_true_row.view(K, T)

        # Build Bernstein trajectories
        pred_points = pred_3xT.T.numpy()
        gt_points = gt_3xT.T.numpy()

        # Unpack 17 predicted control points (cp2 through cp18)
        cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10, cp11, cp12, cp13, cp14, cp15, cp16, cp17, cp18 = pred_points

        # Build control points: 10 per segment with cp10 duplicated
        # Segment 1: [sx, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10]
        # Segment 2: [cp10, cp11, cp12, cp13, cp14, cp15, cp16, cp17, cp18, ex]
        control_points_x = np.array([sx, cp2[0], cp3[0], cp4[0], cp5[0], cp6[0], cp7[0], cp8[0], cp9[0], cp10[0],
                                     cp10[0], cp11[0], cp12[0], cp13[0], cp14[0], cp15[0], cp16[0], cp17[0], cp18[0],
                                     ex])
        control_points_y = np.array([sy, cp2[1], cp3[1], cp4[1], cp5[1], cp6[1], cp7[1], cp8[1], cp9[1], cp10[1],
                                     cp10[1], cp11[1], cp12[1], cp13[1], cp14[1], cp15[1], cp16[1], cp17[1], cp18[1],
                                     ey])
        control_points_z = np.array([sz, cp2[2], cp3[2], cp4[2], cp5[2], cp6[2], cp7[2], cp8[2], cp9[2], cp10[2],
                                     cp10[2], cp11[2], cp12[2], cp13[2], cp14[2], cp15[2], cp16[2], cp17[2], cp18[2],
                                     ez])

        tknots = np.array([0, 0.5, 1.0])
        t_eval = np.linspace(0, 1, 100)

        pred_traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
        pred_traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
        pred_traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

        # Build ground truth trajectory if requested
        if args.show_gt:
            cp2_gt, cp3_gt, cp4_gt, cp5_gt, cp6_gt, cp7_gt, cp8_gt = gt_points

            control_points_x_gt = np.array([sx, cp2_gt[0], cp3_gt[0], cp4_gt[0], cp5_gt[0],
                                            cp5_gt[0], cp6_gt[0], cp7_gt[0], cp8_gt[0], ex])
            control_points_y_gt = np.array([sy, cp2_gt[1], cp3_gt[1], cp4_gt[1], cp5_gt[1],
                                            cp5_gt[1], cp6_gt[1], cp7_gt[1], cp8_gt[1], ey])
            control_points_z_gt = np.array([sz, cp2_gt[2], cp3_gt[2], cp4_gt[2], cp5_gt[2],
                                            cp5_gt[2], cp6_gt[2], cp7_gt[2], cp8_gt[2], ez])

            gt_traj_x = PiecewiseBernsteinPoly(control_points_x_gt, tknots, t_eval)[0, :]
            gt_traj_y = PiecewiseBernsteinPoly(control_points_y_gt, tknots, t_eval)[0, :]
            gt_traj_z = PiecewiseBernsteinPoly(control_points_z_gt, tknots, t_eval)[0, :]

        # Create Plotly figure
        fig = go.Figure()

        # Predicted trajectory
        fig.add_trace(go.Scatter3d(
            x=pred_traj_x, y=pred_traj_y, z=pred_traj_z,
            mode='lines',
            line=dict(color='blue', width=4),
            name='Prediction'
        ))

        # Predicted control points
        fig.add_trace(go.Scatter3d(
            x=pred_points[:, 0], y=pred_points[:, 1], z=pred_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='green'),
            name='Pred CPs'
        ))

        if args.show_gt:
            # Ground truth trajectory
            fig.add_trace(go.Scatter3d(
                x=gt_traj_x, y=gt_traj_y, z=gt_traj_z,
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name='Ground Truth'
            ))
            # Ground truth control points
            fig.add_trace(go.Scatter3d(
                x=gt_points[:, 0], y=gt_points[:, 1], z=gt_points[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', symbol='x'),
                name='GT CPs'
            ))

        # Start/End
        fig.add_trace(go.Scatter3d(
            x=[sx, ex], y=[sy, ey], z=[sz, ez],
            mode='markers',
            marker=dict(size=8, color=['green', 'orange'], symbol='diamond'),
            name='Start/End'
        ))

        if args.show_obs:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            xs = ox + r * np.cos(u) * np.sin(v)
            ys = oy + r * np.sin(u) * np.sin(v)
            zs = oz + r * np.cos(v)
            fig.add_trace(go.Surface(
                x=xs, y=ys, z=zs,
                opacity=0.7,
                colorscale='Reds',
                showscale=False,
                name='Obstacle'
            ))

        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[0, 1], title='X'),
                yaxis=dict(range=[0, 1], title='Y'),
                zaxis=dict(range=[0, 1], title='Z'),
            ),
            title=f"{run_dir.name} | {args.subset} sample {j}"
        )

        # Save as HTML and PNG
        out_html = run_dir / f"traj3d_bebot_{j:03d}.html"
        out_png = run_dir / f"traj3d_bebot_{j:03d}.png"
        fig.write_html(out_html)
        fig.write_image(out_png, width=800, height=800)
        print(f"[SAVED] {out_html} (interactive)")
        print(f"[SAVED] {out_png}")
        fig.show()


if __name__ == "__main__":
    main()