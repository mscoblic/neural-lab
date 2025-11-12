import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import time
import plotly.graph_objects as go
from scipy import stats
import json


# Import BeBOT
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'tools' / 'extra'))
from BeBOT import PiecewiseBernsteinPoly

# Global features
TRAIN = False        # train or load saved model
MODEL_PATH = "models/best_model.pth"
DEBUG = False       # prints shapes for sanity
TIME_EVAL = True   # run timing benchmark
VISCOL = False      # if true, search and visualize collision samples
SELF_EVAL = False   # user input (bottom of script)


@torch.no_grad()
def time_inference_comparison(model, dataset, n_samples=100, n_warmup=10):
    """
    Compare model-only vs full pipeline timing for Transformer
    Uses INDEPENDENT samples for unbiased measurement
    """
    import time

    device = next(model.parameters()).device
    model.eval()

    # Load norm stats
    X_mean_t = torch.from_numpy(X_mean.squeeze(0)).to(device)
    X_std_t = torch.from_numpy(X_std.squeeze(0)).to(device)
    Y_mean_t = torch.from_numpy(Y_mean.squeeze(0)).to(device)
    Y_std_t = torch.from_numpy(Y_std.squeeze(0)).to(device)

    # Warmup
    x, _ = dataset[0]
    x_norm = (x - X_mean_t) / X_std_t
    x_in = x_norm.unsqueeze(0).to(device)
    for _ in range(n_warmup):
        y_out = model(x_in)
        _ = y_out * Y_std_t + Y_mean_t
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
        x_norm = (x_raw - X_mean_t) / X_std_t
        x_in = x_norm.unsqueeze(0).to(device)
        y_norm_out = model(x_in)
        y_out = y_norm_out * Y_std_t + Y_mean_t
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pipeline_times.append((t1 - t0) * 1000)

    # === MEASURE MODEL ONLY (separate loop, independent samples) ===
    model_times = []
    for idx in indices_model:
        x_raw, _ = dataset[idx]

        # Pre-normalize (NOT timed)
        x_norm = (x_raw - X_mean_t) / X_std_t
        x_in = x_norm.unsqueeze(0).to(device)

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
    print("TRANSFORMER INFERENCE TIMING")
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

# Converts tensors into arrays for plotting
def _to_np(t):
   return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def plot_sample_interactive_from_input(model, test_input):
    """Interactive 3D plot using Plotly from raw input array"""
    model.eval()
    device = next(model.parameters()).device

    # Normalize input
    test_input_norm = (test_input - X_mean) / X_std
    test_input_tensor = torch.from_numpy(test_input_norm).to(device)

    # Get prediction
    with torch.no_grad():
        output_norm = model(test_input_tensor)
        output = output_norm.cpu().numpy() * Y_std + Y_mean  # Denormalize

    # Extract points from test_input (already denormalized)
    x0, y0, z0 = test_input[0, 0]
    xf, yf, zf = test_input[0, 1]
    ox, oy, oz = test_input[0, 2]
    cpx, cpy, cpz = test_input[0, 3]
    r = 0.1  # radius

    # Build trajectory
    pred_points = output[0]  # (T_out, 3)
    cp2, cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

    control_points_x = np.array([x0, cp2[0], cp3[0], cp4[0], cp5[0],
                                 cp5[0], cp6[0], cp7[0], cp8[0], xf])
    control_points_y = np.array([y0, cp2[1], cp3[1], cp4[1], cp5[1],
                                 cp5[1], cp6[1], cp7[1], cp8[1], yf])
    control_points_z = np.array([z0, cp2[2], cp3[2], cp4[2], cp5[2],
                                 cp5[2], cp6[2], cp7[2], cp8[2], zf])

    tknots = np.array([0, 0.5, 1.0])
    t_eval = np.linspace(0, 1, 50)

    traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
    traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
    traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

    # Create figure
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
        text=[str(i + 1) for i in range(len(pred_points))],
        textposition='top center',
        name='Predictions'
    ))

    # Start/End
    fig.add_trace(go.Scatter3d(
        x=[x0, xf], y=[y0, yf], z=[z0, zf],
        mode='markers',
        marker=dict(size=10, color=['green', 'orange'], symbol='diamond'),
        name='Start/End'
    ))

    # Heading control point
    fig.add_trace(go.Scatter3d(
        x=[cpx], y=[cpy], z=[cpz],
        mode='markers',
        marker=dict(size=8, color='purple'),
        name='Initial Velocity'
    ))

    # Obstacle sphere
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
            camera=dict(
                eye=dict(x=1.6, y=-1.3, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            )
        ),
        title=f'Center Obstacle Test - 0.699'
    )
    fig.write_image("my_trajectory.png")
    fig.show()  # Opens in browser with full interactivity!

'''
# this function is same as above, but allows for sweeping of inputs for gif
def plot_sample_interactive_from_input(model, test_input, *,
                                       save_path=None, title=None, show=False):

    import time
    t0 = time.perf_counter()

    print("\n[plot] ===============================================")
    print("[plot] Starting interactive plot generation…")

    # ------------------------------------------------------
    # Normalize input
    print("[plot] Normalizing input…")
    test_input_norm = (test_input - X_mean) / X_std
    test_input_tensor = torch.from_numpy(test_input_norm).to(
        next(model.parameters()).device
    )

    # ------------------------------------------------------
    # Forward pass
    print("[plot] Running model inference…")
    with torch.no_grad():
        output_norm = model(test_input_tensor)

    print("[plot] Model output (normalized) shape:", output_norm.shape)

    # Denormalize
    output = output_norm.cpu().numpy() * Y_std + Y_mean
    print("[plot] Output denormalized.")

    # ------------------------------------------------------
    # Extract tokens
    print("[plot] Extracting tokens (start, end, obstacle, control)…")
    x0, y0, z0 = test_input[0, 0]
    xf, yf, zf = test_input[0, 1]
    ox, oy, oz = test_input[0, 2]
    cpx, cpy, cpz = test_input[0, 3]

    # ------------------------------------------------------
    # Build CP trajectory
    print("[plot] Building Bernstein trajectory…")
    pred_points = output[0]
    cp2, cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

    control_points_x = np.array([x0, cp2[0], cp3[0], cp4[0], cp5[0],
                                 cp5[0], cp6[0], cp7[0], cp8[0], xf])
    control_points_y = np.array([y0, cp2[1], cp3[1], cp4[1], cp5[1],
                                 cp5[1], cp6[1], cp7[1], cp8[1], yf])
    control_points_z = np.array([z0, cp2[2], cp3[2], cp4[2], cp5[2],
                                 cp5[2], cp6[2], cp7[2], cp8[2], zf])

    tknots = np.array([0, 0.5, 1.0])
    t_eval = np.linspace(0, 1, 50)

    traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
    traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
    traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

    print("[plot] Trajectory computed.")

    # ------------------------------------------------------
    # Create Plotly figure
    print("[plot] Creating figure and adding traces…")
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z,
        mode='lines', line=dict(color='blue', width=4),
        name='Trajectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=pred_points[:, 0], y=pred_points[:, 1], z=pred_points[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='green'),
        text=[str(i + 1) for i in range(len(pred_points))],
        textposition='top center',
        name='Predictions'
    ))

    fig.add_trace(go.Scatter3d(
        x=[x0, xf], y=[y0, yf], z=[z0, zf],
        mode='markers',
        marker=dict(size=10, color=['green', 'orange'], symbol='diamond'),
        name='Start/End'
    ))

    fig.add_trace(go.Scatter3d(
        x=[cpx], y=[cpy], z=[cpz],
        mode='markers',
        marker=dict(size=8, color='purple'),
        name='Initial Velocity'
    ))

    # Obstacle sphere
    print("[plot] Adding obstacle sphere…")
    r = 0.1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
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

    # ------------------------------------------------------
    # Layout
    print("[plot] Updating layout…\n")
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
        title=title or "Plot"
    )

    # ------------------------------------------------------
    # Save
    if save_path is not None:
        print(f"[plot] Saving PNG → {save_path}")
        fig.write_image(save_path, scale=2)

    # ------------------------------------------------------
    # Show
    if show:
        print("[plot] Displaying figure in browser…")

    dt = time.perf_counter() - t0
    print(f"[plot] Done in {dt:.3f}s")
    print("[plot] ===============================================\n")

    if show:
        fig.show()

    return fig
'''

@torch.no_grad()
def plot_dataset_sample(model, ds, idx, save_path=None, title_prefix="test",
                        elev=45, azim=-70, n_eval=50, plot_continuous=True):
    # inputs:
    #   model -
    #   ds - data set you want to plot
    #   idx - index of the sample
    #

    model.eval()
    device = next(model.parameters()).device        # GPU or CPU

    # fetch one sample (normalized tensors)
    X, Y_true = ds[idx]  # X: (T_in,3), Y_true: (T_out,3)
    Y_pred = model(X.unsqueeze(0).to(device))[0].cpu()  # (T_out,3)

    # prepare stats
    X_mean_t = torch.from_numpy(X_mean.squeeze(0)).to(X.dtype)
    X_std_t  = torch.from_numpy(X_std.squeeze(0)).to(X.dtype)
    Y_mean_t = torch.from_numpy(Y_mean.squeeze(0)).to(Y_pred.dtype)
    Y_std_t  = torch.from_numpy(Y_std.squeeze(0)).to(Y_pred.dtype)

    # denormalize
    X_denorm      = X * X_std_t + X_mean_t
    Y_true_denorm = Y_true * Y_std_t + Y_mean_t
    Y_pred_denorm = Y_pred * Y_std_t + Y_mean_t

    # recover tokens
    x0, y0, z0 = X_denorm[0].tolist()    # start
    xf, yf, zf = X_denorm[1].tolist()    # end
    ox, oy, oz = X_denorm[2].tolist()    # obstacle
    cpx, cpy, cpz = X_denorm[3].tolist() # control

    # --- Build polynomial ---
    if plot_continuous:
        pred_points = Y_pred_denorm.numpy()

        cp2 = pred_points[0]
        cp3 = pred_points[1]
        cp4 = pred_points[2]
        cp5 = pred_points[3]        # duplicate cp
        cp6 = pred_points[4]
        cp7 = pred_points[5]
        cp8 = pred_points[6]

        control_points_x = np.array([x0, cp2[0], cp3[0], cp4[0], cp5[0],
                                     cp5[0], cp6[0], cp7[0], cp8[0], xf])
        control_points_y = np.array([y0, cp2[1], cp3[1], cp4[1], cp5[1],
                                     cp5[1], cp6[1], cp7[1], cp8[1], yf])
        control_points_z = np.array([z0, cp2[2], cp3[2], cp4[2], cp5[2],
                                     cp5[2], cp6[2], cp7[2], cp8[2], zf])

        tknots = np.array([0, 0.5, 1.0])
        t_eval = np.linspace(0, 1, n_eval)

        traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
        traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
        traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]


    # --- 3D scatter plot ---
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)  # <<< set camera
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    if plot_continuous:
        ax.plot(traj_x, traj_y, traj_z, 'b-', linewidth=2,
                label='piecewise bpoly', alpha=0.7, zorder=1)

    # start / end / control / obstacle
    ax.scatter([x0], [y0], [z0], s=80, marker = 'o', label='start', depthshade=False)
    ax.scatter([xf], [yf], [zf], s=120, marker = '*', label='end', depthshade=False)
    ax.scatter([ox], [oy], [oz], s=120, color='red', label='obstacle center', depthshade=False)

    # ground truth points
    Yt = _to_np(Y_true_denorm)
    ax.scatter(Yt[:, 0], Yt[:, 1], Yt[:, 2], s=36, color='black', marker='x', label='ground truth', depthshade=False)

    # predicted points
    Yp = _to_np(Y_pred_denorm)
    ax.scatter(Yp[:, 0], Yp[:, 1], Yp[:, 2], s=28, marker='o', label='prediction', depthshade=False)

    # label each predicted point
    for i, (x, y, z) in enumerate(Yp):
        ax.text(float(x), float(y), float(z), str(i + 1),
                fontsize=8, ha='left', va='bottom', color='blue')

    ax.scatter([cpx], [cpy], [cpz], s=80, marker='o', label='heading control point', depthshade=False)

    # draw obstacle sphere (optional visualization)
    R = 0.1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = ox + R * np.cos(u) * np.sin(v)
    ys = oy + R * np.sin(u) * np.sin(v)
    zs = oz + R * np.cos(v)
    ax.plot_surface(xs, ys, zs, color='red', alpha=1, linewidth=0)

    # cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{title_prefix} sample idx={idx}")
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

@torch.no_grad()
def plot_collision_sample(model, ds, idx, save_path=None, title_prefix="collision",
                          elev=45, azim=-70, n_eval=50, obstacle_radius=0.1):
    """
    Visualize a trajectory sample highlighting collision points with the obstacle.
    """
    model.eval()
    device = next(model.parameters()).device

    # fetch one sample (normalized tensors)
    X, Y_true = ds[idx]
    Y_pred = model(X.unsqueeze(0).to(device))[0].cpu()

    # prepare stats
    X_mean_t = torch.from_numpy(X_mean.squeeze(0)).to(X.dtype)
    X_std_t = torch.from_numpy(X_std.squeeze(0)).to(X.dtype)
    Y_mean_t = torch.from_numpy(Y_mean.squeeze(0)).to(Y_pred.dtype)
    Y_std_t = torch.from_numpy(Y_std.squeeze(0)).to(Y_pred.dtype)

    # denormalize
    X_denorm = X * X_std_t + X_mean_t
    Y_true_denorm = Y_true * Y_std_t + Y_mean_t
    Y_pred_denorm = Y_pred * Y_std_t + Y_mean_t

    # recover tokens
    x0, y0, z0 = X_denorm[0].tolist()
    xf, yf, zf = X_denorm[1].tolist()
    ox, oy, oz = X_denorm[2].tolist()
    cpx, cpy, cpz = X_denorm[3].tolist()

    # Build polynomial trajectory
    pred_points = Y_pred_denorm.numpy()
    cp3 = pred_points[0]
    cp4 = pred_points[1]
    cp5 = pred_points[2]
    cp6 = pred_points[3]
    cp7 = pred_points[4]
    cp8 = pred_points[5]

    control_points_x = np.array([x0, cpx, cp3[0], cp4[0], cp5[0],
                                 cp5[0], cp6[0], cp7[0], cp8[0], xf])
    control_points_y = np.array([y0, cpy, cp3[1], cp4[1], cp5[1],
                                 cp5[1], cp6[1], cp7[1], cp8[1], yf])
    control_points_z = np.array([z0, cpz, cp3[2], cp4[2], cp5[2],
                                 cp5[2], cp6[2], cp7[2], cp8[2], zf])

    tknots = np.array([0, 0.5, 1.0])
    t_eval = np.linspace(0, 1, n_eval)

    traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
    traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
    traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

    # Check for collisions along trajectory
    obstacle_center = np.array([ox, oy, oz])
    distances = np.sqrt((traj_x - ox) ** 2 + (traj_y - oy) ** 2 + (traj_z - oz) ** 2)
    collision_mask = distances < obstacle_radius
    num_collisions = np.sum(collision_mask)
    min_distance = np.min(distances)

    # Find closest point to obstacle
    closest_idx = np.argmin(distances)

    # Split trajectory into safe and collision segments
    safe_mask = ~collision_mask

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Plot trajectory - safe parts in blue, collision parts in red
    if np.any(safe_mask):
        ax.plot(traj_x[safe_mask], traj_y[safe_mask], traj_z[safe_mask],
                'b-', linewidth=2, label='Safe trajectory', alpha=0.7)

    if np.any(collision_mask):
        ax.plot(traj_x[collision_mask], traj_y[collision_mask], traj_z[collision_mask],
                'r-', linewidth=3, label='COLLISION ZONE', alpha=0.9, zorder=10)

    # Highlight closest point to obstacle
    ax.scatter([traj_x[closest_idx]], [traj_y[closest_idx]], [traj_z[closest_idx]],
               s=150, color='orange', marker='X', label=f'Closest point (d={min_distance:.4f})',
               edgecolors='black', linewidths=2, depthshade=False, zorder=11)

    # Start / end / control / obstacle
    ax.scatter([x0], [y0], [z0], s=80, marker='o', label='start', depthshade=False)
    ax.scatter([xf], [yf], [zf], s=120, marker='*', label='end', depthshade=False)
    ax.scatter([ox], [oy], [oz], s=120, color='red', label='obstacle center',
               depthshade=False, zorder=5)

    # Ground truth points
    Yt = Y_true_denorm.numpy()
    ax.scatter(Yt[:, 0], Yt[:, 1], Yt[:, 2], s=36, color='black',
               marker='x', label='ground truth', depthshade=False)

    # Predicted control points with distance labels
    Yp = Y_pred_denorm.numpy()
    cp_distances = [np.linalg.norm(cp - obstacle_center) for cp in Yp]

    for i, (cp, dist) in enumerate(zip(Yp, cp_distances)):
        color = 'green' if dist > obstacle_radius else 'red'
        ax.scatter([cp[0]], [cp[1]], [cp[2]], s=50, marker='o',
                   color=color, alpha=0.6, depthshade=False)
        ax.text(float(cp[0]), float(cp[1]), float(cp[2]),
                f'{i + 1}\nd={dist:.3f}',
                fontsize=7, ha='left', va='bottom', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.scatter([cpx], [cpy], [cpz], s=80, marker='o',
               label='heading control point', depthshade=False)

    # Draw obstacle sphere - make it semi-transparent
    R = obstacle_radius
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    xs = ox + R * np.cos(u) * np.sin(v)
    ys = oy + R * np.sin(u) * np.sin(v)
    zs = oz + R * np.cos(v)
    ax.plot_surface(xs, ys, zs, color='red', alpha=0.3, linewidth=0)

    # Also draw wireframe for clarity
    ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.5, linewidth=0.5)

    # Draw convex hulls for each segment (optional visualization)
    # Segment 1: [x0, cpx, cp3, cp4, cp5]
    seg1_points = np.array([[x0, y0, z0], [cpx, cpy, cpz],
                            cp3, cp4, cp5])
    # Segment 2: [cp5, cp6, cp7, cp8, xf]
    seg2_points = np.array([cp5, cp6, cp7, cp8, [xf, yf, zf]])

    from scipy.spatial import ConvexHull
    try:
        hull1 = ConvexHull(seg1_points)
        hull2 = ConvexHull(seg2_points)

        # Plot convex hull edges
        for simplex in hull1.simplices:
            ax.plot(seg1_points[simplex, 0], seg1_points[simplex, 1],
                    seg1_points[simplex, 2], 'c--', alpha=0.3, linewidth=0.5)
        for simplex in hull2.simplices:
            ax.plot(seg2_points[simplex, 0], seg2_points[simplex, 1],
                    seg2_points[simplex, 2], 'm--', alpha=0.3, linewidth=0.5)
    except:
        pass  # Skip if convex hull fails (coplanar points, etc.)

    # Cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    collision_status = "COLLISION" if num_collisions > 0 else "SAFE"
    ax.set_title(f"{title_prefix} idx={idx} | {collision_status}\n"
                 f"Collision points: {num_collisions}/{n_eval} | "
                 f"Min distance: {min_distance:.4f} | Radius: {obstacle_radius}")

    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # Print diagnostic info
    print(f"\n{'=' * 60}")
    print(f"Sample {idx} Analysis:")
    print(f"{'=' * 60}")
    print(f"Collision Status: {collision_status}")
    print(f"Collision points: {num_collisions}/{n_eval} ({100 * num_collisions / n_eval:.2f}%)")
    print(f"Minimum distance to obstacle: {min_distance:.6f}")
    print(f"Obstacle radius: {obstacle_radius}")
    print(f"Safety margin: {min_distance - obstacle_radius:.6f}")
    print(f"\nControl Point Distances to Obstacle:")
    all_cps = [('start', x0, y0, z0), ('heading', cpx, cpy, cpz)] + \
              [(f'cp{i + 3}', *cp) for i, cp in enumerate(Yp)] + \
              [('end', xf, yf, zf)]
    for name, x, y, z in all_cps:
        dist = np.linalg.norm(np.array([x, y, z]) - obstacle_center)
        status = "✓ SAFE" if dist > obstacle_radius else "✗ PENETRATES"
        print(f"  {name:8s}: {dist:.6f}  {status}")
    print(f"{'=' * 60}\n")

# Visualize many samples quickly
@torch.no_grad()
def plot_many_samples(model, ds, indices, title_prefix="test"):
    """Convenience: plot several dataset samples by index."""
    for j, idx in enumerate(indices):
        plot_dataset_sample(model, ds, idx, title_prefix=title_prefix)

# START HERE
# Splits input into multiple attention heads, applies attention, combines results
class MultiHeadAttention(nn.Module):
    # d_model: dimensionality of the input, number of elements in the embedded input
    # num_heads: number of attention heads
    # d_k: dimension of each head's key, query, value
    # W_x: transformation weights for query, key, value, output
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Make sure the input dimension is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model      # model width
        self.num_heads = num_heads      # number of attention heads
        self.d_k = d_model // num_heads     # per-head dimension

        # Transformations
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    # attn_scores: dot product between query and all keys (in a matrix)
    # attn_probs: relative importance probability between 0 and 1
    # output: new embedding for tokens after importance is applied
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided - no look ahead
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, v)

        return output

    # Allow for parallel computation of multiple attention heads, split for each attention head
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    # After applying attention to each head separately, this combines the results back into a single tensor
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    # Main computation
    def forward(self, q, k, v, mask=None):
        # Apply linear transformations and split into multiple heads
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        # Scaled dot product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Simple two layer feedforward newtork
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))   # Add and normalize
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))     # Add and normalize
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, output_dim):
        super(TransformerEncoder, self).__init__()

        # Project raw inputs into model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_seq_length)

        # Stacked encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

        # Optional output projection
        self.head = nn.Linear(d_model, output_dim) if output_dim is not None else None

    def forward(self, x):

        # Project inputs into model dim + add positions
        h = self.input_proj(x)
        h = self.pos_enc(h)

        # Pass through encoder layers
        for layer in self.layers:
            h = layer(h, mask=None)

        # Normalzie final representation
        h = self.norm(h)

        # Optionally project to outputs
        return self.head(h) if self.head is not None else h

class FlattenMLPHead(nn.Module):
    def __init__(self, seq_len, d_model, hidden=256, t_out=7, out_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.t_out   = t_out
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(seq_len * d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, t_out * out_dim)
        )

    def forward(self, h):  # h: (B, seq_len, d_model)
        B, L, D = h.shape
        y = self.net(h.reshape(B, L * D))     # (B, t_out*out_dim)
        return y.view(B, self.t_out, self.out_dim)  # (B, 7, 2)

class TrajModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x):
        h = self.encoder(x)   # (B, T_in, d_model)
        return self.head(h)   # (B, 7, 2)


class TrajDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):  return self.X.shape[0]
    def __getitem__(self, i):  return self.X[i], self.Y[i]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


T_in = 4
file_path = "../data/CA3D/CA3D_constant_radius_10k.xlsx"
df = pd.read_excel(file_path)

# Build 4 semantic tokens, each 2-D: start, end, obstacle, control
start    = df[["x0","y0", "z0"]].to_numpy(np.float32)  # (N, 2)
end      = df[["xf","yf", "zf"]].to_numpy(np.float32)  # (N, 2)
obstacle = df[["ox","oy", "oz"]].to_numpy(np.float32)  # (N, 2)
control  = df[["vxinit","vyinit", "vzinit"]].to_numpy(np.float32)  # (N, 2)

output_cols = ["x2", "x3", "x4", "x5", "x6", "x7", "x8","y2", "y3", "y4", "y5", "y6", "y7", "y8","z2", "z3", "z4", "z5", "z6", "z7","z8"]
#output_cols = ["x3", "x4", "x5", "x6", "x7", "x8", "x9", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "z3", "z4", "z5", "z6", "z7", "z8", "z9"]
T_out = len(output_cols) // 3
if DEBUG == True:
    print("Derived T_out =", T_out)

# Stack into tokens → (N, 4, 3)
X_np = np.stack([start, end, obstacle, control], axis=1)
Y_np = df[output_cols].to_numpy(dtype=np.float32)    # (N, 20)

# Targets (unchanged): (N,14) -> (N, 7, 2)
N = Y_np.shape[0]
Y_np = Y_np.reshape(N, 3, T_out).transpose(0, 2, 1)  # (N, 7, 2)

# Check statistics of NEW data only (rows 20000+)
if len(df) > 20000:
    new_data_mask = np.arange(len(df)) >= 20000
    new_obstacles = df.iloc[20000:][['ox', 'oy', 'oz']].to_numpy()
    new_velocities = df.iloc[20000:][['vxinit', 'vyinit', 'vzinit']].to_numpy()
    new_outputs = df.iloc[20000:][output_cols].to_numpy()

    print("\n=== NEW PATCH DATA ANALYSIS ===")
    print(f"New data rows: {len(df) - 20000}")
    print(f"Obstacle positions - mean: {new_obstacles.mean(axis=0)}")
    print(f"Obstacle positions - std: {new_obstacles.std(axis=0)}")
    print(f"Velocity magnitudes - mean: {np.linalg.norm(new_velocities, axis=1).mean():.4f}")
    print(f"Output trajectories - any NaN? {np.isnan(new_outputs).any()}")
    print(f"Output trajectories - any Inf? {np.isinf(new_outputs).any()}")

# Create data
seed = 42
rng = np.random.default_rng(seed)
perm = rng.permutation(N)

# train split
n_train = int(0.8 * N)
idx_train = perm[:n_train]
idx_test  = perm[n_train:]

print(f"n_train: {n_train}, n_test: {len(idx_test)}")

X_train = X_np[idx_train]  # Extract training samples
Y_train = Y_np[idx_train]

X_mean = X_train.mean(axis=0, keepdims=True)  # (1, 4, 3)
X_std = X_train.std(axis=0, keepdims=True) + 1e-8
Y_mean = Y_train.mean(axis=0, keepdims=True)  # (1, 6, 3)
Y_std = Y_train.std(axis=0, keepdims=True) + 1e-8

# === 4. APPLY NORMALIZATION TO ALL DATA (using training stats) ===
X_np_normalized = (X_np - X_mean) / X_std  # Apply to full array
Y_np_normalized = (Y_np - Y_mean) / Y_std

train_ds = TrajDataset(X_np_normalized[idx_train], Y_np_normalized[idx_train])
test_ds = TrajDataset(X_np_normalized[idx_test], Y_np_normalized[idx_test])

print("\n=== Normalization Check ===")
print(f"Training X - mean: {X_np_normalized[idx_train].mean():.6f}, std: {X_np_normalized[idx_train].std():.6f}")
print(f"Training Y - mean: {Y_np_normalized[idx_train].mean():.6f}, std: {Y_np_normalized[idx_train].std():.6f}")
print(f"Test X - mean: {X_np_normalized[idx_test].mean():.6f}, std: {X_np_normalized[idx_test].std():.6f}")
print(f"Test Y - mean: {Y_np_normalized[idx_test].mean():.6f}, std: {Y_np_normalized[idx_test].std():.6f}")
print("Training should be ~0 mean, ~1 std. Test will be slightly different.")

# === 7. SAVE STATS (for inference later) ===
np.savez("models/norm_stats.npz",
         X_mean=X_mean, X_std=X_std,
         Y_mean=Y_mean, Y_std=Y_std)

if DEBUG == True:
    print("X mean/std:", X_np.mean(), X_np.std())
    print("Y mean/std:", Y_np.mean(), Y_np.std())

if DEBUG == True:
    print("After normalization:")
    print("X mean:", X_np.mean(), "std:", X_np.std())
    print("Y mean:", Y_np.mean(), "std:", Y_np.std())
    print("Example normalized X[0]:", X_np[0])


# Hyperparams
input_dim = 3
d_model   = 64
num_heads = 4
num_layers= 2
d_ff      = 128
dropout   = 0.2
output_dim= 3        # (x,y) per step
max_seq_length = T_in
batch_size = 64
lr = 1e-3

if DEBUG == True:
    print("max_seq_length =", max_seq_length)

# Identify excel row for test data
def excel_row_from_test_idx(ds_idx):
    """
    Map a test-dataset index to the original DataFrame row and Excel row number.
    ds_idx: index into test_ds (e.g., 2)
    """
    orig_idx = int(idx_test[ds_idx])     # row in df / X_np before the split
    excel_row = orig_idx + 2             # +1 for 1-based, +1 for header row
    print(f"test_ds idx {ds_idx} -> df index {orig_idx} -> Excel row {excel_row}")
    # Optional: show the actual row values for sanity
    print(df.iloc[orig_idx])
    return orig_idx, excel_row

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size, shuffle=False)

# Build model (using your classes)
encoder = TransformerEncoder(
    input_dim=input_dim, d_model=d_model, num_heads=num_heads,
    num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length,
    dropout=dropout, output_dim=None  # IMPORTANT: None → return embeddings
).to(device)

head = FlattenMLPHead(seq_len=T_in, d_model=d_model, hidden=256, t_out=T_out, out_dim=output_dim).to(device)
if DEBUG == True:
    print("Head t_out =", head.t_out)  # should be 6

model = TrajModel(encoder, head).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

test_mse_hist = []
coll_rate_hist = []

def train_one_epoch():
    model.train()
    running = 0.0
    for X, Y in train_dl:
        X, Y = X.to(device), Y.to(device)         # X: (B,T,3), Y: (B,T,2)
        optimizer.zero_grad()
        Y_hat = model(X)                           # (B,T,2)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()
        running += loss.item() * X.size(0)

    if DEBUG == True:
        for X, Y in train_dl:
            print("X shape:", X.shape, "Y shape:", Y.shape)  # expect [B, 4, 2], [B, 6, 2]
            print("Sample tokens [start, end, obstacle, cp2]:\n", X[0])
            break

    return running / len(train_dl.dataset)

@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    running = 0.0
    for X, Y in loader:                # loop over whatever loader you give it
        X, Y = X.to(device), Y.to(device)
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        running += loss.item() * X.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def count_collisions(model, loader, radius=0.1):
    """
    Returns fraction of trajectories with at least one predicted point
    inside the obstacle sphere of given radius (in ORIGINAL units).
    Works with 3D tokens and per-token or global norm stats.
    """
    model.eval()
    device = next(model.parameters()).device

    # bring stats to device
    X_mean_t = torch.from_numpy(X_mean).to(device)  # (1, T_in or 1, 3)
    X_std_t  = torch.from_numpy(X_std).to(device)
    Y_mean_t = torch.from_numpy(Y_mean).to(device)  # (1, T_out or 1, 3)
    Y_std_t  = torch.from_numpy(Y_std).to(device)

    def token_stats(stats_t, idx):
        # stats_t: (1, K, 3). If K==1 it's global; else per-token.
        return stats_t[:, 0, :] if stats_t.size(1) == 1 else stats_t[:, idx, :]

    total = 0
    collided = 0

    for X, _ in loader:                       # X: (B, T_in, 3) normalized
        X = X.to(device)
        Yp = model(X)                          # (B, T_out, 3) normalized

        # denorm predictions (support global or per-token Y stats)
        if Y_mean_t.size(1) == 1:
            Yp_den = Yp * Y_std_t[:, 0, :] + Y_mean_t[:, 0, :]
        else:
            Yp_den = Yp * Y_std_t + Y_mean_t  # broadcast per-step

        # obstacle center = token index 2
        obs_norm = X[:, 2, :]                 # (B, 3)
        xm = token_stats(X_mean_t, 2)         # (1, 3)
        xs = token_stats(X_std_t, 2)          # (1, 3)
        obs_den = obs_norm * xs + xm          # (B, 3)

        # distances to obstacle center
        dist = torch.linalg.norm(Yp_den - obs_den[:, None, :], dim=-1)  # (B, T_out)
        inside = (dist < radius -  0)   # was 0.005

        collided += inside.any(dim=1).sum().item()
        total += X.size(0)

    return collided / max(total, 1)

@torch.no_grad()
def count_collisions_continuous(model, loader, radius=0.1, buffer=0.0, n_eval=500):
    """
    Count collisions on the continuous Bernstein polynomial trajectory.

    radius: obstacle radius
    buffer: safety margin (positive = more conservative)
    n_eval: number of points to evaluate along curve (higher = more accurate)
    """
    model.eval()
    device = next(model.parameters()).device

    # Load normalization stats
    X_mean_t = torch.from_numpy(X_mean).to(device)
    X_std_t = torch.from_numpy(X_std).to(device)
    Y_mean_t = torch.from_numpy(Y_mean).to(device)
    Y_std_t = torch.from_numpy(Y_std).to(device)

    total = 0
    collided = 0

    for X, _ in loader:
        X = X.to(device)
        Yp = model(X)  # (B, 6, 3)

        # Denormalize predictions
        Yp_den = Yp * Y_std_t + Y_mean_t

        # Denormalize inputs
        X_den = X * X_std_t + X_mean_t

        # For each trajectory in batch
        for i in range(X.size(0)):
            # Extract points
            x0, y0, z0 = X_den[i, 0].cpu().numpy()
            xf, yf, zf = X_den[i, 1].cpu().numpy()
            ox, oy, oz = X_den[i, 2].cpu().numpy()
            cpx, cpy, cpz = X_den[i, 3].cpu().numpy()

            pred_points = Yp_den[i].cpu().numpy()  # (6, 3)
            cp3 = pred_points[0]
            cp4 = pred_points[1]
            cp5 = pred_points[2]
            cp6 = pred_points[3]
            cp7 = pred_points[4]
            cp8 = pred_points[5]

            # Build control point arrays with cp5 duplicated
            control_points_x = np.array([x0, cpx, cp3[0], cp4[0], cp5[0],
                                         cp5[0], cp6[0], cp7[0], cp8[0], xf])
            control_points_y = np.array([y0, cpy, cp3[1], cp4[1], cp5[1],
                                         cp5[1], cp6[1], cp7[1], cp8[1], yf])
            control_points_z = np.array([z0, cpz, cp3[2], cp4[2], cp5[2],
                                         cp5[2], cp6[2], cp7[2], cp8[2], zf])

            # Evaluate continuous trajectory
            tknots = np.array([0, 0.5, 1.0])
            t_eval = np.linspace(0, 1, n_eval)

            traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
            traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
            traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

            # Stack into (n_eval, 3)
            trajectory = np.column_stack([traj_x, traj_y, traj_z])

            # Compute distances to obstacle center
            obstacle_center = np.array([ox, oy, oz])
            distances = np.linalg.norm(trajectory - obstacle_center, axis=1)

            # Check if any point violates the constraint
            if np.any(distances < radius - buffer):
                collided += 1

            total += 1

    return collided / max(total, 1)

def find_collision_samples(model, dataset, radius=0.1, n_eval=500, n_samples=None, buffer=0):
    """
    Find all samples in dataset that have bpoly trajectory collisions.

    Args:
        model: trained model
        dataset: TrajDataset (test_ds or train_ds)
        radius: obstacle radius
        n_eval: number of points to evaluate along curve
        n_samples: if provided, only check first n_samples (for speed)

    Returns:
        collision_indices: list of dataset indices that collide
    """
    model.eval()
    device = next(model.parameters()).device

    collision_indices = []
    n_check = n_samples if n_samples else len(dataset)

    print(f"Checking {n_check} samples for collisions...")

    for i in range(n_check):
        if i % 100 == 0:
            print(f"  Checked {i}/{n_check}...")

        X, Y_true = dataset[i]
        X_batch = X.unsqueeze(0).to(device)

        with torch.no_grad():
            Yp = model(X_batch)

        # Denormalize
        X_mean_t = torch.from_numpy(X_mean).to(device)
        X_std_t = torch.from_numpy(X_std).to(device)
        Y_mean_t = torch.from_numpy(Y_mean).to(device)
        Y_std_t = torch.from_numpy(Y_std).to(device)

        X_den = X_batch * X_std_t + X_mean_t
        Yp_den = Yp * Y_std_t + Y_mean_t

        # Extract points
        x0, y0, z0 = X_den[0, 0].cpu().numpy()
        xf, yf, zf = X_den[0, 1].cpu().numpy()
        ox, oy, oz = X_den[0, 2].cpu().numpy()
        cpx, cpy, cpz = X_den[0, 3].cpu().numpy()

        pred_points = Yp_den[0].cpu().numpy()
        cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

        # Build control points
        control_points_x = np.array([x0, cpx, cp3[0], cp4[0], cp5[0],
                                     cp5[0], cp6[0], cp7[0], cp8[0], xf])
        control_points_y = np.array([y0, cpy, cp3[1], cp4[1], cp5[1],
                                     cp5[1], cp6[1], cp7[1], cp8[1], yf])
        control_points_z = np.array([z0, cpz, cp3[2], cp4[2], cp5[2],
                                     cp5[2], cp6[2], cp7[2], cp8[2], zf])

        # Evaluate trajectory
        tknots = np.array([0, 0.5, 1.0])
        t_eval = np.linspace(0, 1, n_eval)

        traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
        traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
        traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

        trajectory = np.column_stack([traj_x, traj_y, traj_z])
        obstacle_center = np.array([ox, oy, oz])
        distances = np.linalg.norm(trajectory - obstacle_center, axis=1)

        # Check collision
        if np.any(distances < radius - buffer):
            collision_indices.append(i)

    print(f"\nFound {len(collision_indices)} collision samples out of {n_check}")
    print(f"Collision rate: {100 * len(collision_indices) / n_check:.2f}%")

    return collision_indices

# Scheduled LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = 'min', # minimize metric
    factor = 0.5, # reduce LR by half
    patience = 3, # wait 3 epochs before reducing
)

EPOCHS = 100
best_test_mse = float('inf')
patience_counter = 0
patience_limit = 10
train_losses = []
if TRAIN:
    for epoch in range(EPOCHS):
        # train
        tr = train_one_epoch()
        train_losses.append(tr)

        # test MSE this epoch
        te = eval_epoch(test_dl)
        test_mse_hist.append(te)

        # collision rate this epoch (denormed, 3D)
        #coll = count_collisions_continuous(model, test_dl, radius=0.1, buffer = 0, n_eval=50)
        coll = count_collisions(model, test_dl)
        coll_rate_hist.append(coll)

        scheduler.step(te)
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping logic
        if te < best_test_mse:
            best_test_mse = te
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch + 1}!")
            break

        print(f"epoch {epoch+1:02d} | train MSE {tr:.6f} | test MSE {te:.6f} | collisions {coll:.2%} | LR {lr:.6f}")
    # ---- PLOTS ----
    os.makedirs("figs", exist_ok=True)

    # (A) Collision rate vs epoch (percentage)
    plt.figure(figsize=(6,4))
    xs = np.arange(1, len(coll_rate_hist)+1)
    plt.plot(xs, np.array(coll_rate_hist)*100.0, linewidth=2, label='Collision rate (%)')
    plt.xlabel('Epoch'); plt.ylabel('Collision rate (%)')
    plt.title('Collision rate over epochs')
    plt.grid(True); plt.legend()
    plt.savefig("figs/collision_rate_over_epochs.png", dpi=160, bbox_inches="tight")
    plt.close()

    # (B) Train/Test MSE vs epoch (solid lines, no markers)
    plt.figure(figsize=(6,4))
    xs = np.arange(1, len(test_mse_hist)+1)
    plt.plot(xs, test_mse_hist, linestyle='-', linewidth=2, label='Test MSE')
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, linestyle='-', linewidth=2, label='Train MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.title('MSE over epochs')
    plt.grid(True); plt.legend()
    plt.savefig("figs/mse_over_epochs.png", dpi=160, bbox_inches="tight")
    plt.close()
else:
    # Load the saved model when not training
    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")

if TIME_EVAL:

    timing_stats = time_inference_comparison(model, test_ds, n_samples=100)

    # Save results
    transformer_results = {
        'model_only_ms': timing_stats['model_mean'],
        'full_pipeline_ms': timing_stats['pipeline_mean'],
        'model_only_hz': 1000.0 / timing_stats['model_mean'],
        'full_pipeline_hz': 1000.0 / timing_stats['pipeline_mean']
    }

    with open('transformer_timing_results.json', 'w') as f:
        json.dump(transformer_results, f, indent=2)

    print("[SAVED] transformer_timing_results.json")

# Standard evaluation
print("\nComputing test metrics...")
test_mse = eval_epoch(test_dl)
#test_coll = count_collisions_continuous(model, test_dl, radius=0.1, buffer = 0, n_eval=50)
test_coll = count_collisions(model, test_dl)
print(f"Test MSE: {test_mse:.6f}")
print(f"Test collision rate: {test_coll * 100:.2f}%")

# plot first 3 validation samples
plot_many_samples(model, test_ds, indices=[0,1,2], title_prefix="test")
#_ = excel_row_from_test_idx(2)

# or one specific index and save it
#plot_dataset_sample(model, test_ds, idx=5, save_path="test_sample_005.png")

if VISCOL:
    # Find and visualize collision samples
    print("\n" + "=" * 60)
    print("FINDING COLLISION SAMPLES")
    print("=" * 60)

    collision_indices = find_collision_samples(model, test_ds, radius=0.1, n_eval=50)

    if len(collision_indices) > 0:
        print(f"\nVisualizing first collision sample...")
        plot_collision_sample(model, test_ds, idx=collision_indices[0])

        # Get Excel row for this collision
        excel_row_from_test_idx(collision_indices[0])

        # Optionally visualize a few more
        if len(collision_indices) > 1:
            print(f"\nVisualizing second collision sample...")
            plot_collision_sample(model, test_ds, idx=collision_indices[1])

        if len(collision_indices) > 2:
            print(f"\nVisualizing third collision sample...")
            plot_collision_sample(model, test_ds, idx=collision_indices[2])
    else:
        print("No collisions found! 🎉")

# Analyze initial velocity distributions
obstacles = X_train[:, 2, :]  # Shape: (N, 3)
start_points = X_train[:, 0, :]
init_velocity = X_train[:, 3, :]  # The 4th token is initial velocity

# Distance from start to obstacle
distances = np.linalg.norm(obstacles - start_points, axis=1)

# All velocity magnitudes
all_velocity_magnitudes = np.linalg.norm(init_velocity, axis=1)

# Close obstacle cases (< 0.2)
close_mask = distances < 0.2
close_velocities = init_velocity[close_mask]
close_velocity_magnitudes = np.linalg.norm(close_velocities, axis=1)

# temporary check
# Check the shape and structure
print(f"X_train shape: {X_train.shape}")
print(f"init_velocity shape: {init_velocity.shape}")

# Look at the actual values
print(f"\nFirst 5 init_velocity samples:")
print(init_velocity[:5])

# Check individual component maxes
print(f"\nMax of each component:")
print(f"X-component max: {init_velocity[:, 0].max():.4f}")
print(f"Y-component max: {init_velocity[:, 1].max():.4f}")
print(f"Z-component max: {init_velocity[:, 2].max():.4f}")

# Manually compute magnitude for first sample to verify
sample_0_mag = np.sqrt(init_velocity[0, 0]**2 +
                       init_velocity[0, 1]**2 +
                       init_velocity[0, 2]**2)
print(f"\nFirst sample magnitude (manual): {sample_0_mag:.4f}")
print(f"First sample magnitude (np.linalg.norm): {all_velocity_magnitudes[0]:.4f}")

# Print statistics
print(f"\n=== ALL TRAINING DATA ({len(all_velocity_magnitudes)} samples) ===")
print(f"Initial velocity magnitude - mean: {all_velocity_magnitudes.mean():.4f}")
print(f"Initial velocity magnitude - std: {all_velocity_magnitudes.std():.4f}")
print(f"Initial velocity magnitude - min: {all_velocity_magnitudes.min():.4f}")
print(f"Initial velocity magnitude - max: {all_velocity_magnitudes.max():.4f}")
print(f"Number with zero velocity (< 0.01): {np.sum(all_velocity_magnitudes < 0.01)}")
print(f"Percentage with zero velocity: {np.sum(all_velocity_magnitudes < 0.01) / len(all_velocity_magnitudes) * 100:.2f}%")

if SELF_EVAL:

    # for sweeping

    # Sweep oz from 0.68 to 0.72 by 0.001 and save PNGs from the *interactive* plot
    '''
    zsweep = np.arange(0.68, 0.72, 0.001)
    #zsweep = np.arange(0.2, 0.8, 0.01)

    base = np.array([[
        [0.0, 0.0, 0.0],  # start
        [1.0, 1.0, 1.0],  # end
        [0.7, 0.7, 0.68],  # obstacle (oz will be overwritten)
        [0.0, 0.0, 0.0]  # control: ZERO VELOCITY
    ]], dtype=np.float32)

    for oz in zsweep:
        ti = base.copy()
        ti[0, 2, 2] = float(oz)  # set obstacle z
        fname = f"figs/interactive_sweep/oz_{oz:.3f}.png"
        title = f"Center Obstacle Test – oz={oz:.3f}"
        plot_sample_interactive_from_input(model, ti, save_path=fname, title=title, show=False)

        print(f"Obstacle position: {ti[0, 2, :]}")
    '''

    # Create a test input with zero velocity
    test_input = np.array([[
        [0.0, 0.0, 0.0],   # start: x0, y0, z0
        [1.0, 1.0, 1.0],   # end: xf, yf, zf
        [0.5, 0.5, 0.5],   # obstacle: ox, oy, oz
        [0.0, 0.0, 0.0]    # control: ZERO VELOCITY
    ]], dtype=np.float32)  # Shape: (1, 4, 3)

    plot_sample_interactive_from_input(model, test_input)

    # Normalize it
    test_input_norm = (test_input - X_mean) / X_std
    test_input_tensor = torch.from_numpy(test_input_norm).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output_norm = model(test_input_tensor)
        output = output_norm.cpu().numpy() * Y_std + Y_mean  # Denormalize

    # Print results
    print("\n=== ZERO VELOCITY TEST ===")
    print("Start:", test_input[0, 0])
    print("End:", test_input[0, 1])
    print("Obstacle:", test_input[0, 2])
    print("Initial velocity:", test_input[0, 3])
    print("\nPredicted trajectory shape:", output.shape)
    print("First predicted point:", output[0, 0])
    print("Last predicted point:", output[0, -1])

    # Create fake dataset with this one sample
    test_Y_dummy = np.zeros((1, T_out, 3), dtype=np.float32)  # We don't care about ground truth
    zero_vel_ds = TrajDataset(test_input_norm, (test_Y_dummy - Y_mean) / Y_std)

    # Plot it using your existing function!
    plot_dataset_sample(model, zero_vel_ds, idx=0,
                        save_path="figs/zero_velocity_test.png",
                        title_prefix="Zero Velocity Test")

    # Check for collisions
    pred_points = output[0]  # Shape: (17, 3)
    obstacle_center = test_input[0, 2]  # [0.5, 0.5, 0.5]
    R = 0.1  # obstacle radius

    # Calculate distances from each predicted point to obstacle center
    distances = np.linalg.norm(pred_points - obstacle_center, axis=1)

    # Check if any point is inside the obstacle
    collisions = distances < R
    num_collisions = np.sum(collisions)

    print("\n=== COLLISION DETECTION ===")
    print(f"Obstacle center: {obstacle_center}")
    print(f"Obstacle radius: {R}")
    print(f"Minimum distance to obstacle: {distances.min():.6f}")
    print(f"Number of collision points: {num_collisions}/{len(pred_points)}")
    if num_collisions > 0:
        print("⚠️  COLLISION DETECTED!")
        collision_indices = np.where(collisions)[0]
        print(f"Collision at points: {collision_indices + 2}")  # +2 because we start counting from point 2
    else:
        print("✓ No collisions - trajectory is safe!")

# Find which Excel row corresponds to test sample 0
print("\n=== FINDING EXCEL ROW FOR TEST SAMPLE 0 ===")
orig_idx, excel_row = excel_row_from_test_idx(0)
print(f"\nTest sample 0 is Excel row {excel_row}")