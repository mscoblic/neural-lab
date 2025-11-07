import numpy as np
import plotly.graph_objects as go
import time

from common.bernstein import build_control_points, eval_piecewise_curve


def plot_sample_interactive_from_input(
    model,
    X_input_denorm,
    X_mean,
    X_std,
    Y_mean,
    Y_std,
    *,
    save_path=None,
    title=None,
    show=False
):
    """
    Direct equivalent of your original interactive plot function.
    Arguments:
        X_input_denorm : (1,4,3) ndarray, *denormalized* input tokens
        model          : torch model
        X_mean, X_std  : normalization stats
        Y_mean, Y_std  : normalization stats
    """

    t0 = time.perf_counter()
    print("\n[plot] ===============================================")
    print("[plot] Starting interactive plot generation…")

    # ------------------------------------------------------
    # Normalize input (exactly as before)
    print("[plot] Normalizing input…")
    X_norm = (X_input_denorm - X_mean) / X_std

    # Convert to tensor inside caller, so here we assume caller already did:
    # test_input_tensor = torch.from_numpy(X_norm).to(device)
    # To match your original code, we accept numpy here and expect caller
    # to pass a tensor.
    # So X_norm_tensor is passed by caller, not created here.

    # caller will do:
    #   out_norm = model(X_norm_tensor)
    #   pred = out_norm.cpu().numpy() * Y_std + Y_mean

    # ------------------------------------------------------
    # Model inference
    print("[plot] Running model inference…")
    device = next(model.parameters()).device
    import torch
    X_norm_tensor = torch.from_numpy(X_norm).to(device)

    with torch.no_grad():
        out_norm = model(X_norm_tensor)
    print("[plot] Model output (normalized) shape:", out_norm.shape)

    # Denormalize
    pred = out_norm.cpu().numpy() * Y_std + Y_mean
    print("[plot] Output denormalized.")

    # ------------------------------------------------------
    # Extract tokens
    print("[plot] Extracting tokens (start, end, obstacle, control)…")
    start = X_input_denorm[0, 0]
    end   = X_input_denorm[0, 1]
    obs   = X_input_denorm[0, 2]
    head  = X_input_denorm[0, 3]

    # predicted control points
    preds = pred[0]  # (T_out,3)

    # ------------------------------------------------------
    # Build Bernstein trajectory (exact logic preserved)
    print("[plot] Building Bernstein trajectory…")
    cx, cy, cz = build_control_points(start, end, head, preds)
    tx, ty, tz = eval_piecewise_curve(cx, cy, cz, n_eval=50)

    print("[plot] Trajectory computed.")

    # ------------------------------------------------------
    # Create Plotly figure (same style)
    print("[plot] Creating figure and adding traces…")
    fig = go.Figure()

    # trajectory
    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Trajectory'
    ))

    # predicted CPs
    fig.add_trace(go.Scatter3d(
        x=preds[:,0], y=preds[:,1], z=preds[:,2],
        mode='markers+text',
        marker=dict(size=6, color='green'),
        text=[str(i+1) for i in range(len(preds))],
        textposition='top center',
        name='Predictions'
    ))

    # start/end
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode='markers',
        marker=dict(size=10, color=['green','orange'], symbol='diamond'),
        name='Start/End'
    ))

    # heading control
    fig.add_trace(go.Scatter3d(
        x=[head[0]], y=[head[1]], z=[head[2]],
        mode='markers',
        marker=dict(size=8, color='purple'),
        name='Initial Velocity'
    ))

    # obstacle sphere
    print("[plot] Adding obstacle sphere…")
    r = 0.1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = obs[0] + r*np.cos(u)*np.sin(v)
    ys = obs[1] + r*np.sin(u)*np.sin(v)
    zs = obs[2] + r*np.cos(v)

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
    dt = time.perf_counter() - t0
    print(f"[plot] Done in {dt:.3f}s")
    print("[plot] ===============================================\n")

    if show:
        print("[plot] Displaying figure in browser…")
        fig.show()

    return fig
