import numpy as np
import torch
from common.data import apply_denorm
from common.bernstein import build_control_points, eval_piecewise_curve


@torch.no_grad()
def count_collisions(model, loader, X_mean, X_std, Y_mean, Y_std, radius=0.1):
    """
    Discrete collision check:
    Checks only the predicted control points, not the full Bernstein curve.
    """
    model.eval()
    device = next(model.parameters()).device

    total = 0
    collided = 0

    for X, _ in loader:
        X = X.to(device)
        Yp = model(X)  # normalized predictions

        # denorm predictions
        Yp_den = apply_denorm(Yp.cpu().numpy(), Y_mean, Y_std)

        # denorm obstacle centers
        X_den = apply_denorm(X.cpu().numpy(), X_mean, X_std)
        obstacles = X_den[:, 2, :]  # (B, 3)

        # distances
        for i in range(len(Yp_den)):
            obs = obstacles[i]
            preds = Yp_den[i]  # (T_out,3)
            d = np.linalg.norm(preds - obs, axis=1)
            if np.any(d < radius):
                collided += 1
            total += 1

    return collided / max(total, 1)


@torch.no_grad()
def count_collisions_continuous(
    model, loader, X_mean, X_std, Y_mean, Y_std,
    radius=0.1, n_eval=200
):
    """
    Continuous collision check:
    Uses Bernstein curve evaluation (full trajectory).
    """
    model.eval()
    device = next(model.parameters()).device

    total = 0
    collided = 0

    for X, _ in loader:
        X = X.to(device)
        Yp = model(X)

        # denorm both X and Y
        X_den = apply_denorm(X.cpu().numpy(), X_mean, X_std)
        Yp_den = apply_denorm(Yp.cpu().numpy(), Y_mean, Y_std)

        for i in range(len(X_den)):
            # tokens
            start = X_den[i, 0]
            end = X_den[i, 1]
            obstacle = X_den[i, 2]
            heading = X_den[i, 3]

            preds = Yp_den[i]  # (T_out,3)

            # build control points
            cx, cy, cz = build_control_points(start, end, heading, preds)

            # evaluate full Bernstein trajectory
            tx, ty, tz = eval_piecewise_curve(cx, cy, cz, n_eval=n_eval)
            traj = np.column_stack([tx, ty, tz])

            # distances to obstacle center
            d = np.linalg.norm(traj - obstacle, axis=1)

            if np.any(d < radius):
                collided += 1
            total += 1

    return collided / max(total, 1)
