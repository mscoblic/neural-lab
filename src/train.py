"""
File: train.py
Author: Mia Scoblic
Date: 2025-08-15
Description:
This module handles the full training loop. It loads configs, prepares datasets,
builds the model, and then runs training, validation, and testing. It also takes
care of logging results, saving checkpoints, and making runs reproducible.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.config import load_yaml
from src.data import DataSchema, TrajectoryDataset, prepare_datasets
from src.models import build_model_from_config
from src.utils import set_seed, get_device, save_json, CSVLogger, default_run_name
import json
import torch
import yaml
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np  # NEW
# from torch.optim.lr_scheduler import ExponentialLR   # (unused)
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# -----------------------------
# New small helper builders
# -----------------------------
def build_loss(loss_name: str = "mse", huber_beta: float = 0.5) -> nn.Module:
    """
    Build a loss function. Defaults to MSE. Huber is SmoothL1 in PyTorch.
    """
    name = (loss_name or "mse").lower()
    if name == "mse":
        return nn.MSELoss()
    if name in ("huber", "smoothl1"):
        return nn.SmoothL1Loss(beta=float(huber_beta))
    if name in ("l1", "mae"):
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {loss_name}")

def build_optimizer(model: nn.Module, opt_name: str, lr: float, weight_decay: float):
    """
    Build optimizer. Defaults to AdamW for better generalization.
    """
    name = (opt_name or "adamw").lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {opt_name}")


@torch.no_grad()
def count_collisions_continuous(model, loader, device, x_norm, y_norm, K: int, T: int, radius: float = 0.1,
                                n_eval: int = 100):
    """
    Count collisions by evaluating continuous Bernstein polynomial trajectories

    Args:
        model: trained model
        loader: DataLoader for the dataset
        device: computation device
        x_norm: input normalizer
        y_norm: output normalizer
        K: number of dimensions (should be 3)
        T: number of output control points (should be 7)
        radius: obstacle radius
        n_eval: number of points to evaluate along trajectory

    Returns:
        float: collision rate (0.0 to 1.0)
    """
    import sys
    from pathlib import Path

    # Import BeBOT
    tools_dir = Path(__file__).resolve().parent.parent / 'tools' / 'extra'
    if str(tools_dir) not in sys.path:
        sys.path.append(str(tools_dir))
    from BeBOT import PiecewiseBernsteinPoly

    model.eval()

    total = 0
    collided = 0

    for x, _ in loader:
        x = x.to(device)

        # Get predictions and denormalize
        with torch.no_grad():
            y_pred_norm = model(x)
        y_pred = y_norm.inverse(y_pred_norm).cpu()

        # Denormalize inputs
        x_denorm = x_norm.inverse(x).cpu()

        # Process each trajectory in batch
        for i in range(x.size(0)):
            # Extract start/end from inputs
            sx, sy, sz = x_denorm[i, 0].item(), x_denorm[i, 1].item(), x_denorm[i, 2].item()
            ex, ey, ez = x_denorm[i, 3].item(), x_denorm[i, 4].item(), x_denorm[i, 5].item()

            # Extract obstacle center and radius
            ox, oy, oz = x_denorm[i, 6].item(), x_denorm[i, 7].item(), x_denorm[i, 8].item()
            r = 0.1

            # Get predicted control points
            pred_3xT = y_pred[i].view(K, T)  # (3, 7)
            pred_points = pred_3xT.T.numpy()  # (7, 3)

            # Build Bernstein polynomial trajectory
            cp2, cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

            control_points_x = np.array([sx, cp2[0], cp3[0], cp4[0], cp5[0],
                                         cp5[0], cp6[0], cp7[0], cp8[0], ex])
            control_points_y = np.array([sy, cp2[1], cp3[1], cp4[1], cp5[1],
                                         cp5[1], cp6[1], cp7[1], cp8[1], ey])
            control_points_z = np.array([sz, cp2[2], cp3[2], cp4[2], cp5[2],
                                         cp5[2], cp6[2], cp7[2], cp8[2], ez])

            # Evaluate continuous trajectory
            tknots = np.array([0, 0.5, 1.0])
            t_eval = np.linspace(0, 1, n_eval)

            traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
            traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
            traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

            # Stack into (n_eval, 3)
            trajectory = np.column_stack([traj_x, traj_y, traj_z])

            # Check collision
            obstacle_center = np.array([ox, oy, oz])
            distances = np.linalg.norm(trajectory - obstacle_center, axis=1)

            if np.any(distances < r):
                collided += 1

            total += 1

    return collided / max(total, 1)

def build_scheduler(optimizer, sched_cfg: dict | None, total_epochs: int):
    """
    Epoch-wise scheduler. Cosine is a safe default. None disables scheduling.
    """
    if not sched_cfg:
        return None
    name = (sched_cfg.get("name") or "none").lower()
    if name in ("none", "off"):
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_clip: float = 0.0) -> float:
    """
    Run one full training epoch.

    Args:
        model (torch.nn.Module): The model being trained
        loader (DataLoader): DataLoader for the training dataset
        loss_fn (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        device (torch.device): Device to run computations on (CPU/GPU)
        grad_clip (float): Global-norm gradient clip (0.0 disables)

    Returns:
        float: The mean training loss across the entire dataset for this epoch.
    """
    model.train()                       # switch model to training mode
    total_loss = 0.0
    for x, y in loader:                 # iterate over all batches
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device, K: int):
    model.eval()
    total = 0.0
    axis_totals = [0.0 for _ in range(K)]

    n = 0  # robust to empty loaders
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        B, D = y.shape
        assert D % K == 0, f"Output dim ({D}) must be divisible by K={K}"
        chunk = D // K

        for k in range(K):
            s, e = k * chunk, (k + 1) * chunk
            axis_totals[k] += loss_fn(y_pred[:, s:e], y[:, s:e]).item() * B

        total += loss.item() * B
        n += B

    if n == 0:
        # No validation/test samples; return NaNs so callers can skip early stop on NaN
        return float("nan"), *[float("nan") for _ in range(K)]
    return (total / n, *[t / n for t in axis_totals])


# MODIFIED: Plotting function
def plot_training_curves(train_losses, val_losses, save_path, best_epoch=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training MSE values
        val_losses: List of validation MSE values
        save_path: Path to save the plot
        best_epoch: Epoch number with best validation loss (for marking)
    """
    plt.figure(figsize=(6, 4))
    xs = np.arange(1, len(val_losses) + 1)

    plt.plot(xs, val_losses, linestyle='-', linewidth=2, label='Val MSE')
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, linestyle='-', linewidth=2, label='Train MSE')

    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE over epochs')
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved training curve to {save_path}")


def main():

    # ------------------------------------------------------------------------------------------------------------------
    # Defines are parses flags for the command line
    # ------------------------------------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True,
                    help="Which experiment to run (e.g., --task <name of specific configs folder>)")
    ap.add_argument("--split-from", type=str, default="",
                    help="Rerun experiment with split from previous run (e.g., --split-from <name of specific configs folder>)")
    args = ap.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # File path initialization
    # ------------------------------------------------------------------------------------------------------------------
    root = Path(__file__).resolve().parent.parent
    task_dir = root / "configs" / args.task
    data_cfg_path  = (task_dir / f"data_{args.task}.yaml").resolve()
    model_cfg_path = (task_dir / f"model_{args.task}.yaml").resolve()
    run_cfg_path   = (task_dir / f"run_{args.task}.yaml").resolve()

    # ------------------------------------------------------------------------------------------------------------------
    # Load YAML config files
    # ------------------------------------------------------------------------------------------------------------------
    run_cfg = load_yaml(run_cfg_path)
    data_cfg = load_yaml(data_cfg_path)

    device = get_device(run_cfg.get("device"))
    val_split  = float(run_cfg.get("val_split"))
    test_ratio = float(run_cfg.get("test_ratio"))
    seed = int(run_cfg.get("seed"))
    set_seed(seed)

    reuse_split_from = args.split_from.strip()
    split_indices = None            # generate later if not defined
    if reuse_split_from:
        split_path = (root / "runs" / reuse_split_from / "split_indices.json").resolve()
        with split_path.open("r") as f:
            split_indices = json.load(f)

    # ----------------------------------------------------------------------------------------------------------------------
    # Schema & raw dataset
    # ----------------------------------------------------------------------------------------------------------------------
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[list(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K")),
    )

    excel_path = (root / data_cfg["excel_path"]).resolve()

    # instance of TrajectoryDataset in data.py, contains input/output tensors
    raw = TrajectoryDataset(str(excel_path), schema, task_name=data_cfg["task_name"])

    # ----------------------------------------------------------------------------------------------------------------------
    # Prepare normalized splits & dataloaders
    # ----------------------------------------------------------------------------------------------------------------------
    train_ds, val_ds, test_ds, norm_state = prepare_datasets(
        raw,
        norm_inputs=data_cfg["normalization"]["inputs"],
        norm_outputs=data_cfg["normalization"]["outputs"],
        val_ratio=val_split,
        test_ratio=test_ratio,
        seed=seed,
        split_indices=split_indices,  # if none, generate new; otherwise reuse exact indices
    )

    # Prefer K from norm_state (populated from raw.K). Fallback to shape if missing.
    K = int(norm_state.get("K", train_ds.output_dim // train_ds.T))
    print(f"Number of Dimensions: {K}")

    batch_size  = int(run_cfg.get("batch_size", 64))
    num_workers = int(run_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False) if test_ds is not None else None

    # ----------------------------------------------------------------------------------------------------------------------
    # Meta for logging/model init
    # ----------------------------------------------------------------------------------------------------------------------
    meta = {
        "task_name": data_cfg.get("task_name", "task"),
        "T": raw.T,
        "input_dim": raw.input_dim,
        "output_dim": raw.output_dim,
        "excel_path": str(excel_path),
    }

    # ---- Model ----
    model_cfg = load_yaml(model_cfg_path)
    model = build_model_from_config(model_cfg, input_dim=meta["input_dim"], output_dim=meta["output_dim"]).to(device)
    print(f"[MODEL] {model_cfg.get('kind', 'ffn')} on {device}")

    # ---- Optimizer & loss ----
    lr = float(run_cfg.get("lr", 1e-3))
    opt_name = str(run_cfg.get("optimizer", "adamw"))
    weight_decay = float(run_cfg.get("weight_decay", 0.0001))  # modest default
    grad_clip = float(run_cfg.get("grad_clip", 0.0))

    # loss config (default MSE; Huber helps with noisy/multimodal labels)
    loss_name = str(run_cfg.get("loss", "mse"))
    huber_beta = float(run_cfg.get("huber_beta", 0.5))

    optimizer = build_optimizer(model, opt_name, lr, weight_decay)
    loss_fn = build_loss(loss_name, huber_beta)

    # ---- Scheduler (epoch-wise) ----
    sched_cfg = run_cfg.get("scheduler", {"name": "none"}) or {"name": "none"}
    scheduler = build_scheduler(optimizer, sched_cfg, total_epochs=int(run_cfg.get("epochs", 100)))

    # ---- Logging / output dir ----
    out_dir = Path(run_cfg.get("logging", {}).get("out_dir", "runs"))
    run_name = run_cfg.get("logging", {}).get("run_name") or default_run_name(
        meta["task_name"],
        model_cfg.get("kind", "ffn"),
        seed,
        data_cfg_path,
        model_cfg_path,
        run_cfg_path,
    )
    run_dir = (root / out_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load full YAML contents
    with open(data_cfg_path, "r") as f:
        data_cfg_full = yaml.safe_load(f)
    with open(model_cfg_path, "r") as f:
        model_cfg_full = yaml.safe_load(f)
    with open(run_cfg_path, "r") as f:
        run_cfg_full = yaml.safe_load(f)

    # Save configs (contents, not just paths)
    save_json(
        {
            "data_config": data_cfg_full,
            "model_config": model_cfg_full,
            "run_config": run_cfg_full,
        },
        run_dir / "configs_used.json",
    )

    # Save norm stats and metadata as before
    save_json(norm_state, run_dir / "norm.json")
    save_json({"meta": meta}, run_dir / "meta.json")

    # Save split indices if we generated fresh ones this run
    if not reuse_split_from:
        def _extract_indices(view):
            # view is a NormalizedDataset; its .base is _SubsetView with .indices
            return list(view.base.indices)
        splits = {"train": _extract_indices(train_ds), "val": _extract_indices(val_ds)}
        if test_ds is not None:
            splits["test"] = _extract_indices(test_ds)
        with (run_dir / "split_indices.json").open("w") as f:
            json.dump(splits, f, indent=2)

    # Headers for metrics CSV (depends on K)
    if K == 2:
        headers = ["epoch", "train_mse", "val_mse", "val_mse_x", "val_mse_y"]
    elif K == 3:
        headers = ["epoch", "train_mse", "val_mse", "val_mse_x", "val_mse_y", "val_mse_z"]
    else:
        headers = ["epoch", "train_mse", "val_mse"] + [f"val_mse_axis{i + 1}" for i in range(K)]

    csv_logger = CSVLogger(run_dir / "metrics.csv", header=headers)

    # ---- Training loop with early stopping ----
    epochs   = int(run_cfg.get("epochs", 100))
    es_cfg   = run_cfg.get("early_stopping", {}) or {}
    use_es   = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 10))

    best_val = float("inf")
    best_epoch = -1
    epochs_since = 0
    best_ckpt = run_dir / "model_best.pt"

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=grad_clip)
        val_mse, *axis_mse = eval_one_epoch(model, val_loader, loss_fn, device, K)

        # Store losses
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        csv_logger.log([epoch, train_mse, val_mse, *axis_mse])

        if K == 2:
            print(f"[E{epoch:03d}] train MSE={train_mse:.6f} | val MSE={val_mse:.6f} "
                  f"(x={axis_mse[0]:.6f}, y={axis_mse[1]:.6f})")
        elif K == 3:
            print(f"[E{epoch:03d}] train MSE={train_mse:.6f} | val MSE={val_mse:.6f} "
                  f"(x={axis_mse[0]:.6f}, y={axis_mse[1]:.6f}, z={axis_mse[2]:.6f})")
        else:
            parts = ", ".join(f"a{i + 1}={v:.6f}" for i, v in enumerate(axis_mse))
            print(f"[E{epoch:03d}] train MSE={train_mse:.6f} | val MSE={val_mse:.6f} ({parts})")

        # Save best by val MSE (skip NaN)
        improved = (val_mse == val_mse) and (val_mse < best_val - 1e-8)  # NaN-safe check
        if improved:
            best_val = val_mse
            best_epoch = epoch
            epochs_since = 0
            torch.save({"state_dict": model.state_dict()}, best_ckpt)
        else:
            epochs_since += 1

        # Step epoch-wise scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()

        if use_es and (val_mse == val_mse) and epochs_since >= patience:
            print(f"[EarlyStopping] No improvement for {patience} epochs. Best val MSE={best_val:.6f} at epoch {best_epoch}.")
            break
            print("\a")
    print("\a")

    csv_logger.close()

    # Generate and save plot
    plot_training_curves(train_losses, val_losses, run_dir / "training_curve.png", best_epoch=best_epoch)

    # ---- Final test evaluation on held-out set ----
    # ---- Final test evaluation on held-out set ----
    if test_loader is not None:
        best = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best["state_dict"])
        test_mse, *axis_mse = eval_one_epoch(model, test_loader, loss_fn, device, K)

        # Calculate collision rate
        print("\n[INFO] Calculating collision rate...")

        # Import BeBOT at the top level if not already done
        import sys
        tools_dir = Path(__file__).resolve().parent.parent / 'tools' / 'extra'
        if str(tools_dir) not in sys.path:
            sys.path.append(str(tools_dir))
        from BeBOT import PiecewiseBernsteinPoly
        from src.data import Normalizer, NormStats

        # Reconstruct normalizers from norm_state
        x_state = norm_state["inputs"]
        y_state = norm_state["outputs"]

        x_normalizer = Normalizer(x_state["mode"])
        y_normalizer = Normalizer(y_state["mode"])

        # Restore stats
        if x_state["stats"] is not None:
            x_normalizer.fitted = True
            x_normalizer.stats = NormStats(
                mean=torch.tensor(x_state["stats"]["mean"], dtype=torch.float32),
                std=torch.tensor(x_state["stats"]["std"], dtype=torch.float32)
            )

        if y_state["stats"] is not None:
            y_normalizer.fitted = True
            y_normalizer.stats = NormStats(
                mean=torch.tensor(y_state["stats"]["mean"], dtype=torch.float32),
                std=torch.tensor(y_state["stats"]["std"], dtype=torch.float32)
            )

        test_collision_rate = count_collisions_continuous(
            model, test_loader, device,
            x_normalizer,
            y_normalizer,
            K, raw.T, radius=0.1, n_eval=100
        )

        if K == 2:
            payload = {"test_mse": test_mse, "test_mse_x": axis_mse[0], "test_mse_y": axis_mse[1],
                       "test_collision_rate": test_collision_rate}
            print(f"[TEST] MSE={test_mse:.6f} (x={axis_mse[0]:.6f}, y={axis_mse[1]:.6f})")
        elif K == 3:
            payload = {"test_mse": test_mse, "test_mse_x": axis_mse[0], "test_mse_y": axis_mse[1],
                       "test_mse_z": axis_mse[2], "test_collision_rate": test_collision_rate}
            print(f"Test MSE: {test_mse:.6f}")
        else:
            payload = {"test_mse": test_mse, "test_collision_rate": test_collision_rate} | {f"test_mse_axis{i + 1}": v
                                                                                            for i, v in
                                                                                            enumerate(axis_mse)}
            parts = ", ".join(f"a{i + 1}={v:.6f}" for i, v in enumerate(axis_mse))
            print(f"[TEST] MSE={test_mse:.6f} ({parts})")

        print(f"Test collision rate: {test_collision_rate * 100:.2f}%")

        save_json(payload, run_dir / "test_metrics.json")


if __name__ == "__main__":
    main()