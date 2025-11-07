import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import pandas as pd

from common.config import TrainConfig
from common.data import (
    build_XY_from_excel,
    split_indices,
    compute_norm_stats,
    apply_norm,
    TrajDataset
)
from common.metrics import count_collisions
from common.viz_matplotlib import plot_dataset_sample
from common.viz_plotly import plot_sample_interactive_from_input

from models.transformer.encoder import TransformerEncoder
from models.transformer.head import FlattenMLPHead, TrajModel


def load_model(cfg, T_out, device):
    """Build the model and load saved weights."""
    encoder = TransformerEncoder(
        input_dim=cfg.input_dim,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.seq_len,
        dropout=cfg.dropout,
        output_dim=None
    )

    head = FlattenMLPHead(
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        hidden=cfg.head_hidden,
        t_out=T_out,
        out_dim=cfg.output_dim
    )

    model = TrajModel(encoder, head).to(device)

    print(f"Loading model from {cfg.model_path}...")
    state = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")

    return model


def evaluate(model, test_dl, criterion, device):
    """Compute test MSE."""
    running = 0.0
    for Xb, Yb in test_dl:
        Xb, Yb = Xb.to(device), Yb.to(device)
        with torch.no_grad():
            pred = model(Xb)
            loss = criterion(pred, Yb)
        running += loss.item() * Xb.size(0)
    return running / len(test_dl.dataset)


def main():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 1. Load Excel
    # ------------------------------------------------------------
    df = pd.read_excel(cfg.excel_path)
    X_raw, Y_raw, T_out = build_XY_from_excel(df, cfg.output_cols)
    N = len(df)

    # ------------------------------------------------------------
    # 2. Split indices (same seed ensures consistent split)
    # ------------------------------------------------------------
    idx_train, idx_test = split_indices(N, frac=0.8, seed=cfg.seed)

    X_train_raw = X_raw[idx_train]
    Y_train_raw = Y_raw[idx_train]

    # ------------------------------------------------------------
    # 3. Compute normalization stats (train only)
    # ------------------------------------------------------------
    X_mean, X_std, Y_mean, Y_std = compute_norm_stats(X_train_raw, Y_train_raw)

    # normalize full data
    X_norm = apply_norm(X_raw, X_mean, X_std)
    Y_norm = apply_norm(Y_raw, Y_mean, Y_std)

    test_ds = TrajDataset(X_norm[idx_test], Y_norm[idx_test])
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # ------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------
    model = load_model(cfg, T_out, device)
    criterion = torch.nn.MSELoss()

    # ------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------
    print("\n=== Evaluation ===")
    mse = evaluate(model, test_dl, criterion, device)
    print(f"Test MSE: {mse:.6f}")

    # collision rate
    coll = count_collisions(
        model,
        test_dl,
        X_mean,
        X_std,
        Y_mean,
        Y_std,
        radius=cfg.obstacle_radius,
    )
    print(f"Collision rate: {coll * 100:.2f}%")

    # ------------------------------------------------------------
    # 6. Optional plotting
    # ------------------------------------------------------------
    print("\nPlotting a few sample predictions...")

    # pick samples from test set
    sample_indices = [0, 1, 2]

    for i in sample_indices:
        Xn, Yn = test_ds[i]  # normalized tensors
        X_den = Xn.numpy() * X_std + X_mean      # (4,3)
        # Model prediction
        X_tensor = Xn.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(X_tensor).cpu().numpy()

        pred_den = pred_norm * Y_std + Y_mean

        # Y_true
        Y_true_den = Yn.numpy() * Y_std + Y_mean

        plot_dataset_sample(
            X_den[0],
            Y_true_den[0],
            pred_den[0],
            idx=i,
            save_path=None,
            title_prefix="test"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
