import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- project modules ---
from common.config import TrainConfig
from common.data import (
    build_XY_from_excel,
    split_indices,
    compute_norm_stats,
    apply_norm,
    TrajDataset
)
from common.metrics import count_collisions

from models.transformer.encoder import TransformerEncoder
from models.transformer.head import FlattenMLPHead, TrajModel


def main():
    cfg = TrainConfig()

    # ------------------------------------------------------------
    # 1. Load Excel
    # ------------------------------------------------------------
    df = __import__("pandas").read_excel(cfg.excel_path)
    X_raw, Y_raw, T_out = build_XY_from_excel(df, cfg.output_cols)

    N = len(df)

    # ------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------
    idx_train, idx_test = split_indices(N, frac=0.8, seed=cfg.seed)

    X_train_raw = X_raw[idx_train]
    Y_train_raw = Y_raw[idx_train]

    # ------------------------------------------------------------
    # 3. Compute normalization stats (train only)
    # ------------------------------------------------------------
    X_mean, X_std, Y_mean, Y_std = compute_norm_stats(X_train_raw, Y_train_raw)

    # normalize everything
    X_norm = apply_norm(X_raw, X_mean, X_std)
    Y_norm = apply_norm(Y_raw, Y_mean, Y_std)

    # datasets
    train_ds = TrajDataset(X_norm[idx_train], Y_norm[idx_train])
    test_ds  = TrajDataset(X_norm[idx_test],  Y_norm[idx_test])

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    # ------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = TransformerEncoder(
        input_dim=cfg.input_dim,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.seq_len,
        dropout=cfg.dropout,
        output_dim=None       # encoder returns embeddings only
    )

    head = FlattenMLPHead(
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        hidden=cfg.head_hidden,
        t_out=T_out,
        out_dim=cfg.output_dim
    )

    model = TrajModel(encoder, head).to(device)

    # ------------------------------------------------------------
    # 5. Loss + optimizer + LR scheduler
    # ------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfg.lr_patience,
    )

    # ------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------
    best_test_mse = float("inf")
    patience_counter = 0

    print("Starting training…")

    for epoch in range(1, cfg.epochs + 1):

        # ------------------------
        # Training step
        # ------------------------
        model.train()
        running = 0.0

        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)

            optimizer.zero_grad()
            Y_hat = model(Xb)
            loss = criterion(Y_hat, Yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * Xb.size(0)

        train_mse = running / len(train_dl.dataset)

        # ------------------------
        # Evaluation step
        # ------------------------
        model.eval()
        running = 0.0
        with torch.no_grad():
            for Xb, Yb in test_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, Yb)
                running += loss.item() * Xb.size(0)

        test_mse = running / len(test_dl.dataset)

        # ------------------------
        # Collision rate
        # ------------------------
        coll_rate = count_collisions(
            model, test_dl,
            X_mean, X_std, Y_mean, Y_std,
            radius=cfg.obstacle_radius
        )

        # ------------------------
        # Learning rate schedule
        # ------------------------
        scheduler.step(test_mse)
        lr_now = optimizer.param_groups[0]["lr"]

        # ------------------------
        # Early stopping
        # ------------------------
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            patience_counter = 0

            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_transformer.pth")
        else:
            patience_counter += 1

        # ------------------------
        # Logging
        # ------------------------
        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_mse:.6f} | "
            f"Test MSE: {test_mse:.6f} | "
            f"Coll: {coll_rate:.3%} | "
            f"LR: {lr_now:.6f}"
        )

        if patience_counter >= cfg.early_stop_patience:
            print("Early stopping.")
            break

    print("\nTraining complete.")
    print(f"Best test MSE: {best_test_mse:.6f}")


if __name__ == "__main__":
    main()
