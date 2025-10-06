import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

TRAIN = True


# ---------------- Plotting ----------------
def _to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


@torch.no_grad()
def plot_dataset_sample(model, ds, idx, save_path=None, title_prefix="val"):
    """
    Plot a single dataset item with:
      - predicted trajectory (o markers)
      - ground-truth trajectory (x markers)
      - start/end markers
      - obstacle circle
    """
    model.eval()
    device = next(model.parameters()).device

    X, Y_true = ds[idx]  # X: (T_in,2), Y_true: (T_out,2)
    Y_pred = model(X.unsqueeze(0).to(device))[0].cpu()  # (T_out,2)

    # recover inputs
    xin, yin = X[0].tolist()
    xout, yout = X[1].tolist()
    x1, y1 = X[2].tolist()
    ox, oy = X[3].tolist()

    fig, ax = plt.subplots(figsize=(6, 6))

    # start/end
    ax.scatter([xin], [yin], marker='o', s=80, label='start', zorder=2)
    ax.scatter([xout], [yout], marker='*', s=120, label='end', zorder=2)

    # ground truth
    Yt = _to_np(Y_true)
    ax.scatter(Yt[:, 0], Yt[:, 1], marker='x', s=36, color='black', label="ground truth", zorder=3)

    # prediction
    Yp = _to_np(Y_pred)
    ax.scatter(Yp[:, 0], Yp[:, 1], marker='o', s=28, label="prediction", zorder=3)
    for i, (x, y) in enumerate(Yp):
        ax.text(float(x), float(y), str(i + 1), fontsize=8, ha='left', va='bottom', color='blue')

    # obstacle
    circle = patches.Circle((ox, oy), radius=0.1, facecolor='red', edgecolor='black',
                            alpha=0.6, label=f'obstacle R={0.1}')
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{title_prefix} sample idx={idx}")

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def plot_many_samples(model, ds, indices, title_prefix="val"):
    for j, idx in enumerate(indices):
        plot_dataset_sample(model, ds, idx, title_prefix=title_prefix)


# ---------------- Transformer building blocks ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def split_heads(self, x):
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, _, L, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))
        out = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(out))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention (with causal mask)
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(attn_out))

        # Cross-attention to encoder output
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.drop(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.drop(ff_out))
        return x


# ---------------- Encoder-Decoder Transformer ----------------
class EncoderDecoderTransformer(nn.Module):
    """
    Proper encoder-decoder transformer for trajectory prediction.
    The decoder autoregressively generates trajectory points.
    """

    def __init__(self, input_dim, output_dim, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, d_ff,
                 max_seq_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # Encoder
        self.encoder_input_proj = nn.Linear(input_dim, d_model)
        self.encoder_pos_enc = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder_input_proj = nn.Linear(output_dim, d_model)
        self.decoder_pos_enc = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, size, device):
        """Generate causal mask for decoder self-attention."""
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    def encode(self, src):
        """Encode the input sequence."""
        # src: (B, T_in, input_dim)
        x = self.encoder_input_proj(src)
        x = self.encoder_pos_enc(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask=None)

        return self.encoder_norm(x)  # (B, T_in, d_model)

    def decode(self, tgt, enc_output, tgt_mask=None):
        """Decode with cross-attention to encoder output."""
        # tgt: (B, T_out, output_dim)
        x = self.decoder_input_proj(tgt)
        x = self.decoder_pos_enc(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask=None, tgt_mask=tgt_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)  # (B, T_out, output_dim)

    def forward(self, src, tgt=None):
        """
        Forward pass.
        src: (B, T_in, input_dim) - encoder input
        tgt: (B, T_out, output_dim) - decoder input (for teacher forcing during training)

        During training: use teacher forcing with ground truth
        During inference: autoregressive generation
        """
        B = src.size(0)
        device = src.device

        # Encode
        enc_output = self.encode(src)  # (B, T_in, d_model)

        if tgt is not None:
            # Training mode: teacher forcing
            T_out = tgt.size(1)
            tgt_mask = self.generate_causal_mask(T_out, device)
            output = self.decode(tgt, enc_output, tgt_mask)
            return output
        else:
            # Inference mode: autoregressive generation
            # This will be used during evaluation
            return self.autoregressive_decode(enc_output)

    def autoregressive_decode(self, enc_output, max_len=7):
        """
        Autoregressive decoding for inference.
        Start with zeros and generate one point at a time.
        """
        B = enc_output.size(0)
        device = enc_output.device

        # Start with a zero vector (or you could use a learnable start token)
        decoder_input = torch.zeros(B, 1, self.output_dim, device=device)

        outputs = []
        for i in range(max_len):
            # Generate causal mask for current length
            tgt_mask = self.generate_causal_mask(i + 1, device)

            # Decode current sequence
            out = self.decode(decoder_input, enc_output, tgt_mask)

            # Take the last prediction
            next_point = out[:, -1:, :]  # (B, 1, output_dim)
            outputs.append(next_point)

            # Append to decoder input for next iteration
            decoder_input = torch.cat([decoder_input, next_point], dim=1)

        return torch.cat(outputs, dim=1)  # (B, max_len, output_dim)


# ---------------- Dataset ----------------
class TrajDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i): return self.X[i], self.Y[i]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seq lengths
T_in = 5
T_out = 7

file_path = "../data/collision_heading_bpoly_gage.xlsx"
df = pd.read_excel(file_path)

# outputs
output_cols = ["x2", "x3", "x4", "x5", "x7", "x8", "x9",
               "y2", "y3", "y4", "y5", "y7", "y8", "y9"]
Y_np = df[output_cols].to_numpy(dtype=np.float32)
N = Y_np.shape[0]
Y_np = Y_np.reshape(N, 2, T_out).transpose(0, 2, 1)  # (N, T_out, 2)

# inputs (5 tokens × 2 features)
starts = df[["xin", "yin"]].to_numpy(dtype=np.float32)
ends = df[["xout", "yout"]].to_numpy(dtype=np.float32)
ctrls = df[["x1", "y1"]].to_numpy(dtype=np.float32)
obstacles = df[["ox", "oy"]].to_numpy(dtype=np.float32)

# scalar flag (left/right indicator)
flag = df["LR"].to_numpy(dtype=np.float32)[:, None]
flag = np.concatenate([flag, np.zeros_like(flag)], axis=1)

# stack into tokens → (N, 5, 2)
X_np = np.stack([starts, ends, ctrls, obstacles, flag], axis=1)

# ---------------- Hyperparams ----------------
input_dim = 2
output_dim = 2
d_model = 128
num_heads = 4
num_encoder_layers = 3
num_decoder_layers = 3
d_ff = 256
dropout = 0.1
max_seq_length = 10  # Max of T_in and T_out

# ---------------- Data split ----------------
seed = 42
rng = np.random.default_rng(seed)
perm = rng.permutation(N)
n_train = int(0.80 * N)
idx_train = perm[:n_train]
idx_test = perm[n_train:]

train_ds = TrajDataset(X_np[idx_train], Y_np[idx_train])
test_ds = TrajDataset(X_np[idx_test], Y_np[idx_test])
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

# ---------------- Model ----------------
model = EncoderDecoderTransformer(
    input_dim=input_dim,
    output_dim=output_dim,
    d_model=d_model,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=dropout
).to(device)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=5)


def train_one_epoch():
    model.train()
    running_traj = 0.0
    running_obs = 0.0

    for X, Y in train_dl:
        X, Y = X.to(device), Y.to(device)

        # Prepare decoder input for teacher forcing
        # Prepend zeros as start token, use Y[:-1] as input
        decoder_input = torch.cat([
            torch.zeros(Y.size(0), 1, Y.size(2), device=device),
            Y[:, :-1, :]
        ], dim=1)  # (B, T_out, 2)

        optimizer.zero_grad()

        # Forward with teacher forcing
        Y_hat = model(X, decoder_input)  # (B, T_out, 2)

        # Trajectory loss
        loss_traj = criterion(Y_hat, Y)

        # Obstacle avoidance loss
        obstacle_pos = X[:, 3, :]

        # Combined loss
        loss = loss_traj + 5.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_traj += loss_traj.item() * X.size(0)

    return running_traj / len(train_dl.dataset), running_obs / len(train_dl.dataset)


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    running = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        # Use autoregressive decoding during evaluation
        Y_hat = model(X, tgt=None)
        loss = criterion(Y_hat, Y)
        running += loss.item() * X.size(0)
    return running / len(loader.dataset)


# ---------------- Training ----------------
EPOCHS = 50
train_losses = []
test_losses = []
obs_losses = []

if TRAIN:
    best_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(EPOCHS):
        tr_loss, tr_obs = train_one_epoch()
        train_losses.append(tr_loss)
        obs_losses.append(tr_obs)

        # Evaluate every epoch
        te_loss = eval_epoch(test_dl)
        test_losses.append(te_loss)
        scheduler.step(te_loss)

        print(f"Epoch {epoch + 1:02d} | Train MSE: {tr_loss:.6f} | "
              f"Obs Penalty: {tr_obs:.6f} | Test MSE: {te_loss:.6f}")

        # Save best model
        if te_loss < best_loss:
            best_loss = te_loss
            patience_counter = 0
            torch.save(model.state_dict(), "transformer_encdec_best.pth")
            print(f"  → New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(torch.load("transformer_encdec_best.pth", map_location=device))
    final_test = eval_epoch(test_dl)
    print(f"\nBEST TEST MSE: {final_test:.6f}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train MSE')
    ax1.plot(range(1, len(test_losses) + 1), test_losses, marker='s', label='Test MSE')
    ax1.axhline(final_test, linestyle='--', alpha=0.5, label=f'Best = {final_test:.6f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Training & Test Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(range(1, len(obs_losses) + 1), obs_losses, marker='o', color='red', label='Obstacle Penalty')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Penalty')
    ax2.set_title('Obstacle Avoidance Penalty')
    ax2.grid(True)
    ax2.legend()

    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/encdec_loss_curve.png", dpi=160, bbox_inches="tight")
    plt.close()
else:
    model.load_state_dict(torch.load("transformer_encdec_best.pth", map_location=device))
    model.eval()

# ---------------- Plotting examples ----------------
print("\nGenerating sample predictions...")
plot_many_samples(model, test_ds, indices=[0, 1, 2], title_prefix="val")
plot_dataset_sample(model, test_ds, idx=5, save_path="figs/val_sample_005.png")
print("Done!")