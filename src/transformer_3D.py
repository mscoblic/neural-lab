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
project_root = Path(__file__).resolve().parent.parent  # Go up from src/ to neural-lab/
sys.path.append(str(project_root / 'tools' / 'extra'))
from BeBOT import PiecewiseBernsteinPoly

# Enable/disable training
TRAIN = True
DEBUG = False
TIME_EVAL = False

# Plotting
def _to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

# Plot and save a specific index
from mpl_toolkits.mplot3d import Axes3D  # keep this import (safe for all versions)

@torch.no_grad()
def plot_dataset_sample(model, ds, idx, save_path=None, title_prefix="test",
                        elev=45, azim=-70):
    model.eval()
    device = next(model.parameters()).device

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

    # --- 3D scatter plot ---
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)  # <<< set camera
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

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


# Visualize many samples quickly
@torch.no_grad()
def plot_many_samples(model, ds, indices, title_prefix="test"):
    """Convenience: plot several dataset samples by index."""
    for j, idx in enumerate(indices):
        plot_dataset_sample(model, ds, idx, title_prefix=title_prefix)



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
file_path = "../data/collision_3D_freevelocity.xlsx"
df = pd.read_excel(file_path)

# Build 4 semantic tokens, each 2-D: start, end, obstacle, control
start    = df[["x0","y0", "z0"]].to_numpy(np.float32)  # (N, 2)
end      = df[["xf","yf", "zf"]].to_numpy(np.float32)  # (N, 2)
obstacle = df[["ox","oy", "oz"]].to_numpy(np.float32)  # (N, 2)
control  = df[["x2","y2", "z2"]].to_numpy(np.float32)  # (N, 2)

output_cols = ["x3", "x4", "x5", "x6", "x7", "x8", "y3", "y4", "y5", "y6", "y7", "y8", "z3", "z4", "z5", "z6", "z7", "z8"]
T_out = len(output_cols) // 3
if DEBUG == True:
    print("Derived T_out =", T_out)  # expect 6

# Stack into tokens → (N, 4, 2)
X_np = np.stack([start, end, obstacle, control], axis=1)
Y_np = df[output_cols].to_numpy(dtype=np.float32)    # (N, 20)

# Targets (unchanged): (N,14) -> (N, 7, 2)
N = Y_np.shape[0]
Y_np = Y_np.reshape(N, 3, T_out).transpose(0, 2, 1)  # (N, 7, 2)

# === Normalize inputs and outputs ===
# Compute mean/std over the full dataset (all samples, all tokens, both x and y)
X_mean = X_np.mean(axis=0, keepdims=True)        # shape (1, T_in, 2)
X_std  = X_np.std(axis=0, keepdims=True) + 1e-8
Y_mean = Y_np.mean(axis=0, keepdims=True)        # shape (1, T_out, 2)
Y_std  = Y_np.std(axis=0, keepdims=True) + 1e-8

# Apply normalization
X_np = (X_np - X_mean) / X_std
Y_np = (Y_np - Y_mean) / Y_std

if DEBUG == True:
    print("X mean/std:", X_np.mean(), X_np.std())
    print("Y mean/std:", Y_np.mean(), Y_np.std())

np.savez("norm_stats.npz",
         X_mean=X_mean, X_std=X_std,
         Y_mean=Y_mean, Y_std=Y_std)

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
if DEBUG == True:
    print("max_seq_length =", max_seq_length)
batch_size = 32
lr = 1e-3

# Create data
seed = 42
rng = np.random.default_rng(seed)
perm = rng.permutation(N)

n_train = int(0.80 * N)
idx_train = perm[:n_train]
idx_test  = perm[n_train:]

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


train_ds = TrajDataset(X_np[idx_train], Y_np[idx_train])
test_ds  = TrajDataset(X_np[idx_test],  Y_np[idx_test])

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
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

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
        inside = (dist < radius -  0.005)   # was 0.005

        collided += inside.any(dim=1).sum().item()
        total += X.size(0)

    return collided / max(total, 1)

# Scheduled LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = 'min', # minimize metric
    factor = 0.5, # reduce LR by half
    patience = 3, # wait 3 epochs before reducing
)

# Train a few epochs
EPOCHS = 1
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
        coll = count_collisions(model, test_dl, radius=0.1)
        coll_rate_hist.append(coll)

        scheduler.step(te)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"epoch {epoch+1:02d} | train MSE {tr:.6f} | test MSE {te:.6f} | collisions {coll:.2%} | LR {current_lr:.6f}")

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
    model.load_state_dict(torch.load("transformer_traj.pth", map_location=device))
    model.eval()

if TIME_EVAL:
    print("\n" + "=" * 60)
    print("TIMING INFERENCE")
    print("=" * 60)

    # Single sample timing (useful for real-time applications)
    print("\n--- Single Sample Timing ---")
    single_X = test_ds[0][0].unsqueeze(0).to(device)

    # Warmup
    print("Running warmup iterations...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(single_X)

    # Time many single inferences
    print("Running timed iterations...")
    single_times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            _ = model(single_X)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            single_times.append((end - start) * 1000)  # Convert to ms

    single_times = np.array(single_times)
    print(f"\nSingle sample inference time: {single_times.mean():.3f}ms ± {single_times.std():.3f}ms")
    print(f"Min: {single_times.min():.3f}ms, Max: {single_times.max():.3f}ms")
    print(f"Median: {np.median(single_times):.3f}ms")
    print(f"Frequency: {1000 / single_times.mean():.0f} Hz")

    print("\n" + "=" * 60)

# Standard evaluation
print("\nComputing test metrics...")
test_mse = eval_epoch(test_dl)
test_coll = count_collisions(model, test_dl, radius=0.1)
print(f"Test MSE: {test_mse:.6f}")
print(f"Test collision rate: {test_coll * 100:.2f}%")

# plot first 3 validation samples
plot_many_samples(model, test_ds, indices=[0,1,2], title_prefix="test")
_ = excel_row_from_test_idx(2)


# or one specific index and save it
#plot_dataset_sample(model, test_ds, idx=5, save_path="test_sample_005.png")