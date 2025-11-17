# ======================================================================================================================
# Imports
# ======================================================================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import json

# Import BeBOT
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'tools' / 'extra'))
from BeBOT import PiecewiseBernsteinPoly

# ======================================================================================================================
# Global definitions
# ======================================================================================================================
TRAIN = False        # train or load saved model
MODEL_PATH = "models/best_model.pth"
COUNT_COL = False
PLOT_TEST = True
TIME_EVAL = False   # run timing benchmark
SELF_EVAL = True   # user input (bottom of script)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "../data/CA3D/CA3D_15_obstacles_300k.xlsx"
df = pd.read_excel(file_path)

# Build 4 tokens
numObs = 15
radius = 0.05
obstacles = []
start    = df[["x0","y0", "z0"]].to_numpy(np.float32)
end      = df[["xf","yf", "zf"]].to_numpy(np.float32)
initialVel  = df[["vxinit","vyinit", "vzinit"]].to_numpy(np.float32)
control = df[["cpx","cpy", "cpz"]].to_numpy(np.float32)
for i in range(1, numObs + 1):
    obs_i = df[[f"ox{i}", f"oy{i}", f"oz{i}"]].to_numpy(np.float32)
    obstacles.append(obs_i)
output_cols = ["x2", "x3", "x4", "x5", "x6", "x7", "x8","y2", "y3", "y4", "y5", "y6", "y7", "y8", "z2","z3", "z4", "z5", "z6", "z7","z8"]

# Input and Output sizes
T_in = 3 + numObs
T_out = len(output_cols) // 3

# Hyperparameters
EPOCHS = 300
patience_counter = 0
patience_limit = 75
input_dim = 3
d_model = 28
num_heads = 2
num_layers = 1
d_ff = 64
dropout = 0.2
output_dim = 3
max_seq_length = T_in
batch_size = 64
lr = 1e-3
weight_decay = 0.05
warmup_epochs = 10
min_lr = 1e-6

# ======================================================================================================================
# Class definitions
# ======================================================================================================================
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
    def __init__(self, seq_len, d_model, hidden=128, t_out=7, out_dim=3, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.t_out = t_out
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(seq_len * d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),  # Add intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, t_out * out_dim)
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

class CosineAnnealingWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, max_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

# ======================================================================================================================
# Function definitions
# ======================================================================================================================

# Calculates inference time in forward pass and full pipeline
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

# Plots SELF_EVAL in a 3D window for orbiting
def plot_sample_interactive_from_input(model, test_input, ground_truth=None):
    """Interactive 3D plot using Plotly from raw input array with optional ground truth"""
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

    obstacles = test_input[0, 2:2+numObs]

    cp2x, cp2y, cp2z = test_input[0, 2 + numObs]
    r = 0.05  # radius

    # Build predicted trajectory
    pred_points = output[0]  # (T_out, 3)
    cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

    control_points_x = np.array([x0, cp2x, cp3[0], cp4[0], cp5[0],
                                 cp5[0], cp6[0], cp7[0], cp8[0], xf])
    control_points_y = np.array([y0, cp2y, cp3[1], cp4[1], cp5[1],
                                 cp5[1], cp6[1], cp7[1], cp8[1], yf])
    control_points_z = np.array([z0, cp2z, cp3[2], cp4[2], cp5[2],
                                 cp5[2], cp6[2], cp7[2], cp8[2], zf])

    tknots = np.array([0, 0.5, 1.0])
    t_eval = np.linspace(0, 1, 50)

    traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
    traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
    traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

    # Create figure
    fig = go.Figure()

    # Predicted Trajectory
    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Predicted Trajectory'
    ))

    # Predicted Control points
    fig.add_trace(go.Scatter3d(
        x=pred_points[:, 0], y=pred_points[:, 1], z=pred_points[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='green'),
        text=[str(i + 1) for i in range(len(pred_points))],
        textposition='top center',
        name='Predicted CPs'
    ))

    # Ground truth trajectory (if provided)
    if ground_truth is not None:
        gt_points = ground_truth[0]  # (T_out, 3)
        gt_cp3, gt_cp4, gt_cp5, gt_cp6, gt_cp7, gt_cp8 = gt_points

        gt_control_points_x = np.array([x0, cp2x, gt_cp3[0], gt_cp4[0], gt_cp5[0],
                                        gt_cp5[0], gt_cp6[0], gt_cp7[0], gt_cp8[0], xf])
        gt_control_points_y = np.array([y0, cp2y, gt_cp3[1], gt_cp4[1], gt_cp5[1],
                                        gt_cp5[1], gt_cp6[1], gt_cp7[1], gt_cp8[1], yf])
        gt_control_points_z = np.array([z0, cp2z, gt_cp3[2], gt_cp4[2], gt_cp5[2],
                                        gt_cp5[2], gt_cp6[2], gt_cp7[2], gt_cp8[2], zf])

        gt_traj_x = PiecewiseBernsteinPoly(gt_control_points_x, tknots, t_eval)[0, :]
        gt_traj_y = PiecewiseBernsteinPoly(gt_control_points_y, tknots, t_eval)[0, :]
        gt_traj_z = PiecewiseBernsteinPoly(gt_control_points_z, tknots, t_eval)[0, :]

        # Ground truth trajectory line
        fig.add_trace(go.Scatter3d(
            x=gt_traj_x, y=gt_traj_y, z=gt_traj_z,
            mode='lines',
            line=dict(color='red', width=4, dash='dash'),
            name='Ground Truth Trajectory'
        ))

        # Ground truth control points
        fig.add_trace(go.Scatter3d(
            x=gt_points[:, 0], y=gt_points[:, 1], z=gt_points[:, 2],
            mode='markers',
            marker=dict(size=6, color='red', symbol='x'),
            name='Ground Truth CPs'
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
        x=[cp2x], y=[cp2y], z=[cp2z],
        mode='markers',
        marker=dict(size=8, color='purple'),
        name='Heading CP'
    ))

    # Obstacle sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    for i, obs in enumerate(obstacles):
        ox, oy, oz = obs
        xs = ox + r * np.cos(u) * np.sin(v)
        ys = oy + r * np.sin(u) * np.sin(v)
        zs = oz + r * np.cos(v)

        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.7,
            colorscale='Reds',
            showscale=False,
            name=f'Obstacle {i+1}'
        ))

    title_str = 'Predicted vs Ground Truth' if ground_truth is not None else 'Predicted Trajectory'
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
        title=title_str
    )
    fig.write_image("my_trajectory.png")
    fig.show()  # Opens in browser with full interactivity!

# Plots a single test sample
@torch.no_grad()
def plot_dataset_sample(model, ds, idx, save_path=None, title_prefix="test"):
    """Plot dataset sample using interactive plotter"""
    model.eval()

    # fetch one sample (normalized tensors)
    X, Y_true = ds[idx]  # X: (T_in,3), Y_true: (T_out,3)

    # prepare stats
    X_mean_t = torch.from_numpy(X_mean.squeeze(0)).to(X.dtype)
    X_std_t = torch.from_numpy(X_std.squeeze(0)).to(X.dtype)
    Y_mean_t = torch.from_numpy(Y_mean.squeeze(0)).to(Y_true.dtype)
    Y_std_t = torch.from_numpy(Y_std.squeeze(0)).to(Y_true.dtype)

    # denormalize X to get raw input
    X_denorm = X * X_std_t + X_mean_t

    # denormalize Y_true to get ground truth
    Y_true_denorm = Y_true * Y_std_t + Y_mean_t

    # Convert to input format: add batch dimension
    test_input = X_denorm.unsqueeze(0).numpy()  # (1, T_in, 3)
    ground_truth = Y_true_denorm.unsqueeze(0).numpy()  # (1, T_out, 3)

    # Call existing interactive plotter with ground truth
    plot_sample_interactive_from_input(model, test_input, ground_truth=ground_truth)

# Visualize many samples quickly
@torch.no_grad()
def plot_many_samples(model, ds, indices, title_prefix="test"):
    """Convenience: plot several dataset samples by index."""
    for j, idx in enumerate(indices):
        plot_dataset_sample(model, ds, idx, title_prefix=title_prefix)

def plot_lr_schedule(lr_history, save_path="figs/lr_schedule.png"):
    """
    Visualize learning rate schedule over training

    Args:
        lr_history: List of learning rates from each epoch
        save_path: Path to save the figure
    """
    if len(lr_history) == 0:
        print("Warning: No LR history to plot")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(lr_history, linewidth=2, color='blue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule (Warmup + Cosine Annealing)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Add annotations for warmup phase
    if len(lr_history) > warmup_epochs:
        plt.axvline(x=warmup_epochs, color='red', linestyle='--',
                    alpha=0.5, label=f'Warmup complete (epoch {warmup_epochs})')
        plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved LR schedule plot to {save_path}")

def train_one_epoch():
    model.train()
    running = 0.0

    for X, Y in train_dl:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        Y_hat = model(X)

        loss=criterion(Y_hat, Y)

        loss.backward()
        optimizer.step()

        running += loss.item() * X.size(0)

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
def count_collisions(model, loader, radius):
    model.eval()
    device = next(model.parameters()).device

    # bring stats to device
    X_mean_t = torch.from_numpy(X_mean).to(device)
    X_std_t  = torch.from_numpy(X_std).to(device)
    Y_mean_t = torch.from_numpy(Y_mean).to(device)
    Y_std_t  = torch.from_numpy(Y_std).to(device)

    total = 0
    collided = 0

    # Loop through batches
    for X, _ in loader:
        X = X.to(device)
        Yp = model(X) # Make predictions

        # Denorm predictions
        Yp_den = Yp * Y_std_t + Y_mean_t

        # Grab obstacles locations and denorm
        obs_norm = X[:, 2:2 + numObs, :]
        obs_den = obs_norm * X_std_t[:, 2:2 + numObs, :] + X_mean_t[:, 2:2 + numObs, :]

        # Check collision with ANY obstacle
        has_collision = torch.zeros(X.size(0), dtype=torch.bool, device=device)

        for obs_idx in range(numObs):
            obs_center = obs_den[:, obs_idx, :]  # (B, 3)
            dist = torch.linalg.norm(Yp_den - obs_center[:, None, :], dim=-1)  # (B, T_out)
            inside = (dist < radius)
            has_collision |= inside.any(dim=1)  # Mark trajectory as collided if ANY point hits

        collided += has_collision.sum().item()
        total += X.size(0)

    return collided / max(total, 1)

@torch.no_grad()
def count_collisions_continuous(model, loader, radius, buffer=0.0, n_eval=200):
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
        Yp = model(X)

        # Denormalize predictions (global normalization)
        Yp_den = Yp * Y_std_t + Y_mean_t

        # Denormalize inputs
        X_den = X * X_std_t + X_mean_t

        # For each trajectory in batch
        for i in range(X.size(0)):
            # Extract start/end
            x0, y0, z0 = X_den[i, 0].cpu().numpy()
            xf, yf, zf = X_den[i, 1].cpu().numpy()

            # Get all obstacles
            obstacles = X_den[i, 2:2 + numObs].cpu().numpy()  # (numObs, 3)

            # Get predicted control points
            pred_points = Yp_den[i].cpu().numpy()  # (T_out, 3)
            cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

            cp2x, cp2y, cp2z = X_den[i, -1].cpu().numpy()

            # Build control point arrays with cp5 duplicated
            control_points_x = np.array([x0, cp2x, cp3[0], cp4[0], cp5[0],
                                         cp5[0], cp6[0], cp7[0], cp8[0], xf])
            control_points_y = np.array([y0, cp2y, cp3[1], cp4[1], cp5[1],
                                         cp5[1], cp6[1], cp7[1], cp8[1], yf])
            control_points_z = np.array([z0, cp2z, cp3[2], cp4[2], cp5[2],
                                         cp5[2], cp6[2], cp7[2], cp8[2], zf])

            # Evaluate continuous trajectory
            tknots = np.array([0, 0.5, 1.0])
            t_eval = np.linspace(0, 1, n_eval)

            traj_x = PiecewiseBernsteinPoly(control_points_x, tknots, t_eval)[0, :]
            traj_y = PiecewiseBernsteinPoly(control_points_y, tknots, t_eval)[0, :]
            traj_z = PiecewiseBernsteinPoly(control_points_z, tknots, t_eval)[0, :]

            # Stack into (n_eval, 3)
            trajectory = np.column_stack([traj_x, traj_y, traj_z])

            # Check collision with ANY obstacle
            has_collision = False
            for obs_center in obstacles:
                distances = np.linalg.norm(trajectory - obs_center, axis=1)
                if np.any(distances < radius - buffer):
                    has_collision = True
                    break

            if has_collision:
                collided += 1
            total += 1

    return collided / max(total, 1)

# ======================================================================================================================
# Training Setup
# ======================================================================================================================

# Stack into tokens
X_np = np.stack([start, end, *obstacles, initialVel, control], axis=1)
Y_np = df[output_cols].to_numpy(dtype=np.float32)

# Split into each dimension
N = Y_np.shape[0]
Y_np = Y_np.reshape(N, 3, T_out).transpose(0, 2, 1)

# Data processing
N = X_np.shape[0]
seed = 42
rng = np.random.default_rng(seed)
perm = rng.permutation(N)

# Train split
n_train = int(0.8 * N)
idx_train = perm[:n_train]
idx_test  = perm[n_train:]
print("\n=== Training Split ===")
print(f"n_train: {n_train}, n_test: {len(idx_test)}")

# Extract training samples
X_train = X_np[idx_train]
Y_train = Y_np[idx_train]

# Calculate normalization values
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-8
Y_mean = Y_train.mean(axis=0, keepdims=True)
Y_std = Y_train.std(axis=0, keepdims=True) + 1e-8

# Apply normalization
X_np_normalized = (X_np - X_mean) / X_std
Y_np_normalized = (Y_np - Y_mean) / Y_std

# Convert numpy arrays into PyTorch friendly containers
train_ds = TrajDataset(X_np_normalized[idx_train], Y_np_normalized[idx_train])
test_ds = TrajDataset(X_np_normalized[idx_test], Y_np_normalized[idx_test])

print("\n=== Normalization Check ===")
print(f"Training X - mean: {X_np_normalized[idx_train].mean():.6f}, std: {X_np_normalized[idx_train].std():.6f}")
print(f"Training Y - mean: {Y_np_normalized[idx_train].mean():.6f}, std: {Y_np_normalized[idx_train].std():.6f}")
print(f"Test X - mean: {X_np_normalized[idx_test].mean():.6f}, std: {X_np_normalized[idx_test].std():.6f}")
print(f"Test Y - mean: {Y_np_normalized[idx_test].mean():.6f}, std: {Y_np_normalized[idx_test].std():.6f}")

# Save normalization stats
np.savez("models/norm_stats.npz",
         X_mean=X_mean, X_std=X_std,
         Y_mean=Y_mean, Y_std=Y_std)

# Automates batch split, shuffling, iterating through batches during training
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size, shuffle=False)

# Build encoder
encoder = TransformerEncoder(
    input_dim=input_dim, d_model=d_model, num_heads=num_heads,
    num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length,
    dropout=dropout, output_dim=None
).to(device)

# Build FNN
head = FlattenMLPHead(seq_len=T_in, d_model=d_model, hidden=224, t_out=T_out, out_dim=output_dim).to(device)

# Wraps both together
model = TrajModel(encoder, head).to(device)

# Check datasize and model size for overfitting
print("\n=== Model Size to Data Size Check ===")
total_params = sum(p.numel() for p in model.parameters())
samples_per_param = len(train_ds) / total_params

print(f"Total parameters: {total_params:,}")
print(f"Training samples: {len(train_ds):,}")
print(f"Current ratio: {samples_per_param:.3f} samples/parameter")

print(f"\n--- Recommendations ---")

# Optimal params for current dataset
optimal_params_min = len(train_ds) // 5  # conservative (5 samples/param)
optimal_params_max = len(train_ds) // 1  # aggressive (1 sample/param)
print(f"For {len(train_ds):,} samples, aim for:")
print(f"  Conservative: {optimal_params_min:,} - {optimal_params_max:,} parameters")

# Optimal dataset for current model
optimal_samples_min = total_params * 1  # aggressive (1 sample/param)
optimal_samples_max = total_params * 5  # conservative (5 samples/param)
print(f"\nFor {total_params:,} parameters, aim for:")
print(f"  {optimal_samples_min:,} - {optimal_samples_max:,} samples")

# Status indicator
if samples_per_param < 0.5:
    status = "⚠️  SEVERE OVERFITTING RISK"
elif samples_per_param < 1.0:
    status = "⚠️  HIGH OVERFITTING RISK"
elif samples_per_param < 3.0:
    status = "⚡ MODERATE - Watch for overfitting"
else:
    status = "✓ GOOD REGIME"

print(f"\nStatus: {status}")
print(f"{'='*60}\n")

# Pick loss method
criterion = nn.MSELoss()

# Pick optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Scheduled LR
scheduler = CosineAnnealingWarmupScheduler(
    optimizer,
    warmup_epochs=warmup_epochs,
    total_epochs=EPOCHS,
    min_lr=min_lr,
    max_lr=lr
)

# Containers for test MSE, collision rate history, training losses
test_mse_hist = []
coll_rate_hist = []
train_losses = []
lr_history = []

# Track lowest test MSE
best_test_mse = float('inf')

print("\n=== Training ===")
if TRAIN:
    for epoch in range(EPOCHS):
        # train
        tr = train_one_epoch()
        train_losses.append(tr)

        # test MSE for this epoch
        te = eval_epoch(test_dl)
        test_mse_hist.append(te)

        # collision rate for this epoch
        #coll = count_collisions_continuous(model, test_dl, radius, buffer = 0)
        #coll = count_collisions(model, test_dl, radius)
        #coll_rate_hist.append(coll)

        # update learning rate
        current_lr = scheduler.step()
        lr_history.append(current_lr)

        # Early stopping logic
        if te < best_test_mse:      # Save the model if we did better than last time
            best_test_mse = te
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")  # Save best model
        else:                       # Or, increment counter
            patience_counter += 1

        # If we reach the patience maximum without improving, then early stop
        if patience_counter >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        # Print results
        print(f"epoch {epoch + 1:03d} | train_mse {tr:.4f} | test_mse {te:.4f} | "
              f"LR {current_lr:.6f} | patience {patience_counter}/{patience_limit}")

        # Generate training figures
        os.makedirs("figs", exist_ok=True)

        # Collision rate vs. epoch
        plt.figure(figsize=(6,4))
        xs = np.arange(1, len(coll_rate_hist)+1)
        plt.plot(xs, np.array(coll_rate_hist)*100.0, linewidth=2, label='Collision rate (%)')
        plt.xlabel('Epoch'); plt.ylabel('Collision rate (%)')
        plt.title('Collision rate over epochs')
        plt.grid(True); plt.legend()
        plt.savefig("figs/collision_rate_over_epochs.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Train/Test MSE vs. epoch
        plt.figure(figsize=(6,4))
        xs = np.arange(1, len(test_mse_hist)+1)
        plt.plot(xs, test_mse_hist, linestyle='-', linewidth=2, label='Test MSE')
        plt.plot(np.arange(1, len(train_losses)+1), train_losses, linestyle='-', linewidth=2, label='Train MSE')
        plt.xlabel('Epoch'); plt.ylabel('MSE')
        plt.title('MSE over epochs')
        plt.grid(True); plt.legend()
        plt.savefig("figs/mse_over_epochs.png", dpi=160, bbox_inches="tight")
        plt.close()

    print(f"\nTraining complete. Best test MSE: {best_test_mse:.4f}")

    # Visualize LR schedule
    plot_lr_schedule(lr_history)

    model.load_state_dict(torch.load("models/best_model.pth"))

else:
    # Load the saved model when not training
    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load normalization stats
    stats = np.load("models/norm_stats.npz")
    X_mean = stats['X_mean']
    X_std = stats['X_std']
    Y_mean = stats['Y_mean']
    Y_std = stats['Y_std']
    print("Model and normalization stats loaded successfully")

print("\n=== Loss Space Verification ===")

# Get one batch
X_batch, Y_batch = next(iter(test_dl))
X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

# Compute normalized loss (what you're seeing)
with torch.no_grad():
    Y_pred = model(X_batch)
    norm_loss = criterion(Y_pred, Y_batch)
    print(f"Normalized MSE: {norm_loss.item():.6f}")

# Denormalize and compute physical loss
Y_mean_t = torch.from_numpy(Y_mean).to(device)
Y_std_t = torch.from_numpy(Y_std).to(device)

Y_batch_phys = Y_batch * Y_std_t + Y_mean_t
Y_pred_phys = Y_pred * Y_std_t + Y_mean_t

phys_loss = criterion(Y_pred_phys, Y_batch_phys)
print(f"Physical MSE: {phys_loss.item():.6f}")
print(f"Physical RMSE: {torch.sqrt(phys_loss).item():.6f}")

# Show the ratio
print(f"\nRatio (phys/norm): {phys_loss.item() / norm_loss.item():.3f}")
print(f"Y_std mean: {Y_std.mean():.6f}")
print(f"Expected ratio ≈ (Y_std)²: {(Y_std.mean()**2):.6f}")

# Evaluate inference time
if TIME_EVAL:
    timing_stats = time_inference_comparison(model, test_ds, n_samples=100)

# Standard evaluation
print("\n=== Test Metrics ===")

# Calculate average MSE on all test samples
test_mse = eval_epoch(test_dl)

if COUNT_COL:
    # Calculate % of test set where there are collisions
    #test_coll = count_collisions_continuous(model, test_dl, radius=0.05, buffer = 0.01)
    test_coll = count_collisions(model, test_dl, radius)

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test collision rate: {test_coll * 100:.2f}%")

if PLOT_TEST:
    # Plot multiple test samples
    plot_many_samples(model, test_ds, indices=[0], title_prefix="test")

# Plot user input instead of test set
if SELF_EVAL:
    # Get a test sample as the base
    sample_idx = 0  # Change this to try different samples
    X_sample, Y_sample = test_ds[sample_idx]

    # Denormalize to get actual values
    X_mean_t = torch.from_numpy(X_mean.squeeze(0)).to(X_sample.dtype)
    X_std_t = torch.from_numpy(X_std.squeeze(0)).to(X_sample.dtype)
    Y_mean_t = torch.from_numpy(Y_mean.squeeze(0)).to(Y_sample.dtype)
    Y_std_t = torch.from_numpy(Y_std.squeeze(0)).to(Y_sample.dtype)

    X_denorm = (X_sample * X_std_t + X_mean_t).numpy()
    Y_denorm = (Y_sample * Y_std_t + Y_mean_t).numpy()

    # Modify the 2nd control point (last token in X)
    # Original value:
    print(f"Original 2nd control point: {X_denorm[-1]}")

    # Adjust it however you want:
    X_denorm[-1] = np.array([0.0, 0.0, 0.0])  # Set to whatever you want
    # Or shift it:
    # X_denorm[-1] += np.array([0.05, 0.05, 0.05])

    print(f"Modified 2nd control point: {X_denorm[-1]}")

    # Prepare for model (add batch dimension)
    test_input = X_denorm[np.newaxis, :]  # (1, T_in, 3)
    ground_truth = Y_denorm[np.newaxis, :]  # (1, T_out, 3)

    # Plot with ground truth
    plot_sample_interactive_from_input(model, test_input, ground_truth=ground_truth)
