# ======================================================================================================================
# Imports
# ======================================================================================================================
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
import plotly.graph_objects as go
import json

# Import BeBOT
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'tools' / 'extra'))
from BeBOT import PiecewiseBernsteinPoly

# ======================================================================================================================
# Global definitions
# ======================================================================================================================
TRAIN = True        # train or load saved model
MODEL_PATH = "models/best_model.pth"
TIME_EVAL = False   # run timing benchmark
SELF_EVAL = False   # user input (bottom of script)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "../data/CA3D/CA3D_multiple_obstacles.xlsx"
df = pd.read_excel(file_path)

# Build 4 tokens
numObs = 10
radius = 0.05
obstacles = []
start    = df[["x0","y0", "z0"]].to_numpy(np.float32)
end      = df[["xf","yf", "zf"]].to_numpy(np.float32)
control  = df[["vxinit","vyinit", "vzinit"]].to_numpy(np.float32)
for i in range(1, numObs + 1):
    obs_i = df[[f"ox{i}", f"oy{i}", f"oz{i}"]].to_numpy(np.float32)
    obstacles.append(obs_i)
output_cols = ["x2", "x3", "x4", "x5", "x6", "x7", "x8","y2", "y3", "y4", "y5", "y6", "y7", "y8","z2", "z3", "z4", "z5", "z6", "z7","z8"]

# Input and Output sizes
T_in = 3 + numObs
T_out = len(output_cols) // 3

# Hyperparameters
EPOCHS = 200
patience_counter = 0
patience_limit = 200
input_dim = 3
d_model = 64
num_heads = 4
num_layers = 2
d_ff = 128
dropout = 0.2
output_dim = 3
max_seq_length = T_in
batch_size = 64
lr = 1e-3
weight_decay = 0.01

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

    obstacles = test_input[0, 2:2+numObs]

    cpx, cpy, cpz = test_input[0, 2 + numObs]
    r = 0.05  # radius

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

# Plots a single test sample
@torch.no_grad()
def plot_dataset_sample(model, ds, idx, save_path=None, title_prefix="test",
                        elev=45, azim=-70, n_eval=50, plot_continuous=True):
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

    obstacles = X_denorm[2:2+numObs].numpy()    # obstacle

    cpx, cpy, cpz = X_denorm[2+numObs].tolist()  # control

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
    R = 0.05
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]

    for i, obs in enumerate(obstacles):
        ox, oy, oz = obs
        xs = ox + R * np.cos(u) * np.sin(v)
        ys = oy + R * np.sin(u) * np.sin(v)
        zs = oz + R * np.cos(v)
        ax.plot_surface(xs, ys, zs, color='red', alpha=0.3, linewidth=0)

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
        Yp_den = Yp * Y_std_t[:, 0, :] + Y_mean_t[:, 0, :]

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
def count_collisions_continuous(model, loader, radius, buffer=0.0, n_eval=500):
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
        Yp_den = Yp * Y_std_t[:, 0, :] + Y_mean_t[:, 0, :]

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
            cp2, cp3, cp4, cp5, cp6, cp7, cp8 = pred_points

            # Build control point arrays with cp5 duplicated
            control_points_x = np.array([x0, cp2[0], cp3[0], cp4[0], cp5[0],
                                         cp5[0], cp6[0], cp7[0], cp8[0], xf])
            control_points_y = np.array([y0, cp2[1], cp3[1], cp4[1], cp5[1],
                                         cp5[1], cp6[1], cp7[1], cp8[1], yf])
            control_points_z = np.array([z0, cp2[2], cp3[2], cp4[2], cp5[2],
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
X_np = np.stack([start, end, *obstacles, control], axis=1)
Y_np = df[output_cols].to_numpy(dtype=np.float32)

# Split into each dimension
N = Y_np.shape[0]
Y_np = Y_np.reshape(N, 3, T_out).transpose(0, 2, 1)

# Data processing
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
head = FlattenMLPHead(seq_len=T_in, d_model=d_model, hidden=256, t_out=T_out, out_dim=output_dim).to(device)

# Wraps both together
model = TrajModel(encoder, head).to(device)

# Pick loss method
criterion = nn.MSELoss()

# Pick optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay)

# Scheduled LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = 'min', # minimize metric
    factor = 0.5, # reduce LR by half
    patience = 3, # wait 3 epochs before reducing
)

# Containers for test MSE, collision rate history, training losses
test_mse_hist = []
coll_rate_hist = []
train_losses = []

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
        coll = count_collisions_continuous(model, test_dl, radius, buffer = 0, n_eval=200)
        #coll = count_collisions(model, test_dl)
        coll_rate_hist.append(coll)

        # update learning rate
        scheduler.step(te)
        current_lr = optimizer.param_groups[0]['lr']

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
        print(f"epoch {epoch+1:02d} | train MSE {tr:.6f} | test MSE {te:.6f} | collisions {coll:.2%} | LR {current_lr:.6f}")

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

# Evaluate inference time
if TIME_EVAL:
    timing_stats = time_inference_comparison(model, test_ds, n_samples=100)

# Standard evaluation
print("\n=== Test Metrics ===")

# Calculate average MSE on all test samples
test_mse = eval_epoch(test_dl)

# Calculate % of test set where there are collisions
test_coll = count_collisions_continuous(model, test_dl, radius=0.05, buffer = 0, n_eval=200)
#test_coll = count_collisions(model, test_dl)

print(f"Test MSE: {test_mse:.6f}")
print(f"Test collision rate: {test_coll * 100:.2f}%")

# Plot multiple test samples
plot_many_samples(model, test_ds, indices=[0,1,2], title_prefix="test")

# Plot user input instead of test set
if SELF_EVAL:
    # Create test input with 10 obstacles
    test_input = np.array([[
        [0.0, 0.0, 0.0],  # start
        [1.0, 1.0, 1.0],  # end
        [0.3, 0.3, 0.35], # 10 obstacles
        [0.45, 0.4, 0.4],
        [0.5, 0.5, 0.55],
        [0.6, 0.64, 0.6],
        [0.7, 0.72, 0.7],
        [0.25, 0.35, 0.45],
        [0.35, 0.45, 0.55],
        [0.45, 0.55, 0.65],
        [0.55, 0.65, 0.75],
        [0.65, 0.75, 0.85],
        [0.0, 0.0, 0.0]  # initial velocity
    ]], dtype=np.float32)

    # Normalize
    test_input_norm = (test_input - X_mean) / X_std
    test_input_tensor = torch.from_numpy(test_input_norm).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output_norm = model(test_input_tensor)
        output = output_norm.cpu().numpy() * Y_std + Y_mean  # Denormalize

    # Plot in separate orbiting figure
    plot_sample_interactive_from_input(model, test_input)
