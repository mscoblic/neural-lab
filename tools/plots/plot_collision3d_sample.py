# tools/plots/plot_collision3d_sample.py

from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.config import load_yaml
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config

def set_axes_equal(ax):
    """Make 3D plot axes equal (so spheres look like spheres)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        norm = json.load(f)
    x_state = norm["inputs"]; y_state = norm["outputs"]
    x_norm = Normalizer(x_state["mode"]); y_norm = Normalizer(y_state["mode"])

    def _restore(nstate, norm_obj):
        if nstate["stats"] is None:
            norm_obj.fitted = True; norm_obj.stats = None
        else:
            mean = torch.tensor(nstate["stats"]["mean"], dtype=torch.float32)
            std  = torch.tensor(nstate["stats"]["std"],  dtype=torch.float32)
            norm_obj.fitted = True; norm_obj.stats = NormStats(mean=mean, std=std)

    _restore(x_state, x_norm); _restore(y_state, y_norm)
    return x_norm, y_norm


def draw_sphere(ax, center, radius=0.1, color="red", alpha=0.3):
    """Draw a translucent sphere at center=(x,y,z)."""
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--random", action="store_true", help="Pick random indices instead of first N")
    ap.add_argument("--outputs-are-normalized", action="store_true")
    ap.add_argument("--save", action="store_true", help="Also save PNGs in run-dir")
    ap.add_argument("--radius", type=float, default=0.05, help="Obstacle radius for plotting sphere")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    run_dir = (root / args.run_dir).resolve()
    ckpt_path = run_dir / "model_best.pt"
    norm_path = run_dir / "norm.json"

    with (run_dir / "split_indices.json").open("r") as f:
        splits = json.load(f)
    indices = splits[args.subset]

    if args.random:
        chosen_indices = random.sample(indices, k=min(args.num_samples, len(indices)))
    else:
        chosen_indices = indices[:args.num_samples]

    x_norm, y_norm = load_norm(norm_path)

    data_cfg = load_yaml(root / args.data_config)
    excel_path = (root / data_cfg["excel_path"]).resolve()
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[tuple(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K")),
    )
    ds = TrajectoryDataset(str(excel_path), schema)
    T = ds.T

    model_cfg = load_yaml(root / args.model_config)
    model = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    for j, idx in enumerate(chosen_indices):
        x_raw, y_true = ds[idx]

        with torch.no_grad():
            y_pred = model(x_norm.transform(x_raw.unsqueeze(0)))
            y_pred = y_norm.inverse(y_pred)

        Yp = y_pred.view(3, T).cpu().numpy()
        Yt = y_true.view(3, T).cpu().numpy()

        # obstacle xyz are at cols 6,7,8
        ox, oy, oz = [x_raw[k].item() for k in (6, 7, 8)]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(Yt[0], Yt[1], Yt[2], "--", label="ground truth")
        ax.plot(Yp[0], Yp[1], Yp[2], "o-", label="prediction")
        ax.scatter([Yt[0, 0]], [Yt[1, 0]], [Yt[2, 0]], c="green", s=60, label="start")
        ax.scatter([Yt[0, -1]], [Yt[1, -1]], [Yt[2, -1]], c="red", s=60, label="end")

        # Draw obstacle as a translucent red sphere
        draw_sphere(ax, (ox, oy, oz), radius=args.radius, color="red", alpha=0.4)

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{args.subset} idx={idx}")
        ax.legend()

        # <-- NEW: equalize aspect so x=y=z
        set_axes_equal(ax)

        if args.save:
            out_png = run_dir / f"collision3d_sample_{idx:05d}.png"
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            print(f"[SAVED] {out_png}")

        plt.show()



if __name__ == "__main__":
    main()
