from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config

'''
 python -m tools.plots.plot_3d_scatter \
  --run-dir runs/collision_3D_ffn_102725_s42_de9269 \
  --subset test --num-samples 3 --show-gt --show-obs
'''

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--subset", choices=["train","val","test"], default="test")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--show-gt", action="store_true")
    ap.add_argument("--show-obs", action="store_true")
    ap.add_argument("--elev", type=float, default=45)
    ap.add_argument("--azim", type=float, default=-70)
    ap.add_argument("--R", type=float, default=0.10, help="obstacle radius (for viz only)")
    args = ap.parse_args()

    # project root (run with: python -m tools.plots.plot_3d_scatter ...)
    root = Path(__file__).resolve().parents[2]
    run_dir = (root / args.run_dir).resolve()

    with (run_dir / "configs_used.json").open() as f:
        configs = json.load(f)
    data_cfg = configs["data_config"]; model_cfg = configs["model_config"]

    with (run_dir / "split_indices.json").open() as f:
        splits = json.load(f)
    indices = splits[args.subset]

    x_norm, y_norm = load_norm(run_dir / "norm.json")

    excel_path = (root / data_cfg["excel_path"]).resolve()
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[list(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 3)),
    )
    ds = TrajectoryDataset(excel_path, schema)
    T, K = ds.T, schema.K  # T=7, K=3

    # model
    ckpt_path = run_dir / args.ckpt
    model = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"]); model.eval()

    for j, idx in enumerate(indices[:args.num_samples]):
        x_row, y_true_row = ds[idx]  # raw (unnormalized)

        # unpack start/end/obstacle by your schema
        sx, sy, sz = x_row[0].item(), x_row[1].item(), x_row[2].item()
        ex, ey, ez = x_row[3].item(), x_row[4].item(), x_row[5].item()
        ox, oy, oz = x_row[6].item(), x_row[7].item(), x_row[8].item()

        # prediction in original units
        x_in = x_norm.transform(x_row.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

        pred_3xT = y_pred.view(K, T)      # (3,7)
        gt_3xT   = y_true_row.view(K, T)  # (3,7)

        # ---- 3D scatter (no connecting lines) ----
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_box_aspect([1, 1, 1])

        # predicted points (dots)
        ax.scatter(pred_3xT[0].numpy(), pred_3xT[1].numpy(), pred_3xT[2].numpy(),
                   s=28, marker="o", label="prediction", depthshade=False)

        # optional GT points (x’s)
        if args.show_gt:
            ax.scatter(gt_3xT[0].numpy(), gt_3xT[1].numpy(), gt_3xT[2].numpy(),
                       s=36, marker="x", label="ground truth", depthshade=False)

        # start / end markers
        ax.scatter([sx], [sy], [sz], s=80, marker="o", label="start", depthshade=False)
        ax.scatter([ex], [ey], [ez], s=120, marker="*", label="end", depthshade=False)

        # optional obstacle sphere (visual only)
        if args.show_obs:
            R = float(args.R)
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            xs = ox + R * np.cos(u) * np.sin(v)
            ys = oy + R * np.sin(u) * np.sin(v)
            zs = oz + R * np.cos(v)
            ax.plot_surface(xs, ys, zs, alpha=0.25, linewidth=0)  # translucent

        # cosmetics
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{run_dir.name} | {args.subset} sample {j}")
        ax.legend(loc="best"); ax.grid(True)

        out_png = run_dir / f"traj3d_scatter_{j:03d}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
