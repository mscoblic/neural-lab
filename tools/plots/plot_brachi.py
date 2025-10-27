from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt

from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config


def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        norm = json.load(f)
    # Rebuild normalizers
    x_state = norm["inputs"]
    y_state = norm["outputs"]
    x_norm = Normalizer(x_state["mode"])
    y_norm = Normalizer(y_state["mode"])

    def _restore(nstate, norm_obj):
        if nstate["stats"] is None:
            norm_obj.fitted = True
            norm_obj.stats = None
        else:
            mean = torch.tensor(nstate["stats"]["mean"], dtype=torch.float32)
            std  = torch.tensor(nstate["stats"]["std"],  dtype=torch.float32)
            norm_obj.fitted = True
            norm_obj.stats = NormStats(mean=mean, std=std)

    _restore(x_state, x_norm)
    _restore(y_state, y_norm)
    return x_norm, y_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a specific run folder under runs/")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    run_dir = (root / args.run_dir).resolve()
    ckpt_path = run_dir / "model_best.pt"
    norm_path = run_dir / "norm.json"

    # Load configs from run folder
    with (run_dir / "configs_used.json").open() as f:
        configs = json.load(f)
    data_cfg = configs["data_config"]
    model_cfg = configs["model_config"]

    # Load split indices
    split_path = run_dir / "split_indices.json"
    with split_path.open("r") as f:
        splits = json.load(f)
    indices = splits[args.subset]

    # Load norm stats, recover normalizers (for inverse)
    x_norm, y_norm = load_norm(norm_path)

    # Rebuild dataset (raw, unnormalized)
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[list(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 2)),
    )
    ds = TrajectoryDataset(root / data_cfg["excel_path"], schema)
    T = ds.T

    # Rebuild model and load checkpoint
    model = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Plot a few samples
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    plt.figure(figsize=(6, 6))
    for j, idx in enumerate(indices[:args.num_samples]):
        x_raw, y_raw = ds[idx]

        # Normalize inputs to feed the model
        x_in = x_norm.transform(x_raw.unsqueeze(0))
        with torch.no_grad():
            y_pred_norm = model(x_in).squeeze(0)
        # Inverse normalization for plotting in original units
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

        # Reshape
        pred_2xT = y_pred.view(2, T)
        true_2xT = y_raw.view(2, T)

        # Final point error
        fx_pred, fy_pred = pred_2xT[0, -1].item(), pred_2xT[1, -1].item()
        fx_true, fy_true = true_2xT[0, -1].item(), true_2xT[1, -1].item()
        fpe = ((fx_pred - fx_true)**2 + (fy_pred - fy_true)**2) ** 0.5
        print(f"[Sample {j}] Final-point error: {fpe:.6f}")

        # Plot: ground truth dashed black; prediction solid color
        if j == 0:
            plt.plot(true_2xT[0], true_2xT[1], "--", color="black", linewidth=2.0, label="ground truth")
        else:
            plt.plot(true_2xT[0], true_2xT[1], "--", color="gray", alpha=0.3)
        plt.plot(pred_2xT[0], pred_2xT[1], color=colors[j % len(colors)], label=f"pred {j}" if j == 0 else None)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.title(run_dir.name)
    out_png = run_dir / "brachi_samples.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()