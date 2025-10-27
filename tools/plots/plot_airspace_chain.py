# tools/plots/plot_airspace_chain.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.data import DataSchema, TrajectoryDataset, prepare_datasets
from src.models import build_model_from_config


def one_hot_segment(seg_id: int) -> torch.Tensor:
    v = torch.zeros(6, dtype=torch.float32)
    v[seg_id - 1] = 1.0
    return v

@torch.no_grad()
def rollout_chain(start_xy, start_seg: int, model, ds_normed):
    assert 1 <= start_seg <= 6
    x0, y0 = float(start_xy[0]), float(start_xy[1])

    xs, ys = [x0], [y0]
    boundaries = []

    # Reverse order
    for seg in range(start_seg, 0, -1):
        x_phys = torch.cat(
            [torch.tensor([x0, y0], dtype=torch.float32), one_hot_segment(seg)],
            dim=0
        ).unsqueeze(0)  # (1, 8)

        x_in = ds_normed.x_norm.transform(x_phys)
        y_pred_norm = model(x_in).cpu()
        y_pred = ds_normed.y_norm.inverse(y_pred_norm)

        px = y_pred[:, :9].squeeze(0)   # (9,)
        py = y_pred[:, 9:].squeeze(0)   # (9,)

        start_idx = len(xs) - 1
        xs.extend(px.tolist())
        ys.extend(py.tolist())
        end_idx = len(xs) - 1

        boundaries.append((seg, start_idx, end_idx))

        # Next start is the last predicted point of this segment
        x0 = float(px[-1])
        y0 = float(py[-1])

    return torch.tensor(xs), torch.tensor(ys), boundaries


def main():
    ap = argparse.ArgumentParser(description="Chain segments 1..6 (or start_seg..6) from an initial point.")
    ap.add_argument("--run-dir", required=True, help="Path to a specific run folder under runs/")
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test",
                    help="Which split's normalizers to use (fit-on-TRAIN only; split controls which Subset wrapper we grab).")
    ap.add_argument("--start-x", type=float, required=True)
    ap.add_argument("--start-y", type=float, required=True)
    ap.add_argument("--start-seg", type=int, default=6, help="Starting segment (1..6). Chains through 6.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    run_dir = (root / args.run_dir).resolve()
    ckpt_path  = run_dir / "model_best.pt"
    norm_path  = run_dir / "norm.json"
    split_path = run_dir / "split_indices.json"

    # Load configs from run folder
    with (run_dir / "configs_used.json").open() as f:
        configs = json.load(f)
    data_cfg = configs["data_config"]
    model_cfg = configs["model_config"]

    # Load splits
    with split_path.open("r") as f:
        splits = json.load(f)

    # Rebuild raw dataset
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[list(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
        K=int(data_cfg["schema"].get("K", 2)),
    )
    raw = TrajectoryDataset(str((root / data_cfg["excel_path"]).resolve()), schema, task_name=data_cfg["task_name"])

    # Use exact same normalization logic + split layout
    train_ds, val_ds, test_ds, _ = prepare_datasets(
        raw,
        norm_inputs=data_cfg["normalization"]["inputs"],
        norm_outputs=data_cfg["normalization"]["outputs"],
        val_ratio=0.1, test_ratio=0.1,
        seed=42,
        split_indices=splits
    )
    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds if test_ds is not None else val_ds}
    ds_normed = ds_map[args.subset]

    # Build model & load weights
    model = build_model_from_config(model_cfg, raw.input_dim, raw.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()

    # Rollout chain
    xs, ys, boundaries = rollout_chain((args.start_x, args.start_y), args.start_seg, model, ds_normed)

    # Plot: color by segment, annotate ends
    plt.figure(figsize=(7, 6))
    # Draw the full polyline lightly
    plt.plot(xs.numpy(), ys.numpy(), "-", color="0.7", linewidth=1.0, label="full path")

    # Color chunks by segment
    cmap = plt.get_cmap("tab10")
    for k, (seg, s_idx, e_idx) in enumerate(boundaries):
        Xk = xs[s_idx:e_idx+1].numpy()
        Yk = ys[s_idx:e_idx+1].numpy()
        plt.plot(Xk, Yk, "-", linewidth=2.0, color=cmap((seg-1) % 10), label=f"seg {seg}" if k == 0 else None)
        # Annotate the last point of this segment
        plt.scatter([Xk[-1]], [Yk[-1]], s=40, edgecolors="black", facecolors="none")
        plt.text(Xk[-1], Yk[-1], f"  S{seg}", fontsize=9)

    # Mark initial point
    plt.scatter([xs[0].item()], [ys[0].item()], s=60, color="green", label="start")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{run_dir.name} | chained rollout {list(range(args.start_seg, 0, -1))}")
    plt.legend()
    plt.tight_layout()

    out_png = run_dir / f"chain_startseg{args.start_seg}_x{args.start_x}_y{args.start_y}.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()