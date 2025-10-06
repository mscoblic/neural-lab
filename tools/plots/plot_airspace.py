from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.config import load_yaml
from src.data import DataSchema, TrajectoryDataset, prepare_datasets, Normalizer, NormStats
from src.models import build_model_from_config


def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        blob = json.load(f)
    xs, ys = blob["inputs"], blob["outputs"]
    x_norm = Normalizer(str(xs["mode"]).lower())
    y_norm = Normalizer(str(ys["mode"]).lower())

    def _restore(state, norm_obj):
        if state["stats"] is None:
            norm_obj.fitted = True
            norm_obj.stats = None
        else:
            mean = torch.tensor(state["stats"]["mean"], dtype=torch.float32)
            std  = torch.tensor(state["stats"]["std"],  dtype=torch.float32)
            norm_obj.fitted = True
            norm_obj.stats = NormStats(mean=mean, std=std)

    _restore(xs, x_norm); _restore(ys, y_norm)
    return x_norm, y_norm


def segment_from_onehot(x_inputs_denorm: torch.Tensor) -> torch.Tensor:
    # inputs are [x0, y0, seg_1..seg_6]
    seg_oh = x_inputs_denorm[:, 2:8]
    return seg_oh.argmax(dim=1) + 1  # 1..6


def split_xy(vec18: torch.Tensor):
    return vec18[:, :9], vec18[:, 9:]


def reconstruct_full(x0y0: torch.Tensor, pred18: torch.Tensor, true18: torch.Tensor):
    x0 = x0y0[:, 0:1]
    y0 = x0y0[:, 1:2]
    px, py = split_xy(pred18)
    tx, ty = split_xy(true18)
    full_pred_x = torch.cat([x0, px], dim=1)
    full_pred_y = torch.cat([y0, py], dim=1)
    full_true_x = torch.cat([x0, tx], dim=1)
    full_true_y = torch.cat([y0, ty], dim=1)
    return full_pred_x, full_pred_y, full_true_x, full_true_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a specific run folder under runs/")
    ap.add_argument("--data-config",  default="configs/airspace/data_airspace.yaml")
    ap.add_argument("--model-config", default="configs/airspace/model_airspace.yaml")
    ap.add_argument("--subset", choices=["train", "val", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=5, help="samples per figure")
    ap.add_argument("--segment", type=int, default=0, help="plot only this segment (1..6); 0=all")
    ap.add_argument("--all-segments", action="store_true", help="plot six overlays (segments 1..6)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep-normalized", action="store_true",
                    help="Do not inverse-normalize inputs/outputs for plots/metrics")

    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    run_dir = (root / args.run_dir).resolve()
    ckpt_path  = run_dir / "model_best.pt"
    norm_path  = run_dir / "norm.json"
    split_path = run_dir / "split_indices.json"

    # ---- splits & norms ----
    with split_path.open("r") as f:
        splits = json.load(f)

    x_norm_saved, y_norm_saved = load_norm(norm_path)

    # ---- raw dataset ----
    data_cfg = load_yaml(root / args.data_config)
    schema = DataSchema(
        header=data_cfg["schema"]["header"],
        input_cols=data_cfg["schema"]["input_cols"],
        output_slices=[tuple(s) for s in data_cfg["schema"]["output_slices"]],
        infer_T_from_outputs=data_cfg["schema"].get("infer_T_from_outputs", True),
    )
    raw = TrajectoryDataset(str((root / data_cfg["excel_path"]).resolve()), schema, task_name=data_cfg["task_name"])

    # Use same normalization logic + exact splits
    train_ds, val_ds, test_ds, _ = prepare_datasets(
        raw,
        norm_inputs=data_cfg["normalization"]["inputs"],
        norm_outputs=data_cfg["normalization"]["outputs"],
        val_ratio=0.1, test_ratio=0.1,  # ignored because split_indices provided
        seed=42,
        split_indices=splits
    )
    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds if test_ds is not None else val_ds}
    ds = ds_map[args.subset]
    T = raw.T

    # ---- model ----
    model_cfg = load_yaml(root / args.model_config)
    model = build_model_from_config(model_cfg, raw.input_dim, raw.output_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.to(args.device)
    model.eval()

    # ---- pull all samples in the chosen subset (normalized) ----
    Xs, Ys = [], []
    for i in range(len(ds)):
        x_i, y_i = ds[i]
        Xs.append(x_i.unsqueeze(0))
        Ys.append(y_i.unsqueeze(0))
    X = torch.cat(Xs, dim=0)  # normalized
    Y = torch.cat(Ys, dim=0)

    X_denorm = X
    Y_true_den = Y
    with torch.no_grad():
        Y_pred = model(X.to(args.device)).cpu()
    Y_pred_den = Y_pred

    # reconstruct full curves
    full_pred_x, full_pred_y, full_true_x, full_true_y = reconstruct_full(X_denorm[:, :2], Y_pred_den, Y_true_den)
    seg_ids = segment_from_onehot(X_denorm)

    # helper: plot an overlay for a given segment id
    def plot_overlay_for_segment(seg_id: int, max_k: int):
        mask = (seg_ids == seg_id).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            print(f"[WARN] No samples for segment {seg_id}")
            return
        pick = mask[:min(max_k, mask.numel())].tolist()
        plt.figure()
        first = True
        for idx in pick:
            plt.plot(full_true_x[idx].numpy(), full_true_y[idx].numpy(), "-", alpha=0.5,
                     label="true" if first else None)
            plt.plot(full_pred_x[idx].numpy(), full_pred_y[idx].numpy(), "--", alpha=0.9,
                     label="pred" if first else None)
            first = False
        plt.title(f"{run_dir.name} | {args.subset} overlay — segment {seg_id}")
        plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.legend(); plt.tight_layout()
        out_png = run_dir / f"airspace_overlay_{args.subset}_segment{seg_id}.png"
        plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()
        print(f"[SAVED] {out_png}")

    # ---- modes ----
    if args.all_segments:
        for seg in range(1, 7):
            plot_overlay_for_segment(seg, args.num_samples)
    elif 1 <= args.segment <= 6:
        plot_overlay_for_segment(args.segment, args.num_samples)
    else:
        # default: just plot the first N arbitrary samples (mixed segments)
        n = min(args.num_samples, X.shape[0])
        for j in range(n):
            plt.figure()
            plt.plot(full_true_x[j].numpy(), full_true_y[j].numpy(), "-", label="true")
            plt.plot(full_pred_x[j].numpy(), full_pred_y[j].numpy(), "--", label="pred")
            seg = int(seg_ids[j].item())
            plt.title(f"{run_dir.name} | {args.subset} sample {j} | seg {seg}")
            plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.legend(); plt.tight_layout()
            out_png = run_dir / f"airspace_sample_{args.subset}_{j:03d}.png"
            plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()
            print(f"[SAVED] {out_png}")

if __name__ == "__main__":
    main()
