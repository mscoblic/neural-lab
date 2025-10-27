from __future__ import annotations
import argparse, json
from pathlib import Path
import csv

import torch
import numpy as np
from src.data import DataSchema, TrajectoryDataset, Normalizer, NormStats
from src.models import build_model_from_config

'''
python -m tools.eval.eval_collisions_3d \
  --run-dir runs/collision_3D_ffn_102725_s42_de9269 \
  --R 0.10 --verbose
'''

def load_norm(norm_path: Path):
    with norm_path.open("r") as f:
        norm = json.load(f)
    x_state, y_state = norm["inputs"], norm["outputs"]
    x_norm, y_norm = Normalizer(x_state["mode"]), Normalizer(y_state["mode"])

    def _restore(nstate, norm_obj):
        if nstate["stats"] is None:
            norm_obj.fitted, norm_obj.stats = True, None
        else:
            mean = torch.tensor(nstate["stats"]["mean"], dtype=torch.float32)
            std  = torch.tensor(nstate["stats"]["std"],  dtype=torch.float32)
            norm_obj.fitted = True
            norm_obj.stats = NormStats(mean=mean, std=std)
    _restore(x_state, x_norm); _restore(y_state, y_norm)
    return x_norm, y_norm


# ---------- geometry: segment-sphere distance ----------
def _point_in_sphere(p, c, R):
    d = p - c
    return (d @ d) <= R * R

def _segment_hits_sphere(p1, p2, c, R):
    v = p2 - p1
    vv = float(v @ v)
    if vv == 0.0:
        return _point_in_sphere(p1, c, R)
    t = float((c - p1) @ v) / vv
    t = max(0.0, min(1.0, t))
    q = p1 + t * v
    d = q - c
    return (d @ d) <= R * R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="runs/<name>")
    ap.add_argument("--subset", choices=["train","val","test"], default="test")
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--R", type=float, default=0.10, help="Obstacle radius (visualization uses the same)")
    ap.add_argument("--save-csv", action="store_true", help="Write per-sample results to collisions.csv in run dir")
    ap.add_argument("--verbose", action="store_true", help="Print progress for each sample")  # <<< NEW
    args = ap.parse_args()

    # Run from repo root with -m so imports work
    root = Path(__file__).resolve().parents[2]
    run_dir = (root / args.run_dir).resolve()

    # --- configs / splits ---
    with (run_dir / "configs_used.json").open() as f:
        cfgs = json.load(f)
    data_cfg, model_cfg = cfgs["data_config"], cfgs["model_config"]

    with (run_dir / "split_indices.json").open() as f:
        splits = json.load(f)
    indices = splits[args.subset]

    # --- norms, dataset, model ---
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
    T, K = ds.T, schema.K  # expect 7 and 3

    model = build_model_from_config(model_cfg, ds.input_dim, ds.output_dim)
    state = torch.load(run_dir / args.ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    # --- loop over samples ---
    total = 0
    collided = 0
    rows = []

    for j, idx in enumerate(indices):
        total += 1
        x_row, y_true_row = ds[idx]  # raw (unnormalized)
        # obstacle center from inputs
        ox, oy, oz = x_row[6].item(), x_row[7].item(), x_row[8].item()
        c = torch.tensor([ox, oy, oz], dtype=torch.float32)

        # predict and denormalize
        with torch.no_grad():
            y_pred_norm = model(x_norm.transform(x_row.unsqueeze(0))).squeeze(0)
        y_pred = y_norm.inverse(y_pred_norm).detach().cpu()

        pred_3xT = y_pred.view(K, T)  # (3,7)
        pts = pred_3xT.T  # (T,3)

        # collision checks: point hits OR segment intersections
        point_hits = sum(int(_point_in_sphere(pts[t], c, args.R)) for t in range(T))
        seg_hits = 0
        for t in range(T - 1):
            if _segment_hits_sphere(pts[t], pts[t + 1], c, args.R):
                seg_hits += 1

        collides = (point_hits > 0) or (seg_hits > 0)
        collided += int(collides)

        rows.append({
            "index": idx,
            "point_hits": point_hits,
            "segment_hits": seg_hits,
            "collides": int(collides),
        })

    if args.verbose and (j % 100 == 0 or collides):
        print(f"[{j + 1}/{len(indices)}] idx={idx} → "
              f"point_hits={point_hits}, seg_hits={seg_hits}, collides={collides}")

    rate = 100.0 * collided / max(1, total)
    print(f"[COLLISIONS] {collided}/{total} predicted trajectories collided ({rate:.2f}%).")

    if args.save_csv:
        out_csv = run_dir / f"collisions_{args.subset}.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index","point_hits","segment_hits","collides"])
            w.writeheader()
            w.writerows(rows)
        print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
