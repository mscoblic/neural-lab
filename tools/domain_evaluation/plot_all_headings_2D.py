#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def _load_config_maybe_path(obj, *, yaml_ok: bool = True):
    """
    Accept a str path OR a dict that already contains the config.
    Returns (config_dict, base_path) where base_path is the parent
    directory of the config file if it came from a path, else None.
    """
    if isinstance(obj, str):
        p = Path(obj).expanduser().resolve()
        if p.suffix.lower() == ".json":
            return json.load(open(p)), p.parent
        if yaml_ok and p.suffix.lower() in (".yml", ".yaml"):
            import yaml
            return yaml.safe_load(open(p)), p.parent
        raise ValueError(f"Unsupported config file type: {p}")
    if isinstance(obj, dict):
        return obj, None
    if isinstance(obj, (list, tuple)) and len(obj) == 1 and isinstance(obj[0], str):
        return _load_config_maybe_path(obj[0], yaml_ok=yaml_ok)
    raise TypeError(f"Unrecognized config handle: {type(obj)}")

def _first_filelike_string(obj) -> str | None:
    """
    Depth-first search for a string that *looks* like a dataset file.
    """
    exts = (".csv", ".xlsx", ".xls")
    if isinstance(obj, dict):
        for v in obj.values():
            s = _first_filelike_string(v)
            if s: return s
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            s = _first_filelike_string(v)
            if s: return s
    elif isinstance(obj, str):
        low = obj.lower()
        if low.endswith(exts):  # likely a dataset path
            return obj
    return None

def _get_output_slices(cfg: dict) -> list:
    """
    Try a few likely locations/names for output_slices.
    """
    for key_path in [
        ["output_slices"],
        ["outputs", "slices"],
        ["dataset", "output_slices"],
        ["data", "output_slices"],
    ]:
        cur = cfg
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (list, tuple)) and len(cur) > 0:
            return cur
    raise KeyError("Could not locate output_slices in the data config.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to the run dir (the one with split_indices.json)")
    ap.add_argument("--out", default="all_headings.png", help="Overlay plot output (PNG)")
    ap.add_argument("--grid-pdf", default="", help="Optional small-multiples PDF (set e.g. all_headings_grid.pdf)")
    ap.add_argument("--data-config", default="", help="Optional override: path to data config (yaml/json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_used = json.load(open(run_dir / "configs_used.json"))
    splits   = json.load(open(run_dir / "split_indices.json"))

    # ---- get data config (path or dict) robustly
    data_cfg_obj = (
        args.data_config
        or cfg_used.get("data_config")
        or cfg_used.get("data_config_path")
        or (cfg_used.get("configs") or {}).get("data")
    )
    if data_cfg_obj is None:
        raise KeyError("Could not locate data config in configs_used.json. "
                       "Tried keys: data_config, data_config_path, configs.data")

    data_cfg, cfg_base = _load_config_maybe_path(data_cfg_obj)

    # ---- resolve dataset file path (handle many possible key names or nesting)
    # common names we’ll try first:
    likely_keys = [
        "data_file", "file", "path", "csv", "xlsx", "dataset_path",
        "table_path", "table", "input_file"
    ]
    data_file_obj = None
    for k in likely_keys:
        if isinstance(data_cfg, dict) and k in data_cfg:
            data_file_obj = data_cfg[k]
            break
    if data_file_obj is None:
        # maybe stored in configs_used.json meta
        data_file_obj = (cfg_used.get("meta") or {}).get("data_file")
    if data_file_obj is None:
        # final fallback: scan the whole config for a path-like string
        data_file_obj = _first_filelike_string(data_cfg)

    if not isinstance(data_file_obj, str):
        raise KeyError("Could not find a dataset file path in the data config. "
                       "Looked for keys: " + ", ".join(likely_keys))

    data_file = Path(data_file_obj)
    if not data_file.is_absolute():
        base = cfg_base if cfg_base is not None else run_dir
        data_file = (base / data_file).resolve()

    # ---- identify heading slice from output_slices (dict form or [start, end])
    output_slices = _get_output_slices(data_cfg)
    if isinstance(output_slices[0], dict):
        heading_slice = None
        for s in output_slices:
            name = str(s.get("name", "")).lower()
            if "heading" in name:
                heading_slice = slice(s["start"], s["end"])
                break
        if heading_slice is None:
            s0 = output_slices[0]
            heading_slice = slice(s0["start"], s0["end"])
    else:
        s0 = output_slices[0]
        heading_slice = slice(s0[0], s0[1])

    # ---- load table & extract test rows + heading columns
    df = load_table(data_file)
    test_idx = np.array(splits["test"], dtype=int)

    # drop non-numeric columns automatically (and any accidental index column)
    df_vals = df.select_dtypes(include=[np.number]).values
    H = df_vals[np.atleast_1d(test_idx)][:, heading_slice]
    if H.ndim != 2:
        raise RuntimeError(f"Expected heading slice to produce (N,T); got shape {H.shape}")

    T = H.shape[1]
    x = np.arange(T)

    # -------- overlay plot
    plt.figure(figsize=(9, 5))
    for row in H:
        plt.plot(x, row, alpha=0.15)
    plt.title(f"All test headings ({H.shape[0]} sequences)")
    plt.xlabel("timestep")
    plt.ylabel("heading")
    plt.grid(True, alpha=0.2)
    out_path = run_dir / args.out
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")

    # -------- optional small-multiples PDF (pages of 6x6)
    if args.grid_pdf:
        import math
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = run_dir / args.grid_pdf
        pp = PdfPages(pdf_path)
        per_page = 36
        pages = math.ceil(H.shape[0] / per_page)
        for p in range(pages):
            start = p * per_page
            end = min((p + 1) * per_page, H.shape[0])
            r = c = int(math.ceil(math.sqrt(per_page)))  # ~6x6
            fig = plt.figure(figsize=(11, 8.5))
            for i, row in enumerate(H[start:end], start=1):
                ax = fig.add_subplot(r, c, i)
                ax.plot(x, row, linewidth=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"#{start + i - 1}", fontsize=8)
            fig.suptitle(f"Test headings {start}–{end-1}")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pp.savefig(fig, dpi=200)
            plt.close(fig)
        pp.close()
        print(f"[SAVED] {pdf_path}")

if __name__ == "__main__":
    main()
