#!/usr/bin/env python3
# scripts/plot/alife_scripts/map_elites_heatmap.py
import re, os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))

TICK_EVERY = 20
MIN_VAL      = 0.0  
MAX_VAL      = 1.0  
DECIMALS     = 3 
CMAP           = "viridis"

GRID_SIZE = 100

X_KEY = "decay"        # "bin", "lambda", "decay", "plast"
Y_KEY = "lambda"       # "seed", "bin", "lambda", "decay", "plast"

SEED_TO_PLOT =2
EXPERIMENT_DIR = os.path.join(base_path, "37_td_sf_me_dynamic_deter_gated")

PARAM_PATTERNS = {
    "bin":     r"bin\s+(?P<bin>\d+)",
    "id":      r"id:\s+(?P<id>\d+)",
    "fit":     r"fit:\s+(?P<fit>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    "lambda":  r"(?:λ|lambda):\s+(?P<lambda>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    "decay":   r"decay:\s+(?P<decay>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    "plast":   r"plast:\s+(?P<plast>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
}

def load_one_archive(path: str) -> list[dict]:
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if "bin" not in line:
                continue
            record: dict = {}
            for key, pat in PARAM_PATTERNS.items():
                m = re.search(pat, line)
                if m and m.group(key):
                    val = m.group(key)
                    record[key] = int(val) if key in ("bin", "id") else float(val)
            if "fit" in record:           # discard lines we failed to parse
                entries.append(record)
    return entries

def build_matrix(records: list[dict], x_key: str, y_key: str):
    if {"lambda", "decay"} <= {x_key, y_key}:
        bins = max(rec["bin"] for rec in records) + 1
        grid_cols = int(math.sqrt(GRID_SIZE))
        grid_rows = int(math.sqrt(GRID_SIZE))

        lam_vals = [i / grid_cols for i in range(grid_cols)]
        dec_vals = [i / grid_rows for i in range(grid_rows)]

        if x_key == "lambda":
            x_vals, y_vals = lam_vals, dec_vals
        else:
            x_vals, y_vals = dec_vals, lam_vals

        mat = np.full((len(y_vals), len(x_vals)), np.nan)

        for rec in records:
            lam_idx = min(int(math.floor(rec["lambda"] * grid_cols)), grid_cols - 1)
            dec_idx = min(int(math.floor(rec["decay"] * grid_rows)), grid_rows - 1)

            if x_key == "lambda":
                col, row = lam_idx, dec_idx
            else:
                col, row = dec_idx, lam_idx

            cur = mat[row, col]
            if np.isnan(cur) or rec["fit"] > cur:
                mat[row, col] = rec["fit"]

        return x_vals, y_vals, mat

    x_vals = sorted({rec[x_key] for rec in records})
    y_vals = sorted({rec[y_key] for rec in records})
    mat = np.full((len(y_vals), len(x_vals)), np.nan)

    for rec in records:
        xi = x_vals.index(rec[x_key])
        yi = y_vals.index(rec[y_key])
        cur = mat[yi, xi]
        if np.isnan(cur) or rec["fit"] > cur:
            mat[yi, xi] = rec["fit"]
    return x_vals, y_vals, mat

def main():
    logs_dir = os.path.join(EXPERIMENT_DIR, "logs", "map_elites")
    files = sorted(
        glob.glob(os.path.join(logs_dir, "map_elites_archive_*.csv")),
        key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0])
    )
    if not files:
        sys.exit(f"No archives found in {logs_dir}")

    if Y_KEY == "seed":
        records = []
        for f in files:
            seed_num = int(os.path.basename(f).split("_")[-1].split(".")[0])
            for rec in load_one_archive(f):
                rec["seed"] = seed_num
                records.append(rec)
        print(f"Loaded {len(records)} map-elites entries for plotting.")
    else:
        if SEED_TO_PLOT is None:
            sys.exit("SEED_TO_PLOT must be set when Y_KEY is not 'seed'")
        seed_file = os.path.join(logs_dir,
                                 f"map_elites_archive_{SEED_TO_PLOT}.csv")
        if not os.path.isfile(seed_file):
            sys.exit(f"Seed file {seed_file} not found")
        records = load_one_archive(seed_file)
        print(f"Loaded {len(records)} map-elites entries for plotting.")

    x_vals, y_vals, matrix = build_matrix(records, X_KEY, Y_KEY)

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap(CMAP).copy()
    matrix_masked = np.ma.masked_invalid(matrix)

    tick_every = 1 if max(matrix_masked.shape) <= 20 else TICK_EVERY

    img = plt.imshow(
        matrix_masked,
        aspect="equal",
        interpolation="none",
        cmap=cmap,
        origin="upper", 
        # rasterized=True #rasterized makes the colour one per box
    )
    plt.colorbar(img, label="Fitness")
    # plt.xlabel(X_KEY.capitalize())
    # plt.ylabel(Y_KEY.capitalize())
    major_xticks = np.arange(0, len(x_vals), tick_every)
    major_yticks = np.arange(0, len(y_vals), tick_every)
    plt.xticks(
        major_xticks,
        labels=[
            f"{x_vals[i]:.{DECIMALS}f}" if isinstance(x_vals[i], float) else str(x_vals[i])
            for i in major_xticks
        ]
    )
    plt.yticks(
        major_yticks,
        labels=[
            f"{y_vals[i]:.{DECIMALS}f}" if isinstance(y_vals[i], float) else str(y_vals[i])
            for i in major_yticks
        ]
    )

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.invert_yaxis() 

    # plt.title(f"Map‑Elites heat map: {Y_KEY} vs {X_KEY}")
    plt.tight_layout()
    out_name = f"map_elites_heat_map_{Y_KEY}_vs_{X_KEY}_{SEED_TO_PLOT}.pdf"
    plt.savefig(out_name, format="pdf", bbox_inches="tight")
    print(f"Saved {out_name}")

if __name__ == "__main__":
    main()