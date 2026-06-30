#!/usr/bin/env python3
# scripts/plot/alife_scripts/map_elites_animation.py
# Create an MP4 animation (with GIF fallback) of Map-Elites heatmaps across generations.
import re, os, glob, sys, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Paths ----------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))
EXPERIMENT_DIR = os.path.join(base_path, "31_td_sf_me_static_1")   # change if needed
logs_dir = os.path.join(EXPERIMENT_DIR, "logs", "map_elites")

# ---------- Visualization parameters ----------
TICK_EVERY   = 20
MIN_VAL      = None   # set to None to auto-scale
MAX_VAL      = None   # set to None to auto-scale
DECIMALS     = 3
CMAP         = "viridis"
GRID_SIZE    = 100   # for lambda/decay grid (10x10, 100 cells)

X_KEY        = "decay"   # "bin", "lambda", "decay", "plast"
Y_KEY        = "lambda"  # "seed", "bin", "lambda", "decay", "plast"

SEED_TO_PLOT = 8         # which seed to visualize (matches filename pattern)
START_GEN    = 0         # first generation to include
END_GEN      = 2500     # last generation to include (inclusive)

FPS          = 30
BITRATE      = 1800
DPI          = 180

# How to set the color scale across frames:
#  - "dynamic": rescale vmin/vmax every frame (best to avoid clipping, not comparable across frames)
#  - "global": use the min/max across ALL generations (comparable across frames, no clipping)
#  - "first":  keep the first frame’s scale (current behavior)
SCALE_MODE   = "dynamic"   # choose: "dynamic" | "global" | "first"

# ---------- File naming pattern ----------
# Files are assumed to be named like: map_elites_archive_{SEED}_t{GEN}.csv
# Example: map_elites_archive_8_t2027.csv
FNAME_REGEX = re.compile(rf"^map_elites_archive_{SEED_TO_PLOT}_t(?P<gen>\d+)\.csv$")

# ---------- CSV line parsing ----------
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
            if "fit" in record:           # keep lines we successfully parsed
                entries.append(record)
    return entries

def build_matrix(records: list[dict], x_key: str, y_key: str):
    # Special handling for lambda/decay on a fixed GRID_SIZE
    if {"lambda", "decay"} >= {x_key, y_key}:
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

    # General case for other axes
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

def find_generation_files():
    # Only consider files for the selected seed, then sort by generation
    patt = os.path.join(logs_dir, f"map_elites_archive_{SEED_TO_PLOT}_t*.csv")
    candidates = glob.glob(patt)
    items = []
    for f in candidates:
        name = os.path.basename(f)
        m = FNAME_REGEX.match(name)
        if not m:
            continue
        gen = int(m.group("gen"))
        if START_GEN <= gen <= END_GEN:
            items.append((gen, f))
    items.sort(key=lambda x: x[0])
    return items

def compute_global_minmax(gen_files):
    gmin, gmax = np.inf, -np.inf
    for _, path in gen_files:
        recs = load_one_archive(path)
        if not recs:
            continue
        _, _, mat = build_matrix(recs, X_KEY, Y_KEY)
        arr = np.asarray(mat, dtype=float)
        # ignore NaNs when computing extrema
        if np.isfinite(np.nanmin(arr)):
            gmin = min(gmin, np.nanmin(arr))
        if np.isfinite(np.nanmax(arr)):
            gmax = max(gmax, np.nanmax(arr))
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
        # sensible fallback
        gmin, gmax = 0.0, 1.0
    return gmin, gmax

def main():
    if not os.path.isdir(logs_dir):
        sys.exit(f"Logs directory not found: {logs_dir}")

    gen_files = find_generation_files()
    if not gen_files:
        sys.exit(f"No generation files found for seed {SEED_TO_PLOT} in {logs_dir} (range {START_GEN}-{END_GEN}).")

    # Optionally pre-compute global min/max for consistent scale across frames
    vmin_global = vmax_global = None
    if SCALE_MODE == "global" and MIN_VAL is None and MAX_VAL is None:
        vmin_global, vmax_global = compute_global_minmax(gen_files)

    # ---------- Initialize with the first generation ----------
    first_gen, first_path = gen_files[0]
    first_records = load_one_archive(first_path)
    if not first_records:
        sys.exit(f"First file had no records: {first_path}")

    x_vals, y_vals, matrix = build_matrix(first_records, X_KEY, Y_KEY)
    matrix_masked = np.ma.masked_invalid(matrix)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    if MIN_VAL is not None or MAX_VAL is not None:
        vmin = MIN_VAL if MIN_VAL is not None else np.nanmin(matrix_masked)
        vmax = MAX_VAL if MAX_VAL is not None else np.nanmax(matrix_masked)
    elif SCALE_MODE == "global" and vmin_global is not None:
        vmin, vmax = vmin_global, vmax_global
    else:  # "first" or "dynamic" start with the first frame’s range
        vmin = np.nanmin(matrix_masked)
        vmax = np.nanmax(matrix_masked)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    img = ax.imshow(
        matrix_masked,
        aspect="equal",
        interpolation="none",
        cmap=plt.get_cmap(CMAP),
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        rasterized=False,
    )
    cb = plt.colorbar(img, ax=ax, label="Fitness")

    # Ticks / labels
    tick_every = 1 if max(matrix_masked.shape) <= 20 else TICK_EVERY
    major_xticks = np.arange(0, len(x_vals), tick_every)
    major_yticks = np.arange(0, len(y_vals), tick_every)
    ax.set_xticks(
        major_xticks,
        labels=[
            f"{x_vals[i]:.{DECIMALS}f}" if isinstance(x_vals[i], float) else str(x_vals[i])
            for i in major_xticks
        ]
    )
    ax.set_yticks(
        major_yticks,
        labels=[
            f"{y_vals[i]:.{DECIMALS}f}" if isinstance(y_vals[i], float) else str(y_vals[i])
            for i in major_yticks
        ]
    )
    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.invert_yaxis()

    # Overlay text for generation counter
    gen_text = ax.text(0.02, 0.98, f"Generation {first_gen}", transform=ax.transAxes,
                       va="top", ha="left", fontsize=12, bbox=dict(boxstyle="round", alpha=0.3))

    plt.tight_layout()

    # ---------- Frame update ----------
    def update(frame_idx):
        gen, path = gen_files[frame_idx]
        recs = load_one_archive(path)
        _, _, mat = build_matrix(recs, X_KEY, Y_KEY)
        frame_masked = np.ma.masked_invalid(mat)
        img.set_data(frame_masked)
        # If dynamic scaling is requested (and user didn’t force MIN/MAX), update clim each frame
        if SCALE_MODE == "dynamic" and MIN_VAL is None and MAX_VAL is None:
            fmin = np.nanmin(frame_masked)
            fmax = np.nanmax(frame_masked)
            if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin == fmax:
                # keep previous scale or use a tiny epsilon range
                fmin, fmax = img.get_clim()
                if fmin == fmax:
                    fmax = fmin + 1e-9
            img.set_clim(vmin=fmin, vmax=fmax)
            # ensure colorbar reflects the new normalization
            cb.update_normal(img)
        gen_text.set_text(f"Generation {gen}")
        # Optional: print progress every 100 frames
        if gen % 100 == 0:
            print(f"Rendered generation {gen}")
        return [img, gen_text]

    anim = FuncAnimation(fig, update, frames=len(gen_files), interval=1000 / FPS, blit=False)

    # Try saving MP4 via FFMpegWriter; fall back to GIF if ffmpeg is unavailable.
    out_base = f"map_elites_animation_seed{SEED_TO_PLOT}_{Y_KEY}_vs_{X_KEY}_{START_GEN}-{END_GEN}"
    mp4_name = out_base + ".mp4"
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=FPS, bitrate=BITRATE)
        anim.save(mp4_name, writer=writer, dpi=DPI)
        print(f"Saved {mp4_name}")
    except Exception as e:
        print(f"FFmpeg not available or failed ({e}). Falling back to GIF.")
        from matplotlib.animation import PillowWriter
        gif_name = out_base + ".gif"
        anim.save(gif_name, writer=PillowWriter(fps=FPS))
        print(f"Saved {gif_name}")

if __name__ == "__main__":
    main()