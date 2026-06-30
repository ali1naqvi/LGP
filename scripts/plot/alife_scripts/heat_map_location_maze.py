#!/usr/bin/env python3
"""
Coordinate/Fitness heatmap (no Map-Elites, no lambda/decay anywhere).
Reads lines like:  [Heatmap] (x,y) = fitness
and produces a heatmap with the same visual style (grid, colorbar, etc.).
"""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

CMAP = "viridis"
COORD_LINE_RE = re.compile(
    r"\[Heatmap\]\s*\((?P<x>\d+),\s*(?P<y>\d+)\)\s*=\s*(?P<fit>[-+]?\d*\.?\d+(?:[eE][-+]?[\d]+)?)"
)

def heatmap_from_coord_log(log_path: str,
                           out_name: str = "sequential_fitness_heatmap.pdf",
                           origin_zero: bool = True,
                           with_grid: bool = True,
                           cmap_name: str = CMAP) -> None:
    """Parse `[Heatmap] (x,y) = fitness` lines from a log and plot a heatmap."""
    xs, ys, fs = [], [], []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = COORD_LINE_RE.search(line)
            if m:
                xs.append(int(m.group("x")))
                ys.append(int(m.group("y")))
                fs.append(float(m.group("fit")))

    if not xs:
        raise RuntimeError(f"No coordinate + fitness lines found in {log_path}")

    if origin_zero:
        min_x, min_y = min(xs), min(ys)
    else:
        min_x, min_y = 0, 0

    width  = max(xs) - min_x + 1
    height = max(ys) - min_y + 1

    mat = np.full((height, width), np.nan, dtype=float)
    for x, y, f in zip(xs, ys, fs):
        mat[y - min_y, x - min_x] = f

    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap(cmap_name).copy()
    matrix_masked = np.ma.masked_invalid(mat)
    img = plt.imshow(
        matrix_masked,
        aspect="equal",
        interpolation="nearest",           # no blending between cells
        cmap=cmap,
        origin="lower",
        extent=[-0.5, width - 0.5, -0.5, height - 0.5],  # align cell edges to integer grid lines
        rasterized=True                    # force rasterization in vector outputs (PDF/SVG)
    )
    plt.colorbar(img, label="Fitness")

    ax = plt.gca()
    # Label only every 10th cell (0,10,20,...) like the PNG version
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    if with_grid:
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.3, linestyle="-")
    #ax.invert_yaxis()

    img.set_rasterized(True)
    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight", dpi=300)
    print(f"Saved coordinate heatmap to {out_name}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a heatmap from coordinate/fitness logs (no Map-Elites)."
    )
    parser.add_argument("log", type=str,
                        help="Path to a log containing lines like: [Heatmap] (x,y) = fitness")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file name (pdf/png)")
    parser.add_argument("--no-origin-zero", dest="origin_zero", action="store_false",
                        help="Keep original (x,y) coordinates instead of rebasing to (0,0)")
    parser.add_argument("--no-grid", dest="with_grid", action="store_false",
                        help="Disable thin pixel grid lines")
    args = parser.parse_args()

    out = args.out or (os.path.splitext(os.path.basename(args.log))[0] + "_coord_heatmap.pdf")
    heatmap_from_coord_log(
        args.log,
        out_name=out,
        origin_zero=args.origin_zero,
        with_grid=args.with_grid
    )

if __name__ == "__main__":
    main()