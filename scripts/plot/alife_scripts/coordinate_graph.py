import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use("seaborn-v0_8-whitegrid")

import pathlib
import numpy as np

base_colors = ['tab:blue', 'tab:orange']

def lighten_color(color, amount: float = 0.5):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - amount) * rgb + amount)

EPISODE_LEN = 1000
NUM_EPISODES = 20
SMOOTH_WINDOW = 1
DIRECT_TO_LAST = False
SHOW_LEG_BREAK = False
LEG_BREAK_MARKER_SIZE = 1

def load_coords(path: pathlib.Path):
    coords, break_idx = [], []
    idx = -1
    with open(path, "r") as f:
        for line in f:
            # extract coordinate if present
            m = re.search(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", line)
            if m:
                coords.append((float(m.group(1)), float(m.group(2))))
                idx += 1
            # detect leg‑break message
            if "Leg break" in line and idx >= 0:
                break_idx.append(idx)
    return coords, break_idx

def smooth(coords, k: int):
    if k <= 1 or len(coords) < 3:
        return coords
    xs, ys = zip(*coords)
    w = np.ones(k) / k
    xs_s = np.convolve(xs, w, mode="same")
    ys_s = np.convolve(ys, w, mode="same")
    return list(zip(xs_s, ys_s))

def smoothed_break_points(coords, break_idx, k: int = SMOOTH_WINDOW, offset: int = 1):
    pts = []
    for idx in break_idx:
        target = idx + offset
        if target >= len(coords):
            continue 
        start = (target // EPISODE_LEN) * EPISODE_LEN
        segment = coords[start:start + EPISODE_LEN]
        if k > 1:
            segment = smooth(segment, k)
        local = target - start
        if 0 <= local < len(segment):
            pts.append(segment[local])
    return pts

script_dir = pathlib.Path(__file__).resolve().parent
coords,  breaks  = load_coords("ant_break_transferred.txt")
coords2, breaks2 = load_coords("ant_break_train.txt")
# Generate smoothed break‑point lists
break_pts  = smoothed_break_points(coords,  breaks)
break_pts2 = smoothed_break_points(coords2, breaks2)

def plot_segments(ax, coords, color, label=None):
    total_steps = EPISODE_LEN * NUM_EPISODES
    for start in range(0, min(len(coords), total_steps), EPISODE_LEN):
        segment = coords[start:start + EPISODE_LEN]
        if not segment:
            continue
        if DIRECT_TO_LAST:
            last_x, last_y = segment[-1]
            ax.plot([0, last_x], [0, last_y], color=color, linewidth=0.5)
            continue
        if SMOOTH_WINDOW > 1:
            segment = smooth(segment, SMOOTH_WINDOW)
        if len(segment) < 2:
            continue
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color=color, linewidth=1.0)
    if label is not None:
        ax.plot([], [], color=color, linewidth=1.0, label=label)

fig, ax = plt.subplots(figsize=(6, 4))

ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.set_axisbelow(False)
for spine in ax.spines.values():
    spine.set_zorder(5)
ax.xaxis.set_zorder(5)
ax.yaxis.set_zorder(5)

plot_segments(ax, coords,  lighten_color(base_colors[0], 0.2), label="LGP Dynamic 0.2 Trained")
plot_segments(ax, coords2, lighten_color(base_colors[1], 0.2), label="LGP Dynamic 0.2 Transferred")

if SHOW_LEG_BREAK:
    for pts in (break_pts, break_pts2):
        if pts:
            bx, by = zip(*pts)
            ax.scatter(bx, by, color="red", s=LEG_BREAK_MARKER_SIZE,
                       zorder=6)

ax.scatter([0], [0], color="black", zorder=3)
ax.legend(loc="upper right", frameon=True, fontsize=11)
ax.set_aspect("equal", "box")
all_coords = coords + coords2
pad = 0.05 * max(abs(c) for xy in all_coords for c in xy)
ax.set_xlim(min(x for x, _ in all_coords) - pad, max(x for x, _ in all_coords) + pad)
ax.set_ylim(min(y for _, y in all_coords) - pad, max(y for _, y in all_coords) + pad)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
plt.savefig("coordinate_graph.pdf")
