import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter, MultipleLocator
plt.style.use("seaborn-v0_8-whitegrid")  # clean grid‑based style
import os
import glob
import re

# experiment_name_1 = "neuro_imm_stateful"
# experiment_name_2 = "neuro_imm_stateless"
# experiment_name_3 = "imm_stateful"
# experiment_name_4 = "1_imm_stateless"

# experiment_name_1 = "ant_neuro_stateful"
# experiment_name_2 = "ant_neuro_stateless"
# experiment_name_3 = "ant_stateful"
# experiment_name_4 = "1_ant_stateless"

# experiment_name_1 = "36_td_sf_me_dynamic"
# experiment_name_2 = "36_td_sf_dynamic"
# experiment_name_3 = "35_baseline_ntpg_sf_maze_dyn"
# experiment_name_4 = "35_baseline_tpg_sf_maze_dyn"

experiment_name_4 = "40_static"
experiment_name_3 = "40_static_neuro"
experiment_name_2 = "40_static_rew_obs"
experiment_name_1 = "40_static_rew_obs_neuro"

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "tpg", "experiments"))

exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "misc")
print(exp_dir_1)
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "misc")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "misc")
exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "misc")
# exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")

def load_seed_medians_from_replays(exp_dir):
    pattern = os.path.join(exp_dir, "tpg.*.42.replay.std")
    files   = sorted(glob.glob(pattern))
    medians = []
    for f in files:
        try:
            with open(f, 'r') as fh:
                # Scan all lines to find the first occurrence of 'mean' or 'median' followed by a number
                for line in fh:
                    m = re.search(r"\b(?:mean|median)\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", line, flags=re.IGNORECASE)
                    if m:
                        medians.append(float(m.group(1)))
                        break
        except Exception:
            # Skip malformed or unreadable files
            continue
    return np.array(medians)


if __name__ == "__main__":
    vals1 = load_seed_medians_from_replays(exp_dir_1)
    vals2 = load_seed_medians_from_replays(exp_dir_2)
    vals3 = load_seed_medians_from_replays(exp_dir_3)
    vals4 = load_seed_medians_from_replays(exp_dir_4)

    data   = [vals1, vals2, vals3, vals4]
    # labels = ["NeuroTPG:SF",
    #           "NeuroTPG:SL",
    #           "TPG:SF", 'TPG:SL']
    labels = ["NeuroTPG + TD + ME",
              "NeuroTPG + TD",
              "NeuroTPG", 
              "TPG"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    # Lighten colors for better aesthetics
    light_colors = [to_rgba(c, alpha=0.5) for c in colors]

    plt.figure(figsize=(8, 6))
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax = plt.gca()
    # Make y-axis ticks larger (labels and tick marks)
    ax.tick_params(axis='y', which='major', labelsize=22)
    #ax.tick_params(axis='y', which='minor', labelsize=20, length=6, width=1.5)
    # Force y-axis to start at 480 and ensure a major tick at 480
    vals_all = np.concatenate([v for v in data if v.size > 0])
    ymin = 0
    ymax = max(np.ceil(np.max(vals_all) / 50) * 50, ymin + 100)  # ensure a non-zero visible range
    ax.set_ylim(ymin, ymax)

    # Use 20-point major ticks so 480 is a tick (and thus grid line): 480, 500, 520, ...
    ystep = 10
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    # Minor ticks every 10 for readability
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        medianprops={'color': 'black'},
        flierprops={'marker': 'o', 'markersize': 12}
    )
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        max_val = np.max(vals)
        plt.scatter(i+1, max_val, color=light_colors[i], zorder=3)
    for patch, color in zip(bp['boxes'],  light_colors):
        patch.set_facecolor(color)
    plt.gca().set_xticks([])
    handles = [mpatches.Patch(facecolor=lc, label=label) for lc, label in zip(light_colors, labels)]
    plt.legend(handles=handles, loc="lower left", fontsize=18)
    plt.tight_layout()
    plt.savefig("MR_boxplot_fit.pdf", format="pdf", bbox_inches="tight")
    plt.show()