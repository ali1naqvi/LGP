import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter, MultipleLocator
plt.style.use("seaborn-v0_8-whitegrid")  # clean grid‑based style
import os
import glob

# experiment_name_4 = "40_static"
experiment_name_3 = "40_td_static"
experiment_name_2 = "40_static_rew_obs"
experiment_name_1 = "40_static_rew_obs_neuro"

# experiment_name_4 = "41_dynamic"
# experiment_name_3 = "41_dynamic_neuro"
# experiment_name_2 = "41_dynamic_rew_obs"
# experiment_name_1 = "41_dynamic_rew_obs_neuro"

# experiment_name_4 = "41_dynamic"
# experiment_name_3 = "41_td_dynamic"
# experiment_name_2 = "41_dynamic_rew_obs"
# experiment_name_1 = "41_dynamic_rew_obs_neuro"

# experiment_name_1 = "36_td_sf_me_static"
# experiment_name_2 = "36_td_sf_static"
# experiment_name_3 = "35_baseline_ntpg_sf_maze"
# experiment_name_4 = "35_baseline_tpg_sf_maze"

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "tpg", "experiments"))

exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
# exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
# exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")

#best_fitness, program_instruction_count,effective_program_instruction_count
def load_seed_final_fitness(exp_dir, target_gen=5000):
    """
    Return one scalar per seed = best_fitness from generation `target_gen` (default 250)
    if present; otherwise from the last available generation.
    (Skip empty seeds that logged nothing.)
    """
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files   = sorted(glob.glob(pattern))
    finals  = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) == 0:
            continue
        # Prefer the exact target generation if a generation column is present
        if "generation" in df.columns:
            exact = df[df["generation"] == target_gen]
            if not exact.empty:
                finals.append(exact["effective_program_instruction_count"].iloc[-1])
                continue
        # Fallback: use the last logged generation's best_fitness
        finals.append(df["effective_program_instruction_count"].iloc[-1])
    return np.array(finals)


if __name__ == "__main__":
    vals1 = load_seed_final_fitness(exp_dir_1)
    vals2 = load_seed_final_fitness(exp_dir_2)
    vals3 = load_seed_final_fitness(exp_dir_3)
    # vals4 = load_seed_final_fitness(exp_dir_4)

    data   = [vals1, vals2, vals3]
    labels = ["NeuroTPG + Elig",
              "NeuroTPG + Rew",
              "TPG + Elig"]
    # labels = ["NeuroTPG + Rew",
    #           "TPG + Rew",
    #           "NeuroTPG", 
    #           "TPG"]

    colors = [ 'tab:pink', 'tab:green', 'tab:red']
    # Lighten colors for better aesthetics
    light_colors = [to_rgba(c, alpha=0.5) for c in colors]

    plt.figure(figsize=(8, 6))
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # Match box-plot-testing style: y-axis tick labels and tick spacing
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=22)

    # Compute a sensible y-range based on data (start a bit below zero)
    vals_all = np.concatenate([v for v in data if v.size > 0])
    ymin = 10
    ymax = 140
    ax.set_ylim(ymin, ymax)

    # Major ticks every 20 (requested) and minor every 10
    ax.yaxis.set_major_locator(MultipleLocator(20))
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
    plt.legend(handles=handles, loc="upper right", fontsize=18)
    plt.tight_layout()
    plt.savefig("lets_try.pdf", format="pdf", bbox_inches="tight")
    plt.show()