import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")  # clean grid‑based style
import os
import glob
from brokenaxes import brokenaxes
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Set the two y-axis ranges to keep.
# First tuple = upper section, second tuple = lower section.
# Example: ((100, 150), (0, 20))
y_axis_ranges = ((-150, -120),(-15, -5))
y_axis_height_ratios = [2.5, 1]

generations = 3000

experiment_name_1 = "7_reacher_baseline_2"
experiment_name_2 = "7_reacher_baseline_10"
experiment_name_3 = "7_reacher_baseline_17"
experiment_name_4 = "static_starting/7_reacher_dyn_rc_fitness_tot_regs"
experiment_name_5 = "static_starting/7_reacher_dyn_no_rc_fitness_tot_reg"

# experiment_name_1 = "5_ant_baseline_8"
# experiment_name_2 = "5_ant_baseline_27"
# experiment_name_3 = "5_ant_baseline_40"
# experiment_name_4 = "ant_dynamic_start/5_ant_dynamic_fitness_tot-reg"
# experiment_name_5 = "5_ant_dynamic_1_fitness_tot-reg"
# experiment_name_6 = "ant_dynamic_no_register_clone/5_ant_dynamic_fitness_tot-reg"

# experiment_name_5 = "ant_dynamic_register_clone/5_ant_dynamic_fitness_tot-reg"

# experiment_name_1 = "6_cheetah_baseline_6"
# experiment_name_2 = "6_cheetah_baseline_17"
# experiment_name_3 = "6_cheetah_baseline_28"
# experiment_name_4 = "6_cheetah_dyn_rc_fitness_tot-reg"
# experiment_name_5 = "6_cheetah_dyn_no_rc_fitness_tot-reg"
# experiment_name_6 = "6_cheetah_dyn_start_fitness_tot-reg"



base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))
print(base_path)
exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")
# exp_dir_6 = os.path.join(base_path, experiment_name_6, "logs", "selection")
# print(exp_dir_3)
#best_fitness, validation_fitness, program_instruction_count,effective_program_instruction_count, best_agent_register_size, avg_complexity_front_0
def load_best_fitness_reps(exp_dir):
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    reps = []
    for f in files:
        df = pd.read_csv(f, usecols=["validation_fitness"])

        # Coerce to numeric, then keep only finite, non-zero validation entries.
        # We treat each non-zero entry as a "validation generation" and ignore zeros.
        vals = pd.to_numeric(df["validation_fitness"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & (vals != 0.0)
        vals = vals[mask]

        # Skip seeds that contain no usable values.
        if vals.size == 0:
            continue

        # Truncate if a hard cap is desired.
        if vals.size > generations:
            vals = vals[:generations]

        reps.append(vals)
    if not reps:
        return None

    max_len = max(len(r) for r in reps)
    data = np.full((len(reps), max_len), np.nan, dtype=float)
    for i, r in enumerate(reps):
        data[i, :len(r)] = r

    gens = np.arange(max_len)  # counts non-zero validations as "generations"

    medians = np.nanmean(data, axis=0)
    q25 = np.nanpercentile(data, 25, axis=0)
    q75 = np.nanpercentile(data, 75, axis=0)
    return gens, medians, q25, q75

def AddToPlot(gens, median, q25, q75, label, color=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if color is None:
        try:
            color = next(ax._get_lines.prop_cycler)['color']
        except AttributeError:
            color = None

    if hasattr(ax, 'axs'):
        for subax in ax.axs:
            subax.fill_between(gens, q25, q75, alpha=0.15, color=color, zorder=1)
            subax.plot(gens, median, label=label, linewidth=2.0, color=color, zorder=2)
    else:
        ax.fill_between(gens, q25, q75, alpha=0.15, color=color, zorder=1)
        ax.plot(gens, median, label=label, linewidth=2.0, color=color, zorder=2)

if __name__ == "__main__":
    gens1, med1, q25_1, q75_1 = load_best_fitness_reps(exp_dir_1)
    gens2, med2, q25_2, q75_2 = load_best_fitness_reps(exp_dir_2)
    gens3, med3, q25_3, q75_3 = load_best_fitness_reps(exp_dir_3)
    gens4, med4, q25_4, q75_4 = load_best_fitness_reps(exp_dir_4)
    gens5, med5, q25_5, q75_5 = load_best_fitness_reps(exp_dir_5)
    # gens6, med6, q25_6, q75_6 = load_best_fitness_reps(exp_dir_6)

    fig = plt.figure(figsize=(6,4))
    bax = brokenaxes(
        ylims=y_axis_ranges,
        height_ratios=y_axis_height_ratios,
        hspace=.05,
        fig=fig,
        despine=False,
    )
    AddToPlot(gens1, med1, q25_1, q75_1, "Register Low", color='tab:green', ax=bax)
    AddToPlot(gens2, med2, q25_2, q75_2, "27 Registers", color='tab:blue', ax=bax)
    AddToPlot(gens3, med3, q25_3, q75_3, "40 Registers", color='tab:orange', ax=bax)
    AddToPlot(gens4, med4, q25_4, q75_4, "RC", color='tab:pink', ax=bax)
    AddToPlot(gens5, med5, q25_5, q75_5, "No RC", color='teal', ax=bax)
    # AddToPlot(gens6, med6, q25_6, q75_6, "No RC", color='teal', ax=bax)
    
    # ax = plt.gca()
    # ax.grid(which="both", color="lightgray", linewidth=0.5, alpha=0.6)

    # leg = plt.legend(
    # fontsize=14,
    # # loc="lower right",
    # frameon=True,          # ensure the frame is drawn
    # fancybox=False
    # )
    # frame = leg.get_frame()
    # frame.set_facecolor("white")
    # frame.set_alpha(1.0)       # fully opaque
    # leg.set_zorder(100)        # draw above lines
    #         #    , bbox_to_anchor=(1.015, 0.35))

    # Keep the top y-axis visible and avoid overcrowded ticks/grid.
    for i, ax in enumerate(bax.axs):
        ax.tick_params(axis='y', labelsize=16, pad=6)
        ax.grid(axis='y', color='lightgray', linewidth=0.6, alpha=0.6)
        ax.xaxis.set_major_locator(MultipleLocator(10))

        if i == 0:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
        elif i == len(bax.axs) - 1:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        if i < len(bax.axs) - 1:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax.tick_params(axis='x', labelsize=16)

    plt.savefig("validation_fitness.pdf", format="pdf", bbox_inches="tight")
