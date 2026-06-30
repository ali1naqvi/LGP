import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")  # clean grid‑based style
import os
import glob

generations = 3000

# experiment_name_1 = "7_reacher_baseline_2"
# experiment_name_2 = "7_reacher_baseline_10"
# experiment_name_3 = "7_reacher_baseline_17"
# experiment_name_4 = "7_reacher_dyn_no_rc_fitness_tot_reg"
# experiment_name_5 = "7_reacher_dyn_rc_fitness_tot_regs"
# experiment_name_6 = "7_reacher_dyn_start_fitness_tot_regs"

experiment_name_1 = "7_reacher_baseline_2"
experiment_name_2 = "7_reacher_baseline_10"
experiment_name_3 = "7_reacher_baseline_17"
experiment_name_4 = "static_starting/7_reacher_dyn_rc_fitness_tot_regs"
experiment_name_5 = "static_starting/7_reacher_dyn_no_rc_fitness_tot_reg"

# experiment_name_1 = "5_ant_baseline_8"
# experiment_name_2 = "5_ant_baseline_27"
# experiment_name_3 = "5_ant_baseline_40"
# experiment_name_4 = "ant_dynamic_start/5_ant_dynamic_fitness_tot-reg"
# experiment_name_5 = "static_starting/5_ant_dynamic_1_fitness_tot-reg"
# experiment_name_6 = "ant_dynamic_no_register_clone/5_ant_dynamic_fitness_tot-reg"


# experiment_name_1 = "5_ant_baseline_8"
# experiment_name_2 = "5_ant_baseline_27"
# experiment_name_3 = "5_ant_baseline_40"
# experiment_name_4 = "static_starting/5_ant_dynamic_1_fitness_tot-reg"
# experiment_name_5 = "static_starting/5_ant_dynamic_no_gc"


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
        # df = pd.read_csv(f, usecols=["validation_fitness"])

        # # Coerce to numeric, then keep only finite, non-zero validation entries.
        # # We treat each non-zero entry as a "validation generation" and ignore zeros.
        # vals = pd.to_numeric(df["validation_fitness"], errors="coerce").to_numpy(dtype=float)


        df = pd.read_csv(
            f,
            usecols=[
                "validation_fitness",
                "best_agent_effective_register_size",
                "best_agent_register_size",
            ],
        )

        validation_vals = pd.to_numeric(df["validation_fitness"], errors="coerce").to_numpy(dtype=float)
        eff_regs = pd.to_numeric(df["best_agent_effective_register_size"], errors="coerce").to_numpy(dtype=float)
        total_regs = pd.to_numeric(df["best_agent_register_size"], errors="coerce").to_numpy(dtype=float)

        # vals = np.divide(
        #     eff_regs,
        #     total_regs,
        #     out=np.full_like(eff_regs, np.nan, dtype=float),
        #     where=total_regs != 0,
        # )

        vals = df["validation_fitness"].values

        mask = np.isfinite(validation_vals) & (validation_vals != 0.0) & np.isfinite(vals)
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
    # Seeds may have different numbers of validation entries after removing zeros.
    # Align by validation index and pad with NaN so we can compute robust stats.
    max_len = max(len(r) for r in reps)
    data = np.full((len(reps), max_len), np.nan, dtype=float)
    for i, r in enumerate(reps):
        data[i, :len(r)] = r

    gens = np.arange(max_len)  # counts non-zero validations as "generations"

    medians = np.nanmean(data, axis=0)
    q25 = np.nanpercentile(data, 25, axis=0)
    q75 = np.nanpercentile(data, 75, axis=0)
    return gens, medians, q25, q75

def AddToPlot(gens, median, q25, q75, label, color=None):
    ax = plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    ax.fill_between(gens, q25, q75, alpha=0.15, color=color, zorder=1)
    ax.plot(gens, median, label=label, linewidth=2.0, color=color, zorder=2)

if __name__ == "__main__":
    gens1, med1, q25_1, q75_1 = load_best_fitness_reps(exp_dir_1)
    gens2, med2, q25_2, q75_2 = load_best_fitness_reps(exp_dir_2)
    gens3, med3, q25_3, q75_3 = load_best_fitness_reps(exp_dir_3)
    gens4, med4, q25_4, q75_4 = load_best_fitness_reps(exp_dir_4)
    gens5, med5, q25_5, q75_5 = load_best_fitness_reps(exp_dir_5)
    # gens6, med6, q25_6, q75_6 = load_best_fitness_reps(exp_dir_6)

    plt.figure(figsize=(6,4))
    AddToPlot(gens1, med1, q25_1, q75_1, "Static-Low", color='tab:green')
    AddToPlot(gens2, med2, q25_2, q75_2, "Static-Medium", color='tab:blue')
    AddToPlot(gens3, med3, q25_3, q75_3, "Static-High", color='tab:orange')
    AddToPlot(gens4, med4, q25_4, q75_4, "Dynamic-RD", color='tab:red')
    AddToPlot(gens5, med5, q25_5, q75_5, "Dynamic", color='teal')
    # AddToPlot(gens6, med6, q25_6, q75_6, "No RC", color='teal')
    
    ax = plt.gca()
    ax.grid(which="both", color="lightgray", linewidth=0.5, alpha=0.6)

    leg = plt.legend(
    fontsize=14,
    # loc="lower right",
    frameon=True,          # ensure the frame is drawn
    fancybox=False
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)       # fully opaque
    leg.set_zorder(100)        # draw above lines
            #    , bbox_to_anchor=(1.015, 0.35))

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig("validation_fitness.pdf", format="pdf", bbox_inches="tight")
