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
# experiment_name_4 = "7_reacher_dyn_skew"
# experiment_name_5 = "7_reacher_dyn_no_gc_skew"
# experiment_name_6 = "7_reacher_dyn_skew_start"

# experiment_name_1 = "6_cheetah_baseline_6"
# experiment_name_2 = "6_cheetah_baseline_17"
# experiment_name_3 = "6_cheetah_baseline_28"
# experiment_name_4 = "6_cheetah_dyn_range_skew"
# experiment_name_5 = "6_cheetah_dyn_range_no_gc_skew"
# experiment_name_6 = "6_cheetah_dyn_range_start"

experiment_name_1 = "ant_baseline/5_ant_baseline_8_fitness"
experiment_name_2 = "ant_baseline/5_ant_baseline_27_fitness"
experiment_name_3 = "ant_baseline/5_ant_baseline_40_fitness"
experiment_name_4 = "ant_dynamic_start/5_ant_dynamic_fitness_tot-reg"
experiment_name_5 = "ant_dynamic_register_clone/5_ant_dynamic_fitness_tot-reg"
experiment_name_6 = "ant_dynamic_no_register_clone/5_ant_dynamic_fitness_tot-reg"

# experiment_name_1 = "6_cheetah_hurdles_baseline_6"
# experiment_name_2 = "6_cheetah_hurdles_baseline_17"
# experiment_name_3 = "6_cheetah_hurdles_baseline_28"
# experiment_name_4 = "6_cheetah_hurdles_dynamic"
# experiment_name_5 = "6_cheetah_hurdles_dynamic_no_gc"
# experiment_name_6 = "6_cheetah_hurdles_dynamic_start"

# experiment_name_1 = "8_humanoid_baseline_17"
# experiment_name_2 = "8_humanoid_baseline_348"
# experiment_name_3 = "8_humanoid_baseline_365"
# experiment_name_4 = "8_humanoid_dyn"
# experiment_name_5 = "8_humanoid_dyn_no_gc"
# experiment_name_6 = "8_humanoid_dyn_start"


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))
print(base_path)
exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")
exp_dir_6 = os.path.join(base_path, experiment_name_6, "logs", "selection")
# print(exp_dir_3)
#best_fitness, program_instruction_count,effective_program_instruction_count, best_agent_register_size
def load_best_fitness_reps(exp_dir):
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    reps = []
    for f in files:
        df = pd.read_csv(f, usecols=["best_agent_effective_register_size"])
        df2 = pd.read_csv(f, usecols=["best_agent_register_size"])
        vals = df["best_agent_effective_register_size"].values / df2["best_agent_register_size"].values 
        # divide 
        if len(vals) == 0:
            continue
        # Pad or truncate to exactly 250 generations
        if len(vals) < generations:
            pad_val = vals[-1]
            pad = np.full(generations - len(vals), pad_val)
            vals = np.concatenate([vals, pad])
        else:
            vals = vals[:generations]
        reps.append(vals)
    if not reps:
        return None
    data = np.vstack(reps)  # shape: (num_seeds, 250)
    gens = np.arange(generations)
    medians = np.median(data, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
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
    gens6, med6, q25_6, q75_6 = load_best_fitness_reps(exp_dir_6)

    plt.figure(figsize=(8,6))
    AddToPlot(gens1, med1, q25_1, q75_1, "base 17", color='tab:green')
    AddToPlot(gens2, med2, q25_2, q75_2, "base 348", color='tab:blue')
    AddToPlot(gens3, med3, q25_3, q75_3, "base 365", color='tab:orange')
    AddToPlot(gens4, med4, q25_4, q75_4, "dynamic", color='tab:pink')
    AddToPlot(gens5, med5, q25_5, q75_5, "dynamic no gc", color='tab:red')
    AddToPlot(gens6, med6, q25_6, q75_6, "dynamic start", color='teal')

    # final_mean5 = med5[-1]
    # final_mean6 = med5[-1]

    # final_mean2 = np.median(final_mean2)
    # final_mean4 = np.median(final_mean4)

    #pct_diff = ((final_mean4 - final_mean2) / np.mean([final_mean2, final_mean4])) * 100
    # print(f"Final mean of {experiment_name_1}: {final_mean1:.4f}")
    # print(f"Final mean of {experiment_name_2}: {final_mean2:.4f}")
    # print(f"Final mean of {experiment_name_3}: {final_mean3:.4f}")
    #print(f"Final mean of {experiment_name_4}: {final_mean4:.4f}")
    #sign = 'higher' if pct_diff >= 0 else 'lower'
    #print(f"{experiment_name_4} is {abs(pct_diff):.2f}% {sign} than {experiment_name_2}")
    # 52.95% neuro stateful in evolution
    # 29.32% neuro stateless in evolution
    
    #IMM
    # 36.54%% neuro stateful in evolution
    # 55.30% neuro stateless in evolution
    #29-55% through evolution and final solutions with final solutions having 
    ax = plt.gca()
    ax.grid(which="both", color="lightgray", linewidth=0.5, alpha=0.6)

    leg = plt.legend(
    fontsize=14,
    loc="lower right",
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

    plt.savefig("proportion.pdf", format="pdf", bbox_inches="tight")

    
