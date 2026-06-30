import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")  # clean grid‑based style
import os
import glob

generations = 10000

experiment_name_1 = "gradient_test_original"
experiment_name_2 = "gradient_test_static_constants"
experiment_name_3 = "gradient_test_lamark"
experiment_name_4 = "gradient_test_baldwin"
# experiment_name_5 = "gradient_test_10k"

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))
exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
# exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")
# exp_dir_6 = os.path.join(base_path, experiment_name_6, "logs", "selection")
# print(exp_dir_3)
#best_fitness, validation_fitness, program_instruction_count,effective_program_instruction_count, best_agent_register_size, avg_complexity_front_0
def load_best_fitness_reps(exp_dir):
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    reps = []
    for f in files:
        df = pd.read_csv(f, usecols=["best_fitness"])
        vals = df["best_fitness"].values
        
        # df = pd.read_csv(f, usecols=["best_agent_effective_register_size"])
        # df2 = pd.read_csv(f, usecols=["best_agent_register_size"])
        # vals = df["best_agent_effective_register_size"].values / df2["best_agent_register_size"].values 

        # df = pd.read_csv(f, usecols=["effective_program_instruction_count"])
        # df2 = pd.read_csv(f, usecols=["program_instruction_count"])
        # vals = df["effective_program_instruction_count"].values / df2["program_instruction_count"].values 
        
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
    medians = np.mean(data, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    return gens, medians, q25, q75

def AddToPlot(gens, median, q25, q75, label, color=None):
    ax = plt.gca()
    line, = ax.plot(gens, median, label=label, linewidth=2.0, color=color, zorder=2)
    line_color = line.get_color()
    ax.fill_between(gens, q25, q75, alpha=0.10, color=line_color, zorder=1)

if __name__ == "__main__":
    result1 = load_best_fitness_reps(exp_dir_1)
    result2 = load_best_fitness_reps(exp_dir_2)
    result3 = load_best_fitness_reps(exp_dir_3)
    result4 = load_best_fitness_reps(exp_dir_4)
    # result5 = load_best_fitness_reps(exp_dir_5)
    # result6 = load_best_fitness_reps(exp_dir_6)

    gens1, med1, q25_1, q75_1 = result1
    gens2, med2, q25_2, q75_2 = result2
    gens3, med3, q25_3, q75_3 = result3
    gens4, med4, q25_4, q75_4 = result4
    # gens5, med5, q25_5, q75_5 = result5
    # gens6, med6, q25_6, q75_6 = result6

    plt.figure(figsize=(6,4))

    AddToPlot(gens1, med1, q25_1, q75_1, "OG", color='tab:green')
    AddToPlot(gens2, med2, q25_2, q75_2, "STATIC CONSTANTS", color='tab:blue')
    AddToPlot(gens3, med3, q25_3, q75_3, "lamarkism", color='tab:orange')
    AddToPlot(gens4, med4, q25_4, q75_4, "BALDWIN", color='tab:red')
    # AddToPlot(gens5, med5, q25_5, q75_5, "Memory with Reward Difference", color='teal')
    # AddToPlot(gens6, med6, q25_6, q75_6, "No RC", color='teal')
    
    ax = plt.gca()
    ax.grid(which="both", color="lightgray", linewidth=0.5, alpha=0.6)

    leg = plt.legend(
    fontsize=16,
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

    plt.savefig("results.pdf", format="pdf", bbox_inches="tight")
