import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")  # clean grid-based style

validation_interval = 100
generations = 2000

experiment_name_1 = "pendulum_execution_modified_rates"
experiment_name_2 = "pendulum_fixed_rates"
experiment_name_3 = "pendulum_inherited_rates"
# experiment_name_4 = "gradient_test_baldwin"
# experiment_name_5 = "gradient_test_10k"

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "experiments"))
exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
# exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
# exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")
# exp_dir_6 = os.path.join(base_path, experiment_name_6, "logs", "selection")
# print(exp_dir_3)
#best_fitness, validation_fitness, program_instruction_count,effective_program_instruction_count, best_agent_register_size, avg_complexity_front_0
def load_best_fitness_reps(exp_dir):
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No selection logs found matching: {pattern}")

    reps = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["validation_fitness"])
        except pd.errors.EmptyDataError:
            continue
        except ValueError as exc:
            raise ValueError(f"{f} does not contain a validation_fitness column") from exc

        # Validation is logged only when it is performed; zero entries are
        # placeholders and should not count as validation observations.
        vals = pd.to_numeric(df["validation_fitness"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals != 0.0)]

        # Skip seeds that contain no usable values.
        if vals.size == 0:
            continue

        # Keep validation observations only through the requested generation.
        max_validation_points = generations // validation_interval
        vals = vals[:max_validation_points]

        reps.append(vals)
    if not reps:
        raise ValueError(f"No usable validation_fitness values found in: {pattern}")
    # Align runs by validation index, but express the x-axis in regular
    # generation units. Each validation observation represents 100 generations.
    max_len = max(len(vals) for vals in reps)
    data = np.full((len(reps), max_len), np.nan, dtype=float)
    for i, vals in enumerate(reps):
        data[i, :len(vals)] = vals

    # The first non-zero validation is recorded at generation 100.
    gens = np.arange(1, max_len + 1) * validation_interval
    medians = np.nanmean(data, axis=0)
    q25 = np.nanpercentile(data, 25, axis=0)
    q75 = np.nanpercentile(data, 75, axis=0)
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
    # result4 = load_best_fitness_reps(exp_dir_4)
    # result5 = load_best_fitness_reps(exp_dir_5)

    gens1, med1, q25_1, q75_1 = result1
    gens2, med2, q25_2, q75_2 = result2
    gens3, med3, q25_3, q75_3 = result3
    # gens4, med4, q25_4, q75_4 = result4
    # gens5, med5, q25_5, q75_5 = result5

    plt.figure(figsize=(6,4))
    AddToPlot(gens1, med1, q25_1, q75_1, "Modified Rates", color='tab:green')
    AddToPlot(gens2, med2, q25_2, q75_2, "Fixed Rates", color='tab:blue')
    AddToPlot(gens3, med3, q25_3, q75_3, "Inherited Rates", color='tab:orange')
    # AddToPlot(gens4, med4, q25_4, q75_4, "BALDWIN", color='tab:red')
    # AddToPlot(gens5, med5, q25_5, q75_5, "Memory with Reward Difference", color='teal')
    
    ax = plt.gca()
    ax.set_xlim(0, generations)
    ax.grid(which="both", color="lightgray", linewidth=0.5, alpha=0.6)

    leg = plt.legend(
    fontsize=14,
    loc="upper left",
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
