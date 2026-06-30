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
# experiment_name_4 = "static_starting/5_ant_dynamic_1_fitness_tot-reg"
# experiment_name_5 = "static_starting/5_ant_dynamic_no_gc"

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
# experiment_name_4 = "static_starting/6_cheetah_dyn_rc_fitness_tot-reg"
# experiment_name_5 = "static_starting/6_cheetah_dyn_no_rc_fitness_tot-reg"
# experiment_name_6 = "6_cheetah_dyn_start_fitness_tot-reg"



base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))
exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")
# exp_dir_6 = os.path.join(base_path, experiment_name_6, "logs", "selection")
# print(exp_dir_3)
#best_fitness, validation_fitness, program_instruction_count,effective_program_instruction_count, best_agent_register_size, avg_complexity_front_0
def load_scatter_points(exp_dir):
    """Load (last_gen_fitness, proportion, total_regs) for each seed. Proportion = effective_regs / total_regs. Returns (fitnesses, proportions, total_regs) or None."""
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    fitnesses = []
    proportions = []
    total_regs = []
    for f in files:
        df = pd.read_csv(f, usecols=["validation_fitness", "best_agent_effective_register_size", "best_agent_register_size"])
        if len(df) == 0:
            continue
        last_row = df.iloc[-1]
        fit = pd.to_numeric(last_row["validation_fitness"], errors="coerce")
        eff_reg = pd.to_numeric(last_row["best_agent_effective_register_size"], errors="coerce")
        total_reg = pd.to_numeric(last_row["best_agent_register_size"], errors="coerce")
        if np.isnan(fit) or np.isnan(eff_reg) or np.isnan(total_reg) or total_reg == 0:
            continue
        prop = eff_reg
        fitnesses.append(fit)
        proportions.append(prop)
        total_regs.append(int(total_reg))
    if not fitnesses:
        return None
    return np.array(fitnesses), np.array(proportions), np.array(total_regs)

if __name__ == "__main__":
    experiments = [
        (exp_dir_1, "Static - Low", "tab:green"),
        (exp_dir_2, "Static - Medium", "tab:blue"),
        (exp_dir_3, "Static - High", "tab:orange"),
        (exp_dir_4, "Dynamic - RC", "tab:red"),
        (exp_dir_5, "Dynamic", "teal"),
        # (exp_dir_6, "Dynamic - Start", "teal"),
    ]

    plt.figure(figsize=(6, 4))
    all_props, all_fits, all_regs = np.array([]), np.array([]), np.array([])
    for exp_dir, label, color in experiments:
        result = load_scatter_points(exp_dir)
        if result is None:
            continue
        fitnesses, proportions, total_regs = result
        plt.scatter(proportions, fitnesses, label=label, color=color, alpha=0.7, s=40, zorder=2)
        all_props = np.concatenate([all_props, proportions])
        all_fits = np.concatenate([all_fits, fitnesses])
        all_regs = np.concatenate([all_regs, total_regs])

    # Compute the Pareto frontier: keep only points not dominated by any other.
    # A point is dominated if another point has both fewer/equal effective registers
    # and higher/equal fitness, with at least one strict inequality.
    def pareto_frontier(props, fits):
        points = np.column_stack([props, fits])
        n = len(points)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1] and
                        (points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break
        pareto_pts = points[is_pareto]
        pareto_pts = pareto_pts[pareto_pts[:, 0].argsort()]  # sort by effective registers
        return pareto_pts

    if len(all_props) >= 2:
        pareto_pts = pareto_frontier(all_props, all_fits)
        xs = pareto_pts[:, 0]
        ys = pareto_pts[:, 1]

        for i in range(len(pareto_pts) - 1):
            gap = xs[i + 1] - xs[i] > 1
            linestyle = ":" if gap else "-"
            plt.plot(
                xs[i:i+2],
                ys[i:i+2],
                color="k",
                linestyle=linestyle,
                linewidth=1.5,
                zorder=1,
            )

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

    plt.legend(loc="best", fontsize=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig("proportion_instructions_ant.pdf", format="pdf", bbox_inches="tight")
