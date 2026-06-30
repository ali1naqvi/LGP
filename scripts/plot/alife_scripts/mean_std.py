import numpy as np
import pandas as pd
import os
import glob


# experiment_name_1 = "36_td_sf_me_dynamic"
# experiment_name_2 = "36_td_sf_dynamic"
# experiment_name_3 = "35_baseline_ntpg_sf_maze_dyn"
# experiment_name_4 = "35_baseline_tpg_sf_maze_dyn"

experiment_name_1 = "41_dynamic"
experiment_name_2 = "41_dynamic_neuro"
experiment_name_3 = "46_td_static_disc_gate_original"
experiment_name_4 = "46_td_static_disc_gate_original_dynamic"


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "tpg", "experiments"))
print(base_path)
exp_dir_1 = os.path.join(base_path, experiment_name_1, "logs", "selection")
exp_dir_2 = os.path.join(base_path, experiment_name_2, "logs", "selection")
exp_dir_3 = os.path.join(base_path, experiment_name_3, "logs", "selection")
exp_dir_4 = os.path.join(base_path, experiment_name_4, "logs", "selection")
# exp_dir_5 = os.path.join(base_path, experiment_name_5, "logs", "selection")

#best_fitness, program_instruction_count,effective_program_instruction_count
def load_seed_final_fitness(exp_dir, target_gen=5000):
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files   = sorted(glob.glob(pattern))
    finals  = []
    for f in files:
        df = pd.read_csv(f, usecols=["best_fitness"])
        if len(df):
            # Use generation 250 if available (index 249); otherwise use the last available generation
            idx = min(target_gen - 1, len(df) - 1)
            finals.append(df["best_fitness"].iloc[idx])
    return np.array(finals)


if __name__ == "__main__":
    vals1 = load_seed_final_fitness(exp_dir_1)
    vals2 = load_seed_final_fitness(exp_dir_2)
    vals3 = load_seed_final_fitness(exp_dir_3)
    vals4 = load_seed_final_fitness(exp_dir_4)

    data   = [vals1, vals2, vals3, vals4]
    labels = ["tpg",
              "tpg_neuro",
              "tpg+obs", 'tpg_neuro+obs']
    # labels = ["NeuroTPG - 20% MR",
    #           "NeuroTPG - 80% MR",
    #           "NeuroTPG - 100% MR"]
    
    means = [np.median(vals) for vals in data]
    stds  = [np.std(vals)  for vals in data]

    for label, mean, std in zip(labels, means, stds):
        print(f"{label}:  mean = {mean:.2f},  std = {std:.2f}")