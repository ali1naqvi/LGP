import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
import numpy as np
import os, glob

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "tpg", "experiments"))

neuro = os.path.join(base_path, "ant_neuro_stateful_tanh_soa", "logs", "selection")
tpg = os.path.join(base_path, "ant_stateful_tanh_soa", "logs", "selection")

def load_seed_final_fitness(exp_dir):
    pattern = os.path.join(exp_dir, "selection.*.0.csv")
    files   = sorted(glob.glob(pattern))
    finals  = []
    for f in files:
        df = pd.read_csv(f, usecols=["best_fitness"])
        if len(df):
            finals.append(df["best_fitness"].iloc[-1])   # last gen
    return np.array(finals)


neuro = load_seed_final_fitness(neuro)
tpg = load_seed_final_fitness(tpg)

tpg_df  = pd.DataFrame(tpg).values.flatten()
hebb_df = pd.DataFrame(neuro).values.flatten()

u_stat_ep, p_val_ep = stats.mannwhitneyu(tpg_df, hebb_df, alternative='greater')
ks_stat_ep, ks_p_ep = stats.ks_2samp(tpg_df, hebb_df)

print("U = ", u_stat_ep, ", p = ", p_val_ep)
print("D = ", ks_stat_ep, ", p = ", ks_p_ep)

# --- Bootstrap test for median difference ---
def bootstrap_median_diff(x, y, n_resamples=10000):
    diffs = []
    combined = np.concatenate([x, y])
    n_x = len(x)
    for _ in range(n_resamples):
        np.random.shuffle(combined)
        sample_x = combined[:n_x]
        sample_y = combined[n_x:]
        diffs.append(np.median(sample_y) - np.median(sample_x))
    diffs = np.array(diffs)
    # two-sided p-value
    obs_diff = np.median(y) - np.median(x)
    p_boot = np.mean(np.abs(diffs) >= abs(obs_diff))
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return p_boot, (ci_lower, ci_upper)

p_boot, (ci_lower, ci_upper) = bootstrap_median_diff(tpg_df, hebb_df)
print(f"Bootstrap p-value for median difference: {p_boot:.4f}")
print(f"95% CI for median difference: [{ci_lower:.4f}, {ci_upper:.4f}]")