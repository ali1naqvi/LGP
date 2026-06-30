import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sys

probability_setups = {
    0.00:   "5_1_ant_stateful_no_obs_0",
    0.125:  "5_1_ant_stateful_no_obs_25",
    0.25:   "5_1_ant_stateful_no_obs_50",
    0.375:  "5_1_ant_stateful_no_obs_75",
    0.50:   "5_1_ant_stateful_no_obs_100",
}

base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../", "tpg", "experiments")
)

def generations_per_seed(exp_dir):
    pattern = os.path.join(exp_dir, "logs", "selection", "selection.*.0.csv")
    files   = sorted(glob.glob(pattern))
    counts  = []
    for f in files:
        try:
            n_rows = sum(1 for _ in open(f, "rb")) - 1
            if n_rows > 0:
                counts.append(n_rows)
        except (OSError, IOError):
            print(f"[warn] could not read {f}", file=sys.stderr)
    return np.asarray(counts, dtype=np.int32)

probs, medians, q1_err, q3_err = [], [], [], []

for p, exp_name in probability_setups.items():
    exp_dir = os.path.join(base_path, exp_name)
    gen_counts = generations_per_seed(exp_dir)
    if gen_counts.size == 0:
        print(f"[skip] no data in {exp_dir}", file=sys.stderr)
        continue

    probs.append(p)
    med = np.median(gen_counts)
    q1, q3 = np.percentile(gen_counts, [25, 75])

    medians.append(med)
    q1_err.append(med - q1)  
    q3_err.append(q3 - med)   

probs    = np.asarray(probs)
medians  = np.asarray(medians)
yerr     = np.vstack([q1_err, q3_err])

plt.figure(figsize=(6, 4))
plt.errorbar(
    probs,
    medians,
    yerr=yerr,
    fmt="-o",
    linewidth=1.5,
    capsize=4,
    label="median ± IQR",
    color=(131/255, 204/255, 235/255)
)

# plt.xlabel("Connection–mutation probability $p$")
# plt.ylabel("Generations")
#plt.title("Convergence speed vs. connection probability on Ant-v2")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("gen_count_vs_probability.pdf", bbox_inches="tight")
plt.show()
