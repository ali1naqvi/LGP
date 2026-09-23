"""Plot validation fitness with the styling of tpg-plot-learning.py."""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

GENERATIONS = 2000
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "experiments")
)
EXPERIMENTS = (
    ("pendulum_execution_modified_rates", "Modified Rates", "tab:green"),
    ("pendulum_fixed_rates", "Fixed Rates", "tab:blue"),
    ("pendulum_inherited_rates", "Inherited Rates", "tab:orange"),
)


def load_validation_reps(experiment):
    pattern = os.path.join(BASE_PATH, experiment, "logs", "selection", "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No selection logs found matching: {pattern}")

    reps = []
    for path in files:
        try:
            df = pd.read_csv(path, usecols=["generation", "validation_fitness"])
        except pd.errors.EmptyDataError:
            continue
        df = df.apply(pd.to_numeric, errors="coerce")
        # Zero denotes a generation without validation in these logs.
        df = df[(df["generation"] <= GENERATIONS) &
                df["generation"].notna() &
                df["validation_fitness"].notna() &
                (df["validation_fitness"] != 0)]
        if not df.empty:
            reps.append(df.set_index("generation")["validation_fitness"])

    if not reps:
        raise ValueError(f"No usable validation_fitness values found in: {pattern}")

    data = pd.concat(reps, axis=1)
    return (data.index.to_numpy(), data.mean(axis=1).to_numpy(),
            data.quantile(0.25, axis=1).to_numpy(),
            data.quantile(0.75, axis=1).to_numpy())


def plot_summary(ax, generations, mean, q25, q75, label, color):
    ax.plot(generations, mean, label=label, linewidth=2, color=color)
    ax.fill_between(generations, q25, q75, alpha=0.10, color=color)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6, 4))
    for experiment, label, color in EXPERIMENTS:
        plot_summary(ax, *load_validation_reps(experiment), label, color)
    ax.set(xlabel="Generation", ylabel="Validation fitness",
           title="Pendulum Task: Validation fitness", xlim=(0, GENERATIONS))
    ax.legend(fontsize=11, frameon=True)
    fig.savefig("validation_fitness.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
