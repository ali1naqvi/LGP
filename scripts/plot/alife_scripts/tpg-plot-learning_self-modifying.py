"""Plot total, effective, and self-modification instruction counts together."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

GENERATIONS = 5000
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "experiments")
)
EXPERIMENTS = (
    ("pendulum_execution_modified_rates", "Modified Rates", "tab:green"),
)
TOTAL = "program_instruction_count"
EFFECTIVE = "effective_program_instruction_count"
SELF_MODIFICATION_ONLY = "self_modification_only_instruction_count"
ACTION_EFFECTIVE = "action_effective_instruction_count"


def load_instruction_reps(experiment):
    pattern = os.path.join(BASE_PATH, experiment, "logs", "selection", "selection.*.0.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No selection logs found matching: {pattern}")

    columns = [TOTAL, EFFECTIVE, SELF_MODIFICATION_ONLY]
    reps = {column: [] for column in columns + [ACTION_EFFECTIVE]}
    for path in files:
        try:
            df = pd.read_csv(path, usecols=columns)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        # The logged dependency analysis already excludes decoy instructions.
        # Subtract instructions, not the number of reserved register slots.
        df[ACTION_EFFECTIVE] = df[EFFECTIVE] - df[SELF_MODIFICATION_ONLY]
        if (df[ACTION_EFFECTIVE] < 0).any():
            raise ValueError(f"Self-modification-only count exceeds effective count: {path}")
        for column in reps:
            values = df[column].to_numpy()
            # Hold the final observation for runs shorter than the plotted range.
            values = np.pad(values[:GENERATIONS],
                            (0, max(0, GENERATIONS - len(values))),
                            mode="edge")
            reps[column].append(values)

    if not reps[TOTAL]:
        raise ValueError(f"No data rows found in selection logs matching: {pattern}")
    return {column: np.vstack(values) for column, values in reps.items()}


def plot_summary(ax, data, column, label, color):
    values = data[column]
    generations = np.arange(GENERATIONS)
    mean = np.mean(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    ax.plot(generations, mean, label=label, linewidth=2, color=color)
    ax.fill_between(generations, q25, q75, alpha=0.10, color=color)


def save_comparison(data_by_experiment):
    fig, axes = plt.subplots(len(EXPERIMENTS), 1,
                             figsize=(7, 4 * len(EXPERIMENTS)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (experiment, label, color) in zip(axes, EXPERIMENTS):
        data = data_by_experiment[experiment]
        plot_summary(ax, data, TOTAL, "Total", "tab:gray")
        plot_summary(ax, data, EFFECTIVE, "Effective", color)
        plot_summary(ax, data, ACTION_EFFECTIVE,
                     "Effective without self-modification-only", "tab:blue")
        plot_summary(ax, data, SELF_MODIFICATION_ONLY,
                     "Self-modification-only", "tab:orange")
        ax.set(title=label, ylabel="Instruction count")
        ax.legend(loc="best", frameon=True)
    axes[-1].set_xlabel("Generation")
    fig.tight_layout()
    fig.savefig("instruction_counts_comparison.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data_by_experiment = {
        experiment: load_instruction_reps(experiment)
        for experiment, _, _ in EXPERIMENTS
    }
    save_comparison(data_by_experiment)
