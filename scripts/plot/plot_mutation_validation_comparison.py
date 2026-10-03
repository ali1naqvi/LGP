"""Compare validation fitness while all runs contribute in each environment."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "Ant": ("ant_execution_modified_rates_all", "ant_fixed_individual_all", "ant_fixed_rates"),
    "Half Cheetah": ("halfcheetah_execution_modified_rates", "halfcheetah_fixed_individual", "halfcheetah_fixed_rates"),
    "Mountain Car Continuous": ("mountaincar_continuous_execution_modified_rates", "mountaincar_continuous_fixed_individual", "mountaincar_continuous_fixed_rates"),
    "Pendulum": ("pendulum_execution_modified_rates_no_limit", "pendulum_fixed_individual_all", "pendulum_fixed_rates_no_limit"),
    "Reacher": ("reacher_execution_modified_rates", "reacher_fixed_individual", "reacher_fixed_rates"),
}
LABELS = ("Modified mutation rates", "Fixed individual mutation rates", "Fixed mutation rates")
COLORS = ("tab:green", "tab:blue", "tab:orange")
MAX_GENERATION = 5000


def load_experiment(name, metric="validation_fitness"):
    paths = sorted((ROOT / "experiments" / name / "logs" / "selection").glob("selection.*.0.csv"))
    if not paths:
        raise FileNotFoundError(f"No selection logs for {name}")
    runs = []
    for path in paths:
        values = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                generation = int(row["generation"])
                if generation > MAX_GENERATION:
                    continue
                value = float(row[metric])
                if np.isfinite(value) and (metric != "validation_fitness" or value != 0):
                    values[generation] = value
        if values:
            runs.append(values)
    if not runs:
        raise ValueError(f"No {metric} values for {name}")
    return runs, len(paths)


def summarize(runs, endpoint):
    generations = np.array(sorted(set().union(*(run.keys() for run in runs))))
    generations = generations[generations <= endpoint]
    data = np.array([[run.get(int(g), np.nan) for g in generations] for run in runs])
    counts = np.sum(np.isfinite(data), axis=0)
    if np.any(counts != len(runs)):
        raise ValueError(f"Some runs lack data within the shared range ending at {endpoint}")
    return generations, np.median(data, axis=0), np.quantile(data, .25, axis=0), np.quantile(data, .75, axis=0)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (environment, experiments) in zip(axes.flat, EXPERIMENTS.items()):
        loaded = [load_experiment(name) for name in experiments]
        endpoint = min(max(0, min(max(run) for run in runs)) for runs, _ in loaded)
        endpoint = (endpoint // 500) * 500
        for (runs, total), label, color in zip(loaded, LABELS, COLORS):
            generations, median, q25, q75 = summarize(runs, endpoint)
            print(f"{environment}: {label}: {len(runs)}/{total} runs through generation {endpoint}")
            ax.plot(generations, median, color=color, linewidth=1.7, label=label)
            ax.fill_between(generations, q25, q75, color=color, alpha=.12)
        ax.set(title=environment, xlim=(0, endpoint), xlabel="Generations", ylabel="Validation fitness")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=True)
    output = ROOT / "mutation_validation_comparison_all_runs.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
