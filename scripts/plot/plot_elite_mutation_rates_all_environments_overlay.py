"""Overlay elite post-execution mutation rates for every environment."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = (
    ("mountaincar_continuous_execution_modified_rates", "Mountain Car", "#0072B2"),
    ("pendulum_execution_modified_rates", "Pendulum", "#E69F00"),
    ("acrobot_execution_modified_rates", "Acrobot", "#009E73"),
    ("reacher_execution_modified_rates_vector", "Reacher", "#CC79A7"),
    ("halfcheetah_execution_modified_rates_vector", "Half-Cheetah", "#D55E00"),
)
RATES = (("swap", "Swap"), ("delete", "Delete"), ("add", "Add"),
         ("mutate", "One-point mutation"), ("decoy", "Decoy control"))
MAX_GENERATION = 5000


def read_run(path):
    run = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            generation = int(row["generation"])
            if generation <= MAX_GENERATION:
                run[generation] = {
                    rate: float(row[f"elite_avg_output_rate_{rate}"])
                    for rate, _ in RATES
                }
    if MAX_GENERATION not in run or len(run) != MAX_GENERATION + 1:
        return None
    return run


def load_runs(experiment):
    """Use seeds 1–20, replacing any incomplete seed with the next extra seed."""
    selection_dir = ROOT / "experiments" / experiment / "logs" / "selection"
    runs = []
    skipped = []
    for seed in range(1, 21):
        path = selection_dir / f"selection.{seed}.0.csv"
        run = read_run(path) if path.exists() else None
        if run is None:
            skipped.append(seed)
            continue
        runs.append(run)
    extra_seed = 21
    while skipped and len(runs) < 20:
        path = selection_dir / f"selection.{extra_seed}.0.csv"
        run = read_run(path) if path.exists() else None
        if run is not None:
            runs.append(run)
            skipped.pop(0)
        extra_seed += 1
        if extra_seed > 40:
            break
    if not runs:
        raise ValueError(f"No runs reach generation {MAX_GENERATION} in {selection_dir}")
    return runs, skipped


def summarize(runs, generations, rate):
    values = np.array([[run[int(generation)][rate] for generation in generations]
                       for run in runs], dtype=float)
    return (np.median(values, axis=0),
            np.quantile(values, .25, axis=0),
            np.quantile(values, .75, axis=0))


def main():
    generations = np.arange(MAX_GENERATION + 1)
    loaded = {}
    for experiment, name, _ in ENVIRONMENTS:
        runs, skipped = load_runs(experiment)
        loaded[name] = runs
        note = f"; incomplete seeds still missing: {skipped}" if skipped else ""
        print(f"{name}: {len(runs)} complete runs{note}")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    highest = 0.0
    for ax, (rate, title) in zip(axes.flat, RATES):
        for _, name, color in ENVIRONMENTS:
            median, q25, q75 = summarize(loaded[name], generations, rate)
            ax.plot(generations, median, color=color, linewidth=1.8, label=name)
            ax.fill_between(generations, q25, q75, color=color, alpha=.12, linewidth=0)
            highest = max(highest, float(np.nanmax(q75)))
        ax.set(title=title, xlim=(0, MAX_GENERATION),
               xticks=np.arange(0, MAX_GENERATION + 1, 1000),
               xlabel="Generations", ylabel="Proportion")
    ylim = max(0.65, np.ceil(highest * 10) / 10)
    for ax in axes.flat[:len(RATES)]:
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.arange(0, ylim + 1e-9, 0.1))
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Elite Mutation Proportions", fontsize=15)
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    output = ROOT / "output" / "elite_mutation_proportions_all_environments_5000.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {output}")
    print(f"Highest interquartile value: {highest:.3f}; y limit: {ylim:.2f}")


if __name__ == "__main__":
    main()
