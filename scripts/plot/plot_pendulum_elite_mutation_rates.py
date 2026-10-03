"""Plot Pendulum elite mutation probabilities before and after execution."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SELECTION_DIR = ROOT / "experiments" / "pendulum_execution_modified_rates_no_limit" / "logs" / "selection"
RATES = (("swap", "Swap"), ("delete", "Delete"), ("add", "Add"),
         ("mutate", "One-point mutation"), ("decoy", "Decoy control"))
MAX_GENERATION = 1000


def load_runs():
    files = sorted(SELECTION_DIR.glob("selection.*.0.csv"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 Pendulum selection logs; found {len(files)}")
    runs = []
    for path in files:
        run = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                generation = int(row["generation"])
                if generation <= MAX_GENERATION:
                    run[generation] = {
                        (stage, rate): float(row[f"elite_avg_{stage}_rate_{rate}"])
                        for rate, _ in RATES for stage in ("start", "output")
                    }
        if not run:
            raise ValueError(f"No mutation-rate rows in {path}")
        runs.append(run)
    return runs


def summarize(runs, generations, stage, rate):
    values = np.array([[run.get(int(generation), {}).get((stage, rate), np.nan)
                        for generation in generations] for run in runs])
    counts = np.isfinite(values).sum(axis=0)
    if np.any(counts == 0):
        raise ValueError(f"No values for {stage} {rate} at some generations")
    return np.nanmedian(values, axis=0), np.nanquantile(values, .25, axis=0), np.nanquantile(values, .75, axis=0)


def main():
    runs = load_runs()
    generations = np.array(sorted(set().union(*(run.keys() for run in runs))))
    counts = np.array([sum(int(generation) in run for run in runs) for generation in generations])
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             constrained_layout=True)
    for ax, (rate, title) in zip(axes.flat, RATES):
        for stage, label, color, style in (("output", "Post-execution rate", "tab:orange", "-"),
                                           ("start", "Starting rate", "tab:blue", "--")):
            median, q25, q75 = summarize(runs, generations, stage, rate)
            ax.plot(generations, median, color=color, linestyle=style,
                    linewidth=1.7, label=label, zorder=3 if stage == "start" else 2)
            ax.fill_between(generations, q25, q75, color=color, alpha=.16, linewidth=0)
        ax.set(title=title, xlim=(0, MAX_GENERATION), ylim=(0, .65),
               xticks=np.arange(0, MAX_GENERATION + 1, 200),
               yticks=np.arange(0, .61, .1),
               xlabel="Generations", ylabel="Proportion")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    fig.suptitle("Pendulum Elite Mutation and Decoy Proportions", fontsize=15)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    output = ROOT / "output" / "pendulum_elite_mutation_and_decoy_start_vs_post_1000.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"Saved {output}; runs at start: {counts[0]}, at generation {MAX_GENERATION}: {counts[-1]}")


if __name__ == "__main__":
    main()
