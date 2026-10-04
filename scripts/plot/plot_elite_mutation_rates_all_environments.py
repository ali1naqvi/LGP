"""Plot elite mutation and decoy rates for five environments in one PDF."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = (
    ("mountaincar_continuous", "Mountain Car", 2000),
    ("pendulum", "Pendulum", 1000),
    ("halfcheetah", "Half-Cheetah", 4000),
    ("reacher", "Reacher", 3000),
    ("acrobot", "Acrobot", 5000),
)
RATES = (("swap", "Swap"), ("delete", "Delete"), ("add", "Add"),
         ("mutate", "One-point mutation"), ("decoy", "Decoy control"))


def load_runs(environment, max_generation):
    suffix = "_no_limit" if environment == "pendulum" else ""
    selection_dir = (ROOT / "experiments" /
                     f"{environment}_execution_modified_rates{suffix}" / "logs" / "selection")
    files = sorted(selection_dir.glob("selection.*.0.csv"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 selection logs in {selection_dir}; found {len(files)}")
    runs = []
    for path in files:
        run = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                generation = int(row["generation"])
                if generation <= max_generation:
                    run[generation] = {
                        (stage, rate): float(row[f"elite_avg_{stage}_rate_{rate}"])
                        for rate, _ in RATES for stage in ("start", "output")
                    }
        if not run:
            raise ValueError(f"No rows through generation {max_generation} in {path}")
        runs.append(run)
    return runs


def summarize(runs, generations, stage, rate):
    values = np.array([[run.get(int(g), {}).get((stage, rate), np.nan)
                        for g in generations] for run in runs])
    counts = np.isfinite(values).sum(axis=0)
    if np.any(counts == 0):
        raise ValueError(f"No values for {stage} {rate} at some generations")
    return (np.nanmedian(values, axis=0),
            np.nanquantile(values, .25, axis=0),
            np.nanquantile(values, .75, axis=0))


def plot_environment(environment, title, max_generation):
    runs = load_runs(environment, max_generation)
    generations = np.array(sorted(set().union(*(run.keys() for run in runs))))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (rate, panel_title) in zip(axes.flat, RATES):
        for stage, label, color, style in (("output", "Post-execution rate", "tab:orange", "-"),
                                           ("start", "Starting rate", "tab:blue", "--")):
            median, q25, q75 = summarize(runs, generations, stage, rate)
            ax.plot(generations, median, color=color, linestyle=style,
                    linewidth=1.7, label=label, zorder=3 if stage == "start" else 2)
            ax.fill_between(generations, q25, q75, color=color, alpha=.16, linewidth=0)
        ax.set(title=panel_title, xlim=(0, max_generation), ylim=(0, 1),
               xticks=np.linspace(0, max_generation, 6, dtype=int),
               yticks=np.arange(0, 1.01, .2),
               xlabel="Generations", ylabel="Proportion")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{title} Elite Mutation and Decoy Proportions", fontsize=15)
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center", ncol=2, frameon=False)
    print(f"{title}: {len(runs)} runs; "
          f"{sum(max_generation in run for run in runs)} at generation {max_generation}")
    return fig


def main():
    output = ROOT / "output" / "pdf" / "elite_mutation_and_decoy_start_vs_post_all_environments.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for environment, title, max_generation in ENVIRONMENTS:
            fig = plot_environment(environment, title, max_generation)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
