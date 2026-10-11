"""Plot the three mutation variants over the requested environment ranges."""

import csv
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_mutation_validation_comparison import COLORS, EXPERIMENTS, LABELS, ROOT, load_experiment
from plot_test_boxplots import load_scores, METHODS


PANELS = (
    ("Mountain Car Continuous", 2000, EXPERIMENTS["Mountain Car Continuous"]),
    ("Pendulum", 1000, EXPERIMENTS["Pendulum"]),
    ("Acrobot", 5000, ("acrobot_execution_modified_rates", "acrobot_fixed_individual", "acrobot_fixed_rates")),
    ("Reacher", 3000, EXPERIMENTS["Reacher"]),
    ("Half Cheetah", 4000, EXPERIMENTS["Half Cheetah"]),
)
OUTPUT = ROOT / "output" / "training_requested_ranges"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-tests", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2 if args.with_tests else 1, 5,
                             figsize=(18, 9.0 if args.with_tests else 5.0),
                             layout="constrained", squeeze=False)
    rows = []
    for panel_index, (ax, (environment, endpoint, experiments)) in enumerate(zip(axes[0], PANELS)):
        for name, label, color in zip(experiments, LABELS, COLORS):
            runs, total = load_experiment(name, "best_fitness")
            generations = sorted(g for g in set().union(*(run.keys() for run in runs)) if g <= endpoint)
            summaries = []
            for generation in generations:
                values = [run[generation] for run in runs if generation in run]
                q25, median, q75 = np.quantile(values, [.25, .5, .75])
                summaries.append((median, q25, q75))
                rows.append(dict(environment=environment, variant=label, experiment=name,
                                 generation=generation, n_runs=len(values), median=median, q25=q25, q75=q75))
            median, q25, q75 = np.array(summaries).T
            ax.plot(generations, median, color=color, linewidth=1.8, label=label)
            ax.fill_between(generations, q25, q75, color=color, alpha=.14)
            counts = [sum(g in run for run in runs) for g in generations]
            terminal = sum(endpoint in run for run in runs)
            print(f"{environment}: {label}: {total} logs; {min(counts)}–{max(counts)} contributing runs; {terminal} at generation {endpoint}")
        ax.set(xlim=(0, endpoint), xlabel="Generation",
               ylabel="Training fitness" if ax is axes[0, 0] else "")
        panel_environment = environment.replace("Mountain Car Continuous", "Mountain Car")
        ax.set_title(f"({chr(97 + panel_index)}) Training - {panel_environment}", fontsize=17, pad=10)
        ax.set_xlabel("Generation", fontsize=15, labelpad=7)
        if ax is axes[0, 0]:
            ax.set_ylabel("Training fitness", fontsize=15, labelpad=8)
        ax.tick_params(labelsize=13, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
    if args.with_tests:
        scores = load_scores()
        for panel_index, (ax, (environment, endpoint, _)) in enumerate(zip(axes[1], PANELS)):
            generation = None if environment == "Acrobot" else endpoint
            data = [scores[(environment, generation, label)] for label in LABELS]
            result = ax.boxplot(data, widths=.55, patch_artist=True, whis=1.5,
                                medianprops={"linewidth": 2.2},
                                flierprops={"marker": "o", "markersize": 5,
                                            "markerfacecolor": "none", "linestyle": "none"})
            for index, color in enumerate(COLORS):
                result["boxes"][index].set(facecolor=color, edgecolor=color, alpha=.26, linewidth=1.5)
                result["medians"][index].set(color=color)
                result["fliers"][index].set(markeredgecolor=color)
                for item in result["whiskers"][2*index:2*index+2] + result["caps"][2*index:2*index+2]:
                    item.set(color=color, linewidth=1.2)
            panel_environment = environment.replace("Mountain Car Continuous", "Mountain Car")
            title = f"({chr(97 + panel_index)}) Testing - {panel_environment}"
            ax.set(ylabel="Test fitness" if ax is axes[1, 0] else "",
                   xticks=(1, 2, 3), xticklabels=[short for _, short, _ in METHODS])
            ax.set_title(title, fontsize=17, pad=10)
            if ax is axes[1, 0]:
                ax.set_ylabel("Test fitness", fontsize=15, labelpad=8)
            ax.tick_params(axis="y", labelsize=13, pad=5)
            ax.tick_params(axis="x", labelsize=11, pad=6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False, axis="x")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=14,
               title=("Training: median and 25th–75th percentile · Testing: 20 seed means per variant"
                      if args.with_tests else "Median across available runs · Shading: 25th–75th percentile"),
               title_fontsize=13)
    stem = "training_and_testing" if args.with_tests else "training_curves"
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (OUTPUT / "training_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT / f"{stem}.png")


if __name__ == "__main__":
    main()
