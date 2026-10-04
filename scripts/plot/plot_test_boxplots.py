"""Box plots of independent seed-level test means at selected generations."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output" / "test_replay" / "test_seed_scores.csv"
ACROBOT_SOURCE = SOURCE.with_name("acrobot_test_seed_scores.csv")
METHODS = (
    ("Modified mutation rates", "Modified\nrates", "tab:green"),
    ("Fixed individual mutation rates", "Fixed\nindividual", "tab:blue"),
    ("Fixed mutation rates", "Fixed\nrates", "tab:orange"),
)
ENVIRONMENTS = (
    ("Mountain Car Continuous", 2000),
    ("Pendulum", 1000),
    ("Half Cheetah", 4000),
    ("Reacher", 3000),
)
ACROBOT = ("Acrobot", None)


def load_scores(environment_filter=None):
    grouped = defaultdict(list)
    for source in (SOURCE, ACROBOT_SOURCE):
        if not source.exists():
            continue
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if environment_filter and row["environment"].casefold() != environment_filter.casefold():
                    continue
                if source == SOURCE and row["environment"] == "Acrobot" and ACROBOT_SOURCE.exists():
                    continue
                generation = None if row["environment"] == "Acrobot" else int(row["generation"])
                grouped[(row["environment"], generation, row["method"])].append(float(row["test_mean"]))
    if not grouped:
        raise ValueError("No test scores found for the requested environment; collect the completed replays first")
    available = {key[0] for key in grouped}
    for environment, generation in ENVIRONMENTS + (ACROBOT,):
        if environment not in available:
            continue
        for method, _, _ in METHODS:
            if len(grouped[(environment, generation, method)]) != 20:
                raise ValueError(f"Expected 20 seed means for {environment}, {method}, generation {generation}")
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", help="Plot only one environment, e.g. Acrobot")
    args = parser.parse_args()
    scores = load_scores(args.environment)
    available = {key[0] for key in scores}
    environments = tuple(item for item in ENVIRONMENTS + (ACROBOT,) if item[0] in available)
    columns = min(2, len(environments))
    rows = math.ceil(len(environments) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows),
                             constrained_layout=True, squeeze=False)
    for ax, (environment, generation) in zip(axes.flat, environments):
        data = [scores[(environment, generation, method)] for method, _, _ in METHODS]
        result = ax.boxplot(data, positions=(1, 2, 3), widths=.55, patch_artist=True,
                            whis=1.5, showfliers=True,
                            medianprops={"linewidth": 2.2},
                            flierprops={"marker": "o", "markersize": 5,
                                        "markerfacecolor": "none", "linestyle": "none"})
        for index, (_, _, color) in enumerate(METHODS):
            result["boxes"][index].set(facecolor=color, edgecolor=color, alpha=.26, linewidth=2)
            result["medians"][index].set(color=color)
            result["fliers"][index].set(markeredgecolor=color, alpha=.8)
            for item in result["whiskers"][2*index:2*index+2] + result["caps"][2*index:2*index+2]:
                item.set(color=color, linewidth=1.5)
        title = (f"{environment} · final available checkpoint" if generation is None
                 else f"{environment} · generation {generation:,}")
        ax.set(title=title, ylabel="Test score",
               xticks=(1, 2, 3), xticklabels=[label for _, label, _ in METHODS], xlim=(.45, 3.55))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    for ax in axes.flat[len(environments):]:
        ax.axis("off")
    legend = [Patch(facecolor=color, edgecolor=color, alpha=.35, linewidth=2, label=method)
              for method, _, color in METHODS]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0),
               ncol=1 if len(environments) == 1 else 3, frameon=False)
    name = "acrobot_test_score_boxplots.pdf" if args.environment and args.environment.casefold() == "acrobot" else "test_score_boxplots.pdf"
    output = ROOT / "output" / "test_replay" / name
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
