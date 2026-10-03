"""Box plots of independent seed-level test means at selected generations."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output" / "test_replay" / "test_seed_scores.csv"
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


def load_scores():
    grouped = defaultdict(list)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["environment"], int(row["generation"]), row["method"])
            grouped[key].append(float(row["test_mean"]))
    for environment, generation in ENVIRONMENTS:
        for method, _, _ in METHODS:
            if len(grouped[(environment, generation, method)]) != 20:
                raise ValueError(f"Expected 20 seed means for {environment}, {method}, generation {generation}")
    return grouped


def main():
    scores = load_scores()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (environment, generation) in zip(axes.flat, ENVIRONMENTS):
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
        ax.set(title=f"{environment} · generation {generation:,}", ylabel="Test score",
               xticks=(1, 2, 3), xticklabels=[label for _, label, _ in METHODS], xlim=(.45, 3.55))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    legend = [Patch(facecolor=color, edgecolor=color, alpha=.35, linewidth=2, label=method)
              for method, _, color in METHODS]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    output = ROOT / "output" / "test_replay" / "test_score_boxplots.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
