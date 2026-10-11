"""Reacher validation fitness at scheduled evaluations through generation 3000."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_mutation_validation_comparison import COLORS, EXPERIMENTS, LABELS, ROOT, load_experiment, summarize


def main():
    output = ROOT / "output" / "training_requested_ranges"
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    rows = []
    for name, label, color in zip(EXPERIMENTS["Reacher"], LABELS, COLORS):
        runs, total = load_experiment(name, "validation_fitness")
        runs = [{g: value for g, value in run.items() if g % 100 == 0} for run in runs]
        generations, median, q25, q75 = summarize(runs, 3000)
        ax.plot(generations, median, color=color, label=label, linewidth=1.7)
        ax.fill_between(generations, q25, q75, color=color, alpha=.14)
        rows.extend(dict(variant=label, generation=g, n_runs=len(runs), median=m, q25=l, q75=h)
                    for g, m, l, h in zip(generations, median, q25, q75))
        print(f"{label}: {len(runs)}/{total} runs; final median {median[-1]:.4f}; initial median {median[0]:.4f}")
    ax.set(title="Validation - Reacher", xlabel="Generation", ylabel="Validation fitness", xlim=(0, 3000))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=9, title="Median; shading: 25th–75th percentile", title_fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"reacher_validation.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (output / "reacher_validation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
