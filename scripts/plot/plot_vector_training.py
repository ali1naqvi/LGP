"""Plot vector experiment training fitness, with medians and interquartile bands."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_mutation_validation_comparison import ROOT, load_experiment, summarize


VARIANTS = (
    ("execution_modified_rates", "Modified mutation rates", "tab:green"),
    ("fixed_individual", "Fixed individual mutation rates", "tab:blue"),
    ("fixed_rates", "Fixed mutation rates", "tab:orange"),
)
ENVIRONMENTS = (("reacher", "Reacher"), ("halfcheetah", "Half Cheetah"))
OUTPUT = ROOT / "output" / "vector_training"


def plot_panels(panels, stem, summary_name, sharey=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=sharey)
    rows = []
    all_loaded = [[load_experiment(f"{environment}_{variant}_{representation}", "best_fitness")
                   for variant, _, _ in VARIANTS]
                  for environment, _, representation in panels]
    endpoint = min(max(run) for loaded in all_loaded for runs, _ in loaded for run in runs)
    counts = []
    for ax, (environment, title, representation), loaded in zip(axes, panels, all_loaded):
        for (variant, label, color), (runs, total) in zip(VARIANTS, loaded):
            counts.append(len(runs))
            generations, median, q25, q75 = summarize(runs, endpoint)
            ax.plot(generations, median, color=color, linewidth=1.5, label=label)
            ax.fill_between(generations, q25, q75, color=color, alpha=.14, linewidth=0)
            for generation, mid, low, high in zip(generations, median, q25, q75):
                rows.append(dict(environment=title, representation=representation,
                                 variant=label, generation=int(generation),
                                 n_runs=len(runs), median=mid, q25=low, q75=high))
            print(f"{title} {representation}: {label}: {len(runs)}/{total} runs, through generation {endpoint}")
        ax.set(title=f"{title} — {representation.title()}", xlabel="Generation", ylabel="Training fitness",
               xlim=(0, endpoint))
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(16)
        ax.tick_params(labelsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    count_label = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               title=f"Median across {count_label} valid runs · Shading: 25th–75th percentile", fontsize=11)
    fig.tight_layout(rect=(0, .17, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (OUTPUT / summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT / f"{stem}.png")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_panels([(environment, title, "vector") for environment, title in ENVIRONMENTS],
                "vector_training_curves", "training_summary.csv")
    plot_panels([("halfcheetah", "Half Cheetah", representation)
                 for representation in ("vector", "matrix")],
                "halfcheetah_vector_vs_matrix_training", "halfcheetah_vector_vs_matrix_summary.csv",
                sharey=True)


if __name__ == "__main__":
    main()
