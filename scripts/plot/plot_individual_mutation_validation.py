"""Compare single-mutation variants with all-mutations runs on validation fitness."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_mutation_validation_comparison import ROOT, load_experiment, summarize


EXPERIMENTS = {
    "Ant · modified mutation rates": (
        ("All mutations", "ant_execution_modified_rates_all"),
        ("Add", "ant_execution_modified_rates_add"),
        ("Delete", "ant_execution_modified_rates_delete"),
        ("Swap", "ant_execution_modified_rates_swap"),
        ("One-point mutate", "ant_execution_modified_rates_one_mutate"),
    ),
    "Pendulum · fixed individual mutation rates": (
        ("All mutations", "pendulum_fixed_individual_all"),
        ("Add", "pendulum_fixed_individual_add"),
        ("Delete", "pendulum_fixed_individual_delete"),
        ("Swap", "pendulum_fixed_individual_swap"),
        ("One-point mutate", "pendulum_fixed_individual_one_point"),
    ),
}
COLORS = ("black", "tab:blue", "tab:orange", "tab:green", "tab:red")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (title, experiments) in zip(axes, EXPERIMENTS.items()):
        loaded = [(label, *load_experiment(name)) for label, name in experiments]
        endpoint = min(min(max(run) for run in runs) for _, runs, _ in loaded)
        endpoint = (endpoint // 500) * 500
        if endpoint < 500:
            raise ValueError(f"Insufficient shared validation range for {title}")
        for (label, runs, total), color in zip(loaded, COLORS):
            generations, median, q25, q75 = summarize(runs, endpoint)
            print(f"{title}: {label}: {len(runs)}/{total} runs through generation {endpoint}")
            ax.plot(generations, median, label=label, color=color, linewidth=1.7)
            ax.fill_between(generations, q25, q75, color=color, alpha=.10)
        ax.set(title=title, xlim=(0, endpoint), xlabel="Generations", ylabel="Validation fitness")
    axes[0].legend(frameon=True, fontsize=9)
    output = ROOT / "individual_mutations_vs_all_validation.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
