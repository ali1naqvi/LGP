"""Training-fitness versions of the mutation-rate validation comparisons."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_mutation_validation_comparison import (
    COLORS as STRATEGY_COLORS,
    EXPERIMENTS as STRATEGY_EXPERIMENTS,
    LABELS as STRATEGY_LABELS,
    ROOT,
    load_experiment,
    summarize,
)
from plot_individual_mutation_validation import (
    COLORS as INDIVIDUAL_COLORS,
    EXPERIMENTS as INDIVIDUAL_EXPERIMENTS,
)


def shared_endpoint(loaded):
    endpoint = min(min(max(run) for run in runs) for runs, _ in loaded)
    endpoint = (endpoint // 500) * 500
    if endpoint < 500:
        raise ValueError("Insufficient shared training range")
    return endpoint


def save(fig, name):
    output = ROOT / name
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


def plot_strategies():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (environment, experiments) in zip(axes.flat, STRATEGY_EXPERIMENTS.items()):
        loaded = [load_experiment(name, "best_fitness") for name in experiments]
        endpoint = shared_endpoint(loaded)
        for (runs, total), label, color in zip(loaded, STRATEGY_LABELS, STRATEGY_COLORS):
            generations, median, q25, q75 = summarize(runs, endpoint)
            print(f"{environment}: {label}: {len(runs)}/{total} runs through generation {endpoint}")
            ax.plot(generations, median, color=color, linewidth=1.7, label=label)
            ax.fill_between(generations, q25, q75, color=color, alpha=.12)
        ax.set(title=environment, xlim=(0, endpoint), xlabel="Generations", ylabel="Training fitness")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=True)
    save(fig, "mutation_training_comparison_all_runs.pdf")


def plot_individual_mutations():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (title, experiments) in zip(axes, INDIVIDUAL_EXPERIMENTS.items()):
        loaded = [(label, *load_experiment(name, "best_fitness")) for label, name in experiments]
        endpoint = shared_endpoint([(runs, total) for _, runs, total in loaded])
        for (label, runs, total), color in zip(loaded, INDIVIDUAL_COLORS):
            generations, median, q25, q75 = summarize(runs, endpoint)
            print(f"{title}: {label}: {len(runs)}/{total} runs through generation {endpoint}")
            ax.plot(generations, median, label=label, color=color, linewidth=1.7)
            ax.fill_between(generations, q25, q75, color=color, alpha=.10)
        ax.set(title=title, xlim=(0, endpoint), xlabel="Generations", ylabel="Training fitness")
    axes[0].legend(frameon=True, fontsize=9)
    save(fig, "individual_mutations_vs_all_training.pdf")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_strategies()
    plot_individual_mutations()
