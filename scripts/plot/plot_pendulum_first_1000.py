"""Pendulum training and instruction plots through generation 1000."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_mutation_validation_comparison import (  # noqa: E402
    COLORS, EXPERIMENTS, LABELS, ROOT, load_experiment, summarize,
)
from plot_individual_mutation_validation import (  # noqa: E402
    COLORS as INDIVIDUAL_COLORS,
    EXPERIMENTS as INDIVIDUAL_EXPERIMENTS,
)


ENDPOINT = 1000
OUTPUT = ROOT / "output" / "pendulum_first_1000"


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_curve(ax, name, metric, label, color):
    runs, total = load_experiment(name, metric)
    if any(max(run) < ENDPOINT for run in runs):
        raise ValueError(f"{name}: a run ends before generation {ENDPOINT}")
    generations, median, q25, q75 = summarize(runs, ENDPOINT)
    ax.plot(generations, median, label=label, color=color, linewidth=1.7)
    ax.fill_between(generations, q25, q75, color=color, alpha=.12)
    print(f"{name}: {len(runs)}/{total} runs through generation {ENDPOINT}")


def plot_strategies(metric, stem, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, label, color in zip(EXPERIMENTS["Pendulum"], LABELS, COLORS):
        add_curve(ax, name, metric, label, color)
    ax.set(title=f"Pendulum · {ylabel}", xlim=(0, ENDPOINT),
           xlabel="Generation", ylabel=ylabel)
    ax.legend(frameon=True)
    save(fig, stem)


def plot_individual():
    fig, ax = plt.subplots(figsize=(8, 5))
    experiments = INDIVIDUAL_EXPERIMENTS["Pendulum · fixed individual mutation rates"]
    for (label, name), color in zip(experiments, INDIVIDUAL_COLORS):
        add_curve(ax, name, "best_fitness", label, color)
    ax.set(title="Pendulum · individual mutation training fitness",
           xlim=(0, ENDPOINT), xlabel="Generation", ylabel="Training fitness")
    ax.legend(frameon=True)
    save(fig, "individual_mutations_training")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_strategies("best_fitness", "strategy_training", "Training fitness")
    plot_individual()
    plot_strategies("program_instruction_count", "total_instructions", "Total instructions")
