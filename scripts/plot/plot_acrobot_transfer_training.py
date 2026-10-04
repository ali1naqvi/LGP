"""Plot training and validation fitness for Pendulum-to-Acrobot transfer runs."""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "pendulum_to_acrobot_fixed_rates"
OUTPUT = ROOT / "output" / "acrobot_transfer_training"
SEEDS = range(1, 21)


def load_runs():
    training_runs = []
    validation_runs = []
    missing = []
    for seed in SEEDS:
        path = EXPERIMENT / f"seed_{seed}" / "acrobot" / "logs" / "selection" / f"selection.{seed}.0.csv"
        if not path.is_file():
            missing.append(seed)
            continue

        training = {}
        validation = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                generation = int(row["generation"])
                fitness = float(row["best_fitness"])
                if np.isfinite(fitness):
                    training[generation] = fitness

                # Validation is scheduled every 100 generations. Keep the
                # evaluation points only; the logger carries old values in
                # intervening generation rows.
                validation_fitness = float(row["validation_fitness"])
                if generation % 100 == 0 and validation_fitness != 0 and np.isfinite(validation_fitness):
                    validation[generation] = validation_fitness

        if training:
            training_runs.append((seed, training))
        if validation:
            validation_runs.append((seed, validation))

    if not training_runs:
        raise ValueError(f"No Acrobot training logs found under {EXPERIMENT}")
    print(f"Training curves: {len(training_runs)}/20 runs; missing/empty seeds: {missing}")
    print(f"Validation curves: {len(validation_runs)}/20 runs")
    return training_runs, validation_runs


def summarize(runs):
    generations = sorted(set().union(*(values.keys() for _, values in runs)))
    rows = []
    for generation in generations:
        scores = [values[generation] for _, values in runs if generation in values]
        rows.append({
            "generation": generation,
            "n_runs": len(scores),
            "median": statistics.median(scores),
            "q25": float(np.quantile(scores, 0.25)),
            "q75": float(np.quantile(scores, 0.75)),
        })
    return rows


def add_curve(ax, rows, title, ylabel, run_count):
    generations = [row["generation"] for row in rows]
    medians = [row["median"] for row in rows]
    q25 = [row["q25"] for row in rows]
    q75 = [row["q75"] for row in rows]
    ax.plot(generations, medians, color="tab:blue", linewidth=1.8, label="Median")
    ax.fill_between(generations, q25, q75, color="tab:blue", alpha=0.18, label="25th–75th percentile")
    ax.axhline(-100, color="tab:red", linewidth=1, linestyle="--", alpha=0.8, label="Fitness −100")
    ax.set(title=title, xlabel="Acrobot generation", ylabel=ylabel, xlim=(1000, 5000))
    ax.legend(frameon=False, loc="best")
    ax.text(0.99, 0.02, f"{run_count}/20 runs; generations {min(generations)}–{max(generations)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    training_runs, validation_runs = load_runs()
    training = summarize(training_runs)
    validation = summarize(validation_runs)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)
    add_curve(axes[0], training, "Pendulum → Acrobot · Training fitness", "Best training fitness", len(training_runs))
    add_curve(axes[1], validation, "Pendulum → Acrobot · Validation fitness", "Validation fitness", len(validation_runs))
    fig.savefig(OUTPUT / "fitness_curves.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "fitness_curves.png", dpi=180, bbox_inches="tight")
    with (OUTPUT / "fitness_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "generation", "n_runs", "median", "q25", "q75"))
        writer.writeheader()
        writer.writerows({"metric": metric, **row} for metric, rows in (
            ("best_fitness", training), ("validation_fitness", validation)
        ) for row in rows)
    print(OUTPUT / "fitness_curves.pdf")


if __name__ == "__main__":
    main()
