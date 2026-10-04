"""Plot standalone Acrobot training and validation curves for all three variants."""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "acrobot_training"
VARIANTS = (
    ("Modified mutation rates", "acrobot_execution_modified_rates", "tab:green"),
    ("Fixed individual mutation rates", "acrobot_fixed_individual", "tab:blue"),
    ("Fixed mutation rates", "acrobot_fixed_rates", "tab:orange"),
)
EXPECTED_SEEDS = 20


def load_variant(experiment):
    selection_dir = ROOT / "experiments" / experiment / "logs" / "selection"
    paths = sorted(selection_dir.glob("selection.*.0.csv"))
    training_runs = []
    validation_runs = []
    for path in paths:
        training = {}
        validation = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                generation = int(row["generation"])
                training_fitness = float(row["best_fitness"])
                if np.isfinite(training_fitness):
                    training[generation] = training_fitness

                # The logger repeats the last validation value between
                # evaluations; retain only the scheduled evaluation rows.
                validation_fitness = float(row["validation_fitness"])
                if generation % 100 == 0 and validation_fitness != 0 and np.isfinite(validation_fitness):
                    validation[generation] = validation_fitness
        if training:
            training_runs.append(training)
        if validation:
            validation_runs.append(validation)

    if not paths:
        raise FileNotFoundError(f"No selection logs for {experiment} in {selection_dir}")
    return len(paths), training_runs, validation_runs


def summarize(runs):
    generations = sorted(set().union(*(run.keys() for run in runs)))
    summary = []
    for generation in generations:
        values = [run[generation] for run in runs if generation in run]
        summary.append({
            "generation": generation,
            "n_runs": len(values),
            "median": statistics.median(values),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        })
    return summary


def plot_metric(ax, metric, ylabel, data):
    for label, _, color, summaries, total_logs in data:
        x = [row["generation"] for row in summaries]
        median = [row["median"] for row in summaries]
        q25 = [row["q25"] for row in summaries]
        q75 = [row["q75"] for row in summaries]
        terminal = next((row for row in reversed(summaries) if row["generation"] == 5000), None)
        n_terminal = terminal["n_runs"] if terminal else 0
        short_label = {
            "Modified mutation rates": "Modified rates",
            "Fixed individual mutation rates": "Fixed individual",
            "Fixed mutation rates": "Fixed rates",
        }[label]
        ax.plot(x, median, color=color, linewidth=1.7,
                label=f"{short_label} · {n_terminal}/{EXPECTED_SEEDS} at gen 5000")
        ax.fill_between(x, q25, q75, color=color, alpha=0.12)

    ax.axhline(-100, color="0.35", linewidth=1, linestyle="--", alpha=0.8, label="Fitness −100")
    ax.set(title=f"Acrobot · {metric}", xlabel="Generation", ylabel=ylabel, xlim=(0, 5000))
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = []
    validation_panels = []
    csv_rows = []
    for label, experiment, color in VARIANTS:
        total, training_runs, validation_runs = load_variant(experiment)
        training = summarize(training_runs)
        validation = summarize(validation_runs)
        print(f"{label}: {len(training_runs)}/{total} training runs, "
              f"{len(validation_runs)}/{total} validation runs; "
              f"generation-5000 training runs: "
              f"{sum(1 for run in training_runs if 5000 in run)}/{EXPECTED_SEEDS}")
        loaded.append((label, experiment, color, training, total))
        loaded_validation = (label, experiment, color, validation, total)
        for metric, summaries in (("best_fitness", training), ("validation_fitness", validation)):
            csv_rows.extend({"variant": label, "experiment": experiment, "metric": metric, **row}
                            for row in summaries)
        # Retain the same variant metadata in a second list for the validation panel.
        validation_panels.append(loaded_validation)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    plot_metric(axes[0], "training fitness", "Best training fitness", loaded)
    plot_metric(axes[1], "validation fitness", "Validation fitness", validation_panels)
    fig.savefig(OUTPUT / "fitness_curves.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "fitness_curves.png", dpi=180, bbox_inches="tight")
    with (OUTPUT / "fitness_summary.csv").open("w", newline="") as handle:
        fields = ("variant", "experiment", "metric", "generation", "n_runs", "median", "q25", "q75")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(OUTPUT / "fitness_curves.pdf")


if __name__ == "__main__":
    main()
