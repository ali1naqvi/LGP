"""Join Pendulum and transferred Acrobot elite mutation rates by generation."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/pendulum_execution_modified_rates_no_limit/logs/selection"
TRANSFER = ROOT / "pendulum_to_acrobot_execution_modified_rates"
OUTPUT = ROOT / "output/pendulum_to_acrobot_mutation_rates"
RATES = ("swap", "delete", "add", "mutate", "decoy")
SOLVED_THRESHOLDS = {"pendulum": -200.0, "acrobot": -100.0}


def records(path, seed, stage):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            generation = int(row["generation"])
            if stage == "pendulum" and generation > 1000:
                break
            yield {
                "seed": seed,
                "task": stage,
                "generation": generation,
                "best_fitness": float(row["best_fitness"]),
                "normalized_score": 1 + (float(row["best_fitness"]) - SOLVED_THRESHOLDS[stage]) / abs(SOLVED_THRESHOLDS[stage]),
                **{f"{when}_{rate}": float(row[f"elite_avg_{when}_rate_{rate}"])
                   for rate in RATES for when in ("start", "output")},
            }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(1, 21):
        source = SOURCE / f"selection.{seed}.0.csv"
        target = TRANSFER / f"seed_{seed}/acrobot/logs/selection/selection.{seed}.0.csv"
        rows.extend(records(source, seed, "pendulum"))
        rows.extend(records(target, seed, "acrobot"))

    columns = ["seed", "task", "generation", "best_fitness", "normalized_score"] + [f"{when}_{rate}"
               for rate in RATES for when in ("start", "output")]
    with (OUTPUT / "elite_mutation_rates_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(6, 1, figsize=(11, 15.5), sharex=True, constrained_layout=True)
    score_ax = axes[0]
    for task, color in (("pendulum", "#2563a6"), ("acrobot", "#df6b20")):
        task_rows = [row for row in rows if row["task"] == task]
        generations = sorted({row["generation"] for row in task_rows})
        lookup = {(row["seed"], row["generation"]): row["normalized_score"] for row in task_rows}
        scores = np.array([[lookup.get((seed, generation), np.nan) for generation in generations]
                           for seed in range(1, 21)])
        score_ax.plot(generations, np.nanmedian(scores, axis=0), color=color, label=task.capitalize())
        score_ax.fill_between(generations, np.nanquantile(scores, .25, axis=0),
                              np.nanquantile(scores, .75, axis=0), color=color, alpha=.13)
    score_ax.axhline(1, color="#333", linestyle=":", label="Solved threshold (1)")
    score_ax.set_ylabel("Normalized score")
    score_ax.set_title("Best training return: 1 + (return - threshold) / |threshold|\n"
                       "Pendulum threshold: -200 (chosen); Acrobot threshold: -100", fontsize=10)
    score_ax.legend(loc="lower right", ncol=3, fontsize=9)
    score_ax.axvline(1000.5, color="#555", linewidth=1, alpha=.8)
    score_ax.grid(alpha=.2)
    for ax, rate in zip(axes[1:], RATES):
        for when, color, style in (("start", "#2563a6", "--"),
                                   ("output", "#df6b20", "-")):
            values = np.full((20, 5001), np.nan)
            for row in rows:
                values[row["seed"] - 1, row["generation"]] = row[f"{when}_{rate}"]
            count = np.isfinite(values).sum(axis=0)
            median = np.nanmedian(values, axis=0)
            q1 = np.nanquantile(values, .25, axis=0)
            q3 = np.nanquantile(values, .75, axis=0)
            valid = count > 0
            x = np.arange(5001)[valid]
            ax.plot(x, median[valid], color=color, linestyle=style, label=when.capitalize(), linewidth=1.5)
            ax.fill_between(x, q1[valid], q3[valid], color=color, alpha=.13)
        ax.axvline(1000.5, color="#555", linewidth=1, alpha=.8)
        ax.set_ylabel(rate.capitalize() + " rate")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
    axes[1].legend(loc="upper right", ncol=2)
    fig.suptitle("Pendulum → Acrobot transfer: normalized score and elite mutation rates\nMedian and interquartile range across 20 seeds")
    axes[-1].set_xlabel("Generation (transfer after 1,000)")
    fig.savefig(OUTPUT / "elite_mutation_rates.png", dpi=180)
    fig.savefig(OUTPUT / "elite_mutation_rates.pdf")
    plt.close(fig)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
