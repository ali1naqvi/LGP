"""Plot the shared Mountain Car history and both transfer trajectories."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/mountaincar_to_pendulum_acrobot_mutation_rates"
RATES = ("swap", "delete", "add", "mutate", "decoy")
TASKS = (
    ("mountaincar_continuous", "Mountain Car", "#666666", 90.0),
    ("pendulum", "Pendulum", "#2563a6", -200.0),
    ("acrobot", "Acrobot", "#df6b20", -100.0),
)


def main():
    rows = []
    for task, label, color, threshold in TASKS:
        for seed in range(1, 21):
            if task == "mountaincar_continuous":
                path = ROOT / f"experiments/mountaincar_continuous_execution_modified_rates/logs/selection/selection.{seed}.0.csv"
            else:
                path = ROOT / f"mountaincar_continuous_to_{task}_execution_modified_rates/seed_{seed}/{task}/logs/selection/selection.{seed}.0.csv"
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    generation = int(row["generation"])
                    if task == "mountaincar_continuous" and generation > 2000:
                        break
                    if task != "mountaincar_continuous" and generation <= 2000:
                        continue
                    score = float(row["best_fitness"])
                    rows.append({"seed": seed, "task": task, "generation": generation,
                                 "best_fitness": score,
                                 "normalized_score": 1 + (score - threshold) / abs(threshold),
                                 **{f"{when}_{rate}": float(row[f"elite_avg_{when}_rate_{rate}"])
                                    for rate in RATES for when in ("start", "output")}})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "elite_mutation_rates_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(6, 1, figsize=(11, 15.5), sharex=True, constrained_layout=True)
    for task, label, color, threshold in TASKS:
        selected = [row for row in rows if row["task"] == task]
        generations = sorted({row["generation"] for row in selected})
        indexes = {generation: i for i, generation in enumerate(generations)}
        for ax, key, style in [(axes[0], "normalized_score", "-")] + [
            (ax, f"{when}_{rate}", "--" if when == "start" else "-")
            for ax, rate in zip(axes[1:], RATES) for when in ("start", "output")
        ]:
            values = np.full((20, len(generations)), np.nan)
            for row in selected:
                values[row["seed"] - 1, indexes[row["generation"]]] = row[key]
            if not np.isfinite(values).all():
                raise ValueError(f"Incomplete seed coverage for {task}: {key}")
            median = np.median(values, axis=0)
            q1, q3 = np.quantile(values, [.25, .75], axis=0)
            # Historical MPI runs imported validation register states into the
            # training population. Omit only validation-generation output
            # samples. Connect the retained samples instead of inserting NaNs,
            # which break both the line and the interquartile band.
            plot_generations = np.asarray(generations)
            if key.startswith("output_"):
                keep = (plot_generations == 0) | (plot_generations % 100 != 0)
                plot_generations = plot_generations[keep]
                median, q1, q3 = median[keep], q1[keep], q3[keep]
            ax.plot(plot_generations, median, color=color, linestyle=style, linewidth=1.5)
            ax.fill_between(plot_generations, q1, q3, color=color, alpha=.10)
        print(f"{label}: 20 seeds, generations {generations[0]}-{generations[-1]}")

    axes[0].axhline(1, color="#333333", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Normalized score")
    handles = [Line2D([0], [0], color=color, label=label) for _, label, color, _ in TASKS]
    axes[0].legend(handles=handles, loc="lower right", ncol=3, fontsize=9)
    for ax, rate in zip(axes[1:], RATES):
        ax.set_ylabel(rate.capitalize() + " rate")
        ax.set_ylim(0, 1)
    axes[1].legend(handles=[Line2D([0], [0], color="black", linestyle="--", label="Inherited constants"),
                           Line2D([0], [0], color="black", label="Execution outputs")],
                   loc="upper right", ncol=2, fontsize=9)
    for ax in axes:
        ax.axvline(2000.5, color="#555555", linewidth=1, alpha=.8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Generation (transfer after 2,000)")
    fig.suptitle("Mountain Car → Pendulum / Acrobot\n"
                 "Output rates: validation samples omitted; adjacent points connected\n"
                 "Historical training may still reflect validation effects",
                 fontsize=12)
    fig.savefig(OUTPUT / "elite_mutation_rates.png", dpi=180)
    fig.savefig(OUTPUT / "elite_mutation_rates.pdf")
    plt.close(fig)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
