"""Plot the shared Mountain Car history and both transfer trajectories."""

import argparse
import csv
import re
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true",
                        help="Regenerate plots from the saved per-seed CSV")
    parser.add_argument("--allow-missing-seeds", action="store_true",
                        help="Skip missing seed logs")
    parser.add_argument("--inherited-only", action="store_true",
                        help="Plot only gene-encoded starting rates")
    parser.add_argument("--variant", choices=("execution_modified_rates", "fixed_individual", "fixed_rates"),
                        default="execution_modified_rates", help="Experiment mutation-rate mechanism")
    args = parser.parse_args()
    output = (OUTPUT if args.variant == "execution_modified_rates" else
              ROOT / f"output/mountaincar_to_pendulum_acrobot_{args.variant}")
    rates = RATES if args.variant == "execution_modified_rates" else RATES[:4]
    inherited_only = args.inherited_only or args.variant != "execution_modified_rates"
    if args.from_csv:
        with (output / "elite_mutation_rates_by_seed.csv").open(newline="") as handle:
            rows = [{key: value if key == "task" else
                     int(value) if key in ("seed", "generation") else float(value)
                     for key, value in row.items()} for row in csv.DictReader(handle)]
    else:
        rows = []
        for task, label, color, threshold in TASKS:
            for seed in range(1, 21):
                if task == "mountaincar_continuous":
                    path = ROOT / f"experiments/mountaincar_continuous_{args.variant}/logs/selection/selection.{seed}.0.csv"
                else:
                    path = ROOT / f"experiments/mountaincar_continuous_to_{task}_{args.variant}/seed_{seed}/{task}/logs/selection/selection.{seed}.0.csv"
                if not path.exists() and args.allow_missing_seeds:
                    print(f"Skipping missing {label} seed {seed}: {path}")
                    continue
                static_rates = {}
                if args.variant == "fixed_rates":
                    parameters = path.parents[2] / "parameters.yaml"
                    if not parameters.exists():
                        parameters = ROOT / "configs/mountaincar_continuous_fixed_rates.yaml"
                    config_text = parameters.read_text()
                    for rate in rates:
                        match = re.search(rf"^\s*p_instructions_{rate}:\s*([0-9.eE+-]+)", config_text, re.MULTILINE)
                        if match is None:
                            raise ValueError(f"Missing configured {rate} rate in {parameters}")
                        static_rates[rate] = float(match.group(1))
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
                                     **{f"{when}_{rate}": (static_rates[rate] if static_rates else float(row[f"elite_avg_{when}_rate_{rate}"]))
                                        for rate in rates for when in ("start", "output")}})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "elite_mutation_rates_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(rates) + 1, 1, figsize=(11, 15.5 if len(rates) == 5 else 13), sharex=True, constrained_layout=True)
    for task, label, color, threshold in TASKS:
        selected = [row for row in rows if row["task"] == task]
        seeds = sorted({row["seed"] for row in selected})
        seed_indexes = {seed: i for i, seed in enumerate(seeds)}
        generations = sorted({row["generation"] for row in selected})
        indexes = {generation: i for i, generation in enumerate(generations)}
        for ax, key, style in [(axes[0], "normalized_score", "-")] + [
            (ax, f"{when}_{rate}", "--" if when == "start" and not inherited_only else "-")
            for ax, rate in zip(axes[1:], rates)
            for when in (("start",) if inherited_only else ("start", "output"))
        ]:
            values = np.full((len(seeds), len(generations)), np.nan)
            for row in selected:
                values[seed_indexes[row["seed"]], indexes[row["generation"]]] = row[key]
            if not np.isfinite(values).all():
                raise ValueError(f"Incomplete seed coverage for {task}: {key}")
            median = np.median(values, axis=0)
            q1, q3 = np.quantile(values, [.25, .75], axis=0)
            plot_generations = np.asarray(generations)
            ax.plot(plot_generations, median, color=color, linestyle=style, linewidth=1.5)
            ax.fill_between(plot_generations, q1, q3, color=color, alpha=.10)
        print(f"{label}: {len(seeds)} seeds, generations {generations[0]}-{generations[-1]}")

    axes[0].axhline(1, color="#333333", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Normalized score")
    handles = [Line2D([0], [0], color=color, label=label)
               for _, label, color, _ in TASKS]
    axes[0].legend(handles=handles, loc="lower right", ncol=3, fontsize=9)
    for ax, rate in zip(axes[1:], rates):
        ax.set_ylabel(rate.capitalize() + " rate")
        ax.set_ylim(0, 1)
    if not inherited_only:
        axes[1].legend(handles=[Line2D([0], [0], color="black", linestyle="--", label="Inherited constants"),
                               Line2D([0], [0], color="black", label="Execution outputs")],
                       loc="upper right", ncol=2, fontsize=9)
    for ax in axes:
        ax.axvline(2000.5, color="#555555", linewidth=1, alpha=.8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Generation (transfer after 2,000)")
    fig.suptitle("Mountain Car → Pendulum / Acrobot\n"
                 + ("Static configured mutation rates\n" if args.variant == "fixed_rates" else
                    "Inherited mutation-rate genes (fixed_individual)\n" if args.variant == "fixed_individual" else
                    "Gene-encoded inherited rates (before execution)\n" if inherited_only else ""),
                 fontsize=12)
    basename = ("elite_static_mutation_rates" if args.variant == "fixed_rates" else
                "elite_inherited_mutation_rates" if inherited_only else "elite_mutation_rates")
    fig.savefig(output / f"{basename}.png", dpi=180)
    fig.savefig(output / f"{basename}.pdf")
    plt.close(fig)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
