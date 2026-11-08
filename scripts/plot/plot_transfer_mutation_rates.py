"""Plot source training and both transfers for each starting environment."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = ("swap", "delete", "add", "mutate", "decoy")
# Source training is always drawn grey. The two destinations use the same blue and orange.
TASKS = (
    ("mountaincar_continuous", "Mountain Car", "#2563a6", 90.0),
    ("pendulum", "Pendulum", "#2563a6", -200.0),
    ("acrobot", "Acrobot", "#df6b20", -100.0),
)
SOURCE_COLOR = "#666666"
DESTINATION_COLORS = ("#2563a6", "#df6b20")
# Populations are trained for this many generations, then copied into the next environment.
TRANSFER_GENERATION = 5000


def transfer_directory(source, target, variant):
    """New Mountain Car transfers live under experiments/; older runs sit at the repo root."""
    name = f"{source}_to_{target}_{variant}"
    for base in (ROOT / "experiments", ROOT):
        path = base / name
        if any(path.glob("seed_*/transfer.json")):
            return path
    return None


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
    parser.add_argument("--source", choices=[task[0] for task in TASKS],
                        default="mountaincar_continuous")
    parser.add_argument("--target", choices=[task[0] for task in TASKS])
    args = parser.parse_args()
    targets = [task[0] for task in TASKS if task[0] != args.source]
    if args.target:
        targets = [args.target]
    recorded = {}
    directories = {}
    eligible = []
    for target in targets:
        directory = transfer_directory(args.source, target, args.variant)
        if directory is None:
            print(f"Skipping {args.source} → {target}: no transfer logs")
            continue
        points = {json.loads(path.read_text())["transfer_after_generation"]
                  for path in directory.glob("seed_*/transfer.json")}
        if len(points) != 1:
            raise ValueError(f"Inconsistent transfer generations for {target}: {points}")
        recorded[target] = points.pop()
        if recorded[target] == TRANSFER_GENERATION:
            directories[target] = directory
            eligible.append(target)
        else:
            print(f"Skipping {args.source} → {target}: checkpoint is generation {recorded[target]}, not {TRANSFER_GENERATION}")
    if not eligible:
        raise SystemExit(f"No {args.source} {args.variant} transfers from generation {TRANSFER_GENERATION}")
    source_task = next(t for t in TASKS if t[0] == args.source)
    tasks = [(source_task[0], source_task[1], SOURCE_COLOR, source_task[3])]
    tasks += [(task, label, color, threshold)
              for (task, label, _, threshold), color in zip(
                  (t for t in TASKS if t[0] in eligible), DESTINATION_COLORS)]
    source_config = f"{args.source}_{args.variant}"
    if source_config == "pendulum_fixed_individual":
        source_config += "_one_point"
    source_label = next(label for task, label, _, _ in TASKS if task == args.source)
    output = ROOT / "output/transfer_mutation_rates" / args.source / args.variant
    if args.target:
        output = output / f"to_{args.target}"
    rates = RATES if args.variant == "execution_modified_rates" else RATES[:4]
    inherited_only = args.inherited_only or args.variant != "execution_modified_rates"
    if args.from_csv:
        with (output / "elite_mutation_rates_by_seed.csv").open(newline="") as handle:
            rows = [{key: value if key == "task" else
                     int(value) if key in ("seed", "generation") else float(value)
                     for key, value in row.items()} for row in csv.DictReader(handle)]
    else:
        rows = []
        for task, label, color, threshold in tasks:
            if task == args.source:
                paths = sorted((ROOT / f"experiments/{source_config}/logs/selection").glob("selection.*.0.csv"))
            else:
                paths = sorted(directories[task].glob(f"seed_*/{task}/logs/selection/selection.*.0.csv"))
            if not paths:
                raise FileNotFoundError(f"No selection logs for {args.source}: {task}")
            for path in paths:
                seed = int(path.name.split(".")[1])
                static_rates = {}
                if args.variant == "fixed_rates":
                    parameters = path.parents[2] / "parameters.yaml"
                    if not parameters.exists():
                        parameters = ROOT / f"configs/{task}_fixed_rates.yaml"
                    config_text = parameters.read_text()
                    for rate in rates:
                        match = re.search(rf"^\s*p_instructions_{rate}:\s*([0-9.eE+-]+)", config_text, re.MULTILINE)
                        if match is None:
                            raise ValueError(f"Missing configured {rate} rate in {parameters}")
                        static_rates[rate] = float(match.group(1))
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        generation = int(row["generation"])
                        # Source stays on the axis through the checkpoint generation.
                        # Destination logs continue that counter, so they start at 5001.
                        if task == args.source and generation > TRANSFER_GENERATION:
                            continue
                        if task != args.source and generation <= TRANSFER_GENERATION:
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
    for task, label, color, threshold in tasks:
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
            median = np.nanmedian(values, axis=0)
            q1, q3 = np.nanquantile(values, [.25, .75], axis=0)
            plot_generations = np.asarray(generations)
            ax.plot(plot_generations, median, color=color, linestyle=style, linewidth=1.5)
            ax.fill_between(plot_generations, q1, q3, color=color, alpha=.10)
        print(f"{label}: {len(seeds)} seeds, generations {generations[0]}-{generations[-1]}; "
              f"coverage {np.isfinite(values).sum(axis=0).min()}-{len(seeds)} seeds/generation")

    axes[0].axhline(1, color="#333333", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Normalized score")
    handles = [Line2D([0], [0], color=color, label=label)
               for task, label, color, _ in tasks]
    axes[0].legend(handles=handles, loc="lower right", ncol=3, fontsize=9)
    for ax, rate in zip(axes[1:], rates):
        ax.set_ylabel(rate.capitalize() + " rate")
        ax.set_ylim(0, 1)
    if not inherited_only:
        axes[1].legend(handles=[Line2D([0], [0], color="black", linestyle="--", label="Inherited constants"),
                               Line2D([0], [0], color="black", label="Execution outputs")],
                       loc="upper right", ncol=2, fontsize=9)
    for ax in axes:
        ax.axvline(TRANSFER_GENERATION + .5, color="#555555", linewidth=1, alpha=.8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel(f"Generation (transfer after {TRANSFER_GENERATION:,})")
    fig.suptitle(f"{source_label} → {' / '.join(label for task, label, _, _ in tasks if task != args.source)}\nMedian and interquartile range across available seeds\n"
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
