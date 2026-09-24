import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RATE_NAMES = {
    "swap": "Swap",
    "delete": "Delete",
    "add": "Add",
    "mutate": "One-point mutation",
    "decoy": "Decoy",
}

EPSILON = 1e-6


def load_runs(selection_dir, max_generation):
    files = sorted(selection_dir.glob("selection.*.0.csv"))
    if not files:
        raise FileNotFoundError(f"No selection logs found in {selection_dir}")

    columns = [
        f"elite_avg_{stage}_rate_{rate}"
        for rate in RATE_NAMES
        for stage in ("start", "output")
    ]
    runs = []
    generations = set()

    for path in files:
        run = {}
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in columns if column not in reader.fieldnames]
            if missing:
                print(f"Warning: {path.name} lacks {len(missing)} newer rate columns")

            for row in reader:
                generation = int(row["generation"])
                if generation > max_generation:
                    continue
                run[generation] = {
                    column: (
                        float(row[column])
                        if column in row and row[column] not in (None, "")
                        else np.nan
                    )
                    for column in columns
                }
                generations.add(generation)
        runs.append(run)

    return runs, np.asarray(sorted(generations), dtype=int)


def raw_tendency_to_probability(value):
    if not math.isfinite(value):
        return EPSILON
    if value >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-value))
    else:
        exp_value = math.exp(value)
        sigmoid = exp_value / (1.0 + exp_value)
    return EPSILON + (1.0 - 2.0 * EPSILON) * sigmoid


def read_checkpoint_agent_rates(path, best_team_id):
    start_rates = {}
    output_rates = {}
    actions = {}
    teams = {}
    four_rate_layout = set()

    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split(":")
            if fields[0] == "MemoryEigen" and fields[2] == "0":
                program_id = int(fields[1])
                values = [float(value) for value in fields[4:]]
                if len(values) >= 7:
                    start_rates[program_id] = values[2:8]
            elif fields[0] == "self_modifying_state":
                output_rates[int(fields[1])] = [float(value) for value in fields[2:8]]
            elif fields[0] == "RegisterMachine":
                actions[int(fields[1])] = int(fields[3])
                if "SL4" in fields:
                    four_rate_layout.add(int(fields[1]))
            elif fields[0] == "team":
                teams[int(fields[1])] = [int(value) for value in fields[7:]]

    # Old checkpoints include redundancy at offset 4 and decoy at offset 5.
    # New SL4 checkpoints place decoy immediately after the four active rates.
    for program_id in start_rates.keys() & output_rates.keys():
        indices = [0, 1, 2, 3, 4 if program_id in four_rate_layout else 5]
        if max(indices) >= min(len(start_rates[program_id]), len(output_rates[program_id])):
            del start_rates[program_id]
            del output_rates[program_id]
            continue
        start_rates[program_id] = [start_rates[program_id][i] for i in indices]
        output_rates[program_id] = [output_rates[program_id][i] for i in indices]

    program_ids = set()
    pending_teams = [best_team_id]
    visited_teams = set()
    while pending_teams:
        team_id = pending_teams.pop()
        if team_id in visited_teams:
            continue
        visited_teams.add(team_id)
        for program_id in teams.get(team_id, []):
            program_ids.add(program_id)
            action = actions.get(program_id, -1)
            if action >= 0 and action in teams:
                pending_teams.append(action)

    usable = sorted(program_ids & start_rates.keys() & output_rates.keys())
    if not usable:
        return None

    result = {}
    for rate_index, rate in enumerate(RATE_NAMES):
        start = [
            raw_tendency_to_probability(start_rates[program_id][rate_index])
            for program_id in usable
        ]
        output = [
            raw_tendency_to_probability(output_rates[program_id][rate_index])
            for program_id in usable
        ]
        result[f"elite_avg_start_rate_{rate}"] = float(np.mean(start))
        result[f"elite_avg_output_rate_{rate}"] = float(np.mean(output))
    return result


def load_best_runs(experiment_dir, max_generation):
    selection_dir = experiment_dir / "logs" / "selection"
    checkpoint_dir = experiment_dir / "checkpoints"
    files = sorted(selection_dir.glob("selection.*.0.csv"))
    runs = []
    generations = set()

    for selection_path in files:
        seed = selection_path.name.split(".")[1]
        best_teams = {}
        with selection_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                generation = int(row["generation"])
                if generation <= max_generation:
                    best_teams[generation] = int(row["team_id"])

        run = {}
        for checkpoint_path in checkpoint_dir.glob(f"cp.*.{seed}.0.rslt"):
            generation = int(checkpoint_path.name.split(".")[1])
            if generation > max_generation or generation not in best_teams:
                continue
            rates = read_checkpoint_agent_rates(
                checkpoint_path, best_teams[generation]
            )
            if rates is not None:
                run[generation] = rates
                generations.add(generation)
        if run:
            runs.append(run)

    if not runs:
        raise ValueError(f"No recoverable best-individual rates in {checkpoint_dir}")
    return runs, np.asarray(sorted(generations), dtype=int)


def summarize(runs, generations, column):
    values = np.full((len(runs), len(generations)), np.nan)
    for run_index, run in enumerate(runs):
        for generation_index, generation in enumerate(generations):
            if generation in run:
                values[run_index, generation_index] = run[generation][column]

    return np.nanmedian(values, axis=0), np.nanstd(values, axis=0)


def plot_rates(experiment, max_generation, output, scope):
    repo_root = Path(__file__).resolve().parents[3]
    experiment_dir = repo_root / "experiments" / experiment
    if scope == "best":
        runs, generations = load_best_runs(experiment_dir, max_generation)
    else:
        runs, generations = load_runs(
            experiment_dir / "logs" / "selection", max_generation
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True)

    styles = {
        "start": ("Inherited start", "tab:blue"),
        "output": ("Post-execution", "tab:orange"),
    }

    for ax, (rate, title) in zip(axes.flat, RATE_NAMES.items()):
        for stage, (label, color) in styles.items():
            column = f"elite_avg_{stage}_rate_{rate}"
            median, std = summarize(runs, generations, column)
            ax.plot(generations, median, color=color, linewidth=1.7, label=label)
            ax.fill_between(
                generations,
                median - std,
                median + std,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, max_generation)
        ax.margins(y=0.05)
        ax.grid(color="0.88", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.supxlabel("Generation", fontsize=11)
    fig.supylabel("Mutation probability", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    scope_title = "Best Individual" if scope == "best" else "Elite Population"
    fig.suptitle(
        f"{experiment.replace('_', ' ').title()} - {scope_title}",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.87))
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output} from {len(runs)} runs ({scope})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot elite inherited and post-execution mutation rates."
    )
    parser.add_argument("--experiment", default="pendulum_execution_modified_rates")
    parser.add_argument("--generations", type=int, default=5000)
    parser.add_argument("--output", default="mutation_rates.pdf")
    parser.add_argument("--scope", choices=("elite", "best"), default="elite")
    args = parser.parse_args()
    plot_rates(args.experiment, args.generations, args.output, args.scope)
