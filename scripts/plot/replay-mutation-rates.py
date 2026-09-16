#!/usr/bin/env python3

"""Replay each seed's final best agent and plot its S2-S6 rates by timestep."""

import csv
from pathlib import Path
import re
import shlex
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


REPO = Path(__file__).resolve().parents[2]
RECORD_EVERY_TIMESTEPS = 10  # Use 1, 5, 10, etc.
TEST_SETUP = "mountaincar_continuous_execution_modified_rates"
EXPERIMENT_DIR = REPO / "experiments" / TEST_SETUP
PARAMETERS_FILE = REPO / "configs" / f"{TEST_SETUP}.yaml"
OUTPUT_PDF = REPO / "output/pdf" / f"{TEST_SETUP}_mutation_rates.pdf"
# MountainCarContinuous currently defines its replay evaluation count in the
# task itself rather than in YAML.
EPISODES = 100
TIMESTEPS_PER_EPISODE = 200
MPI_PROCESSES = 12
REPLAY_EXECUTABLE = REPO / "build/release/experiments/TPGExperimentMPI"

RATE_NAMES = ("swap", "delete", "add", "mutate", "redundancy")
RATE_LABELS = {
    "swap": "Swap",
    "delete": "Delete",
    "add": "Add",
    "mutate": "Point mutation",
    "redundancy": "Redundancy",
}
RATE_COLOURS = {
    "swap": "#0072B2",
    "delete": "#D55E00",
    "add": "#009E73",
    "mutate": "#CC79A7",
    "redundancy": "#E69F00",
}
EPSILON = 1e-6


def discover_seeds() -> list[int]:
    pattern = re.compile(r"^selection\.(\d+)\..*\.csv$")
    seeds = sorted(
        {
            int(match.group(1))
            for path in (EXPERIMENT_DIR / "logs/selection").glob("selection.*.*.csv")
            if (match := pattern.match(path.name))
        }
    )
    if not seeds:
        raise FileNotFoundError(
            f"No selection CSV files found in {EXPERIMENT_DIR / 'logs/selection'}"
        )
    return seeds


def stable_sigmoid(raw: np.ndarray) -> np.ndarray:
    """Match SelfModifyingRawTendencyToProbability in src/engine/misc.h."""
    raw = np.asarray(raw, dtype=float)
    probability = np.empty_like(raw)
    positive = raw >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exp_raw = np.exp(raw[~positive])
    probability[~positive] = exp_raw / (1.0 + exp_raw)
    return EPSILON + (1.0 - 2.0 * EPSILON) * probability


def final_best_agent(seed: int) -> tuple[int, float]:
    pattern = re.compile(rf"^selection\.{seed}\.(\d+)\.csv$")
    files = sorted(
        (EXPERIMENT_DIR / "logs/selection").glob(f"selection.{seed}.*.csv"),
        key=lambda path: int(match.group(1))
        if (match := pattern.match(path.name))
        else -1,
    )
    if not files:
        raise FileNotFoundError(f"No selection CSV found for seed {seed}")
    with files[-1].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Selection CSV is empty: {files[-1]}")
    # Match the normal replay command: highest training fitness, with the first
    # generation that reached it used as the deterministic tie-breaker.
    best = min(
        rows,
        key=lambda row: (-float(row["best_fitness"]), int(float(row["generation"]))),
    )
    return int(float(best["team_id"])), float(best["best_fitness"])


def latest_checkpoint(seed: int) -> tuple[int, int]:
    # Match the normal replay command, which restores the most recent
    # completed training (phase-0) checkpoint.
    phase = 0
    pattern = re.compile(rf"^cp\.(\d+)\.{seed}\.{phase}\.rslt$")
    candidates = []
    for path in (EXPERIMENT_DIR / "checkpoints").glob(f"cp.*.{seed}.{phase}.rslt"):
        match = pattern.match(path.name)
        if match and path.read_text(errors="ignore").rstrip().endswith("end"):
            candidates.append(int(match.group(1)))
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint found for seed {seed}")
    return max(candidates), phase


def replay(seed: int, team_id: int) -> Path:
    if not 1 <= RECORD_EVERY_TIMESTEPS <= TIMESTEPS_PER_EPISODE:
        raise ValueError(
            "RECORD_EVERY_TIMESTEPS must be between 1 and TIMESTEPS_PER_EPISODE"
        )
    generation, phase = latest_checkpoint(seed)
    executable = REPLAY_EXECUTABLE
    if not executable.is_file():
        raise FileNotFoundError(f"Build the replay executable first: {executable}")
    command = [
        "mpirun", "--oversubscribe", "-np", str(MPI_PROCESSES), str(executable),
        f"parameters_file={PARAMETERS_FILE}", f"seed_tpg={seed}", "seed_aux=42",
        "start_from_checkpoint=1", f"checkpoint_in_phase={phase}",
        f"checkpoint_in_t={generation}", "replay=1", "animate=0",
        f"id_to_replay={team_id}", "task_to_replay=0",
    ]
    print("Running:", shlex.join(command))
    log_dir = EXPERIMENT_DIR / "logs/misc"
    log_dir.mkdir(parents=True, exist_ok=True)
    trace = log_dir / f"mutation_rates.{seed}.{team_id}.csv"
    # Do not mistake output from an older replay for a newly generated trace.
    trace.unlink(missing_ok=True)
    with (log_dir / f"mutation_replay.{seed}.stdout").open("w") as stdout, \
         (log_dir / f"mutation_replay.{seed}.stderr").open("w") as stderr:
        subprocess.run(command, cwd=EXPERIMENT_DIR, stdout=stdout, stderr=stderr, check=True)
    if not trace.is_file():
        raise FileNotFoundError(
            f"Replay completed without producing {trace}. Verify that the executable "
            "was rebuilt with LogReplaySelfModifyingRates enabled."
        )
    return trace


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Mutation-rate trace is empty: {path}")

    required_columns = {
        "episode",
        "timestep",
        *(
            f"{version}_{rate}"
            for rate in RATE_NAMES
            for version in ("start", "output")
        ),
    }
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(
            f"{path} is missing columns from the current S2-S6 trace format: "
            f"{', '.join(sorted(missing_columns))}"
        )

    sample_steps = np.arange(
        RECORD_EVERY_TIMESTEPS,
        TIMESTEPS_PER_EPISODE + 1,
        RECORD_EVERY_TIMESTEPS,
    )
    timesteps = np.concatenate(
        [episode * TIMESTEPS_PER_EPISODE + sample_steps for episode in range(EPISODES)]
    )
    data: dict[str, np.ndarray] = {"timestep": timesteps}
    for rate in RATE_NAMES:
        for version in ("start", "output"):
            data[f"{version}_{rate}"] = np.full(timesteps.size, np.nan)

    samples_per_episode = sample_steps.size
    seen: set[tuple[int, int]] = set()
    for row in rows:
        episode = int(row["episode"])
        timestep = int(row["timestep"])
        if not 0 <= episode < EPISODES:
            raise ValueError(f"{path} contains out-of-range episode {episode}")
        if not 1 <= timestep <= TIMESTEPS_PER_EPISODE:
            raise ValueError(f"{path} contains out-of-range timestep {timestep}")
        key = (episode, timestep)
        if key in seen:
            raise ValueError(f"{path} contains duplicate sample {key}")
        seen.add(key)
        if timestep % RECORD_EVERY_TIMESTEPS:
            continue
        index = episode * samples_per_episode + timestep // RECORD_EVERY_TIMESTEPS - 1
        for rate in RATE_NAMES:
            for version in ("start", "output"):
                data[f"{version}_{rate}"][index] = float(row[f"{version}_{rate}"])

    if not any(timestep % RECORD_EVERY_TIMESTEPS == 0 for _, timestep in seen):
        raise ValueError(
            f"{path} contains no samples at the requested "
            f"{RECORD_EVERY_TIMESTEPS}-timestep interval"
        )
    for rate in RATE_NAMES:
        for version in ("start", "output"):
            data[f"{version}_{rate}"] = stable_sigmoid(data[f"{version}_{rate}"])
    return data


def first_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else np.nan


def last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else np.nan


def plot_page(
    pdf: PdfPages,
    data: dict[str, np.ndarray],
    title: str,
    std_data: dict[str, np.ndarray] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    output_timesteps = data["timestep"]
    x = np.concatenate(([0], output_timesteps))
    for rate in RATE_NAMES:
        colour = RATE_COLOURS[rate]
        # The evolved constant is the initial condition at t=0. The remaining
        # points are the working-register outputs observed during replay.
        y = np.concatenate(
            ([first_finite(data[f"start_{rate}"])], data[f"output_{rate}"])
        )
        if std_data is not None:
            y_std = np.concatenate(
                ([first_finite(std_data[f"start_{rate}"])], std_data[f"output_{rate}"])
            )
            ax.fill_between(
                x,
                np.clip(y - y_std, 0, 1),
                np.clip(y + y_std, 0, 1),
                color=colour,
                alpha=0.16,
                linewidth=0,
            )
        ax.plot(x, y, color=colour, linewidth=2.0, label=RATE_LABELS[rate])
        ax.scatter([0], [y[0]], color=colour, s=28, zorder=3)
    for boundary in range(
        TIMESTEPS_PER_EPISODE,
        EPISODES * TIMESTEPS_PER_EPISODE,
        TIMESTEPS_PER_EPISODE,
    ):
        ax.axvline(boundary + 0.5, color="0.75", linewidth=0.8)
    ax.set(title=title, xlabel="Timestep across episodes", ylabel="Mutation probability")
    ax.set_xlim(0, EPISODES * TIMESTEPS_PER_EPISODE)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9, loc="upper right")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_stats_page(
    pdf: PdfPages,
    traces: list[dict[str, np.ndarray]],
    labels: list[tuple[int, int, float]],
) -> None:
    rows = []
    for data, (seed, team_id, _fitness) in zip(traces, labels):
        row = [str(seed), str(team_id)]
        for rate in RATE_NAMES:
            row.extend(
                [
                    f"{first_finite(data[f'start_{rate}']):.5f}",
                    f"{last_finite(data[f'output_{rate}']):.5f}",
                ]
            )
        rows.append(row)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    ax.set_title("Best-agent mutation-rate start and final values", pad=18)
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Seed", "Team",
            "Swap S", "Swap F", "Delete S", "Delete F",
            "Add S", "Add F", "Mutate S", "Mutate F",
            "Redund. S", "Redund. F",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.06, 0.11] + [0.075] * (2 * len(RATE_NAMES)),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, min(1.45, max(0.62, 22 / max(len(rows), 1))))
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8EDF3")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F8FA")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    if not PARAMETERS_FILE.is_file():
        print(f"Parameters file not found: {PARAMETERS_FILE}")
        return
    if not REPLAY_EXECUTABLE.is_file():
        print(f"Replay executable not found: {REPLAY_EXECUTABLE}")
        return
    if b"mutation_rates." not in REPLAY_EXECUTABLE.read_bytes():
        print(
            "Replay executable does not contain per-timestep mutation-rate tracing. "
            "No seeds were run. The stock replay output does not expose these values; "
            "the engine-side logger must be restored and the executable rebuilt."
        )
        return

    try:
        seeds = discover_seeds()
    except FileNotFoundError as error:
        print(error)
        return

    traces = []
    labels = []
    for seed in seeds:
        try:
            team_id, fitness = final_best_agent(seed)
            traces.append(load_trace(replay(seed, team_id)))
            labels.append((seed, team_id, fitness))
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
            print(f"Skipping seed {seed}: {error}")
    if not traces:
        print("No seeds produced usable mutation-rate traces; no PDF was written.")
        return

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        for data, (seed, _team_id, fitness) in zip(traces, labels):
            plot_page(pdf, data, f"Best agent for seed {seed} (fitness {fitness:.6g})")
        median = {"timestep": traces[0]["timestep"]}
        standard_deviation = {"timestep": traces[0]["timestep"]}
        for rate in RATE_NAMES:
            for version in ("start", "output"):
                values = np.stack(
                    [trace[f"{version}_{rate}"] for trace in traces]
                )
                valid = np.sum(np.isfinite(values), axis=0)
                totals = np.nansum(values, axis=0)
                means = np.divide(
                    totals,
                    valid,
                    out=np.full(values.shape[1], np.nan),
                    where=valid > 0,
                )
                variance = np.divide(
                    np.nansum((values - means) ** 2, axis=0),
                    valid,
                    out=np.full(values.shape[1], np.nan),
                    where=valid > 0,
                )
                median_values = np.full(values.shape[1], np.nan)
                for index in np.flatnonzero(valid):
                    column = values[:, index]
                    median_values[index] = np.median(column[np.isfinite(column)])
                median[f"{version}_{rate}"] = median_values
                standard_deviation[f"{version}_{rate}"] = np.sqrt(variance)
        plot_page(
            pdf,
            median,
            f"Median of {len(traces)} best agents (shading: +/- 1 SD)",
            std_data=standard_deviation,
        )
        plot_stats_page(pdf, traces, labels)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
