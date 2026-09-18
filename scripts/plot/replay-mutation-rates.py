#!/usr/bin/env python3

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

# Plot/replay controls.
RECORD_EVERY_TIMESTEPS = 1  # Use 1, 5, 10, etc.
EPISODES = 20
EPISODE_TIMESTEPS = 200  # Replay-only episode length requested from the executable.
PLOT_TIMESTEPS_PER_EPISODE = 200
SMOOTHING_WINDOW_TIMESTEPS = 10  # Set to 1 to disable smoothing.
REUSE_MUTATION_RATES = True

TEST_SETUP = "mountaincar_continuous_execution_modified_rates"
EXPERIMENT_DIR = REPO / "experiments" / TEST_SETUP
PARAMETERS_FILE = REPO / "configs" / f"{TEST_SETUP}.yaml"
OUTPUT_PDF = REPO / "output/pdf" / f"{TEST_SETUP}_mutation_rates.pdf"
REPLAY_OUTPUT_DIR = REPO / "output/replays" / TEST_SETUP
SMOOTHING_WINDOW_SAMPLES = max(
    1, round(SMOOTHING_WINDOW_TIMESTEPS / RECORD_EVERY_TIMESTEPS)
)
MPI_PROCESSES = 1
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


def selection_rows(seed: int) -> list[dict[str, str]]:
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
    return rows


def final_best_agent(seed: int, generation: int) -> tuple[int, float]:
    rows = selection_rows(seed)
    matching = [
        row for row in rows if int(float(row["generation"])) == generation
    ]
    best = matching[-1] if matching else rows[-1]
    return int(float(best["team_id"])), float(best["best_fitness"])


def checkpoint_complete(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(size - 32, 0))
            return handle.read().rstrip().endswith(b"end")
    except OSError:
        return False


def latest_checkpoint(seed: int) -> tuple[int, int, Path]:
    # Match the normal replay command, which restores the most recent
    # completed training (phase-0) checkpoint.
    phase = 0
    pattern = re.compile(rf"^cp\.(\d+)\.{seed}\.{phase}\.rslt$")
    candidates = []
    for path in (EXPERIMENT_DIR / "checkpoints").glob(f"cp.*.{seed}.{phase}.rslt"):
        match = pattern.match(path.name)
        if match and checkpoint_complete(path):
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint found for seed {seed}")
    generation, path = max(candidates, key=lambda item: item[0])
    return generation, phase, path


def team_in_checkpoint(path: Path, team_id: int) -> bool:
    needle = f"team:{team_id}:".encode()
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(needle):
                return True
    return False


def replay(seed: int, team_id: int, generation: int, phase: int) -> Path:
    if not 1 <= RECORD_EVERY_TIMESTEPS <= EPISODE_TIMESTEPS:
        raise ValueError(
            "RECORD_EVERY_TIMESTEPS must be between 1 and EPISODE_TIMESTEPS"
        )
    REPLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_name = f"mutation_rates.{seed}.{team_id}.csv"
    trace = REPLAY_OUTPUT_DIR / trace_name
    if REUSE_MUTATION_RATES and trace.is_file():
        try:
            load_trace(trace)
        except ValueError as error:
            print(f"Existing trace is invalid; replaying seed {seed}: {error}")
        else:
            print(f"Reusing: {trace}")
            return trace

    executable = REPLAY_EXECUTABLE
    if not executable.is_file():
        raise FileNotFoundError(f"Build the replay executable first: {executable}")
    if b"mutation_rates." not in executable.read_bytes():
        raise FileNotFoundError(
            "Replay executable does not contain per-timestep mutation-rate tracing. "
            "Restore the engine-side logger and rebuild the executable."
        )
    command = [
        "mpirun", "--oversubscribe", "-np", str(MPI_PROCESSES), str(executable),
        f"parameters_file={PARAMETERS_FILE}", f"seed_tpg={seed}", "seed_aux=42",
        "start_from_checkpoint=1", f"checkpoint_in_phase={phase}",
        f"checkpoint_in_t={generation}", "replay=1", "animate=0",
        f"id_to_replay={team_id}", "task_to_replay=0",
        f"replay_max_timesteps={EPISODE_TIMESTEPS}",
    ]
    print("Running:", shlex.join(command))
    stdout_path = REPLAY_OUTPUT_DIR / f"mutation_replay.{seed}.stdout"
    stderr_path = REPLAY_OUTPUT_DIR / f"mutation_replay.{seed}.stderr"
    engine_trace = EXPERIMENT_DIR / "logs/misc" / trace_name
    engine_trace.parent.mkdir(parents=True, exist_ok=True)
    # Do not mistake output from an older replay for a newly generated trace.
    previous_engine_trace = (
        engine_trace.read_bytes() if engine_trace.is_file() else None
    )
    engine_trace.unlink(missing_ok=True)
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            subprocess.run(
                command,
                cwd=EXPERIMENT_DIR,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
    except subprocess.CalledProcessError:
        engine_trace.unlink(missing_ok=True)
        if previous_engine_trace is not None:
            engine_trace.write_bytes(previous_engine_trace)
        raise
    stdout_text = stdout_path.read_text(errors="ignore")
    if f"Evaluation result team:{team_id}" not in stdout_text:
        engine_trace.unlink(missing_ok=True)
        if previous_engine_trace is not None:
            engine_trace.write_bytes(previous_engine_trace)
        raise FileNotFoundError(
            f"Replay finished without evaluating team {team_id} for seed {seed}. "
            "The team is missing from the restored checkpoint."
        )
    if not engine_trace.is_file():
        if previous_engine_trace is not None:
            engine_trace.write_bytes(previous_engine_trace)
        raise FileNotFoundError(
            f"Replay completed without producing {engine_trace}. Verify that the executable "
            "was rebuilt with LogReplaySelfModifyingRates enabled."
        )
    engine_trace.replace(trace)
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
        PLOT_TIMESTEPS_PER_EPISODE + 1,
        RECORD_EVERY_TIMESTEPS,
    )
    timesteps = np.concatenate(
        [episode * PLOT_TIMESTEPS_PER_EPISODE + sample_steps for episode in range(EPISODES)]
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
        if episode < 0:
            raise ValueError(f"{path} contains negative episode {episode}")
        # A replay may contain more episodes than this plot requests. Keep the
        # cache reusable and plot only the configured leading episodes.
        if episode >= EPISODES:
            continue
        if not 1 <= timestep <= EPISODE_TIMESTEPS:
            raise ValueError(f"{path} contains out-of-range timestep {timestep}")
        if timestep > PLOT_TIMESTEPS_PER_EPISODE:
            continue
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


def smooth_series(values: np.ndarray) -> np.ndarray:
    """Return a continuous centered moving average for plotting.

    Replay logging can omit timesteps when no self-modifying program is
    active. Interpolate those missing plotting samples so lines do not break;
    the source CSVs and summary statistics are left unchanged.
    """
    if SMOOTHING_WINDOW_SAMPLES <= 1:
        return values.copy()
    window_size = min(SMOOTHING_WINDOW_SAMPLES, values.size)
    kernel = np.ones(window_size, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values.copy()
    positions = np.arange(values.size)
    filled = np.interp(positions, positions[finite], values[finite])
    left = window_size // 2
    right = window_size - 1 - left
    padded = np.pad(filled, (left, right), mode="edge")
    return np.convolve(padded, kernel / window_size, mode="valid")


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
        # points are smoothed working-register outputs observed during replay.
        y = np.concatenate(
            (
                [first_finite(data[f"start_{rate}"])],
                smooth_series(data[f"output_{rate}"]),
            )
        )
        if std_data is not None:
            y_std = np.concatenate(
                (
                    [first_finite(std_data[f"start_{rate}"])],
                    smooth_series(std_data[f"output_{rate}"]),
                )
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
        PLOT_TIMESTEPS_PER_EPISODE,
        EPISODES * PLOT_TIMESTEPS_PER_EPISODE,
        PLOT_TIMESTEPS_PER_EPISODE,
    ):
        ax.axvline(boundary + 0.5, color="0.75", linewidth=0.8)
    smoothing_timesteps = SMOOTHING_WINDOW_SAMPLES * RECORD_EVERY_TIMESTEPS
    ax.set(
        title=f"{title}\nCentered moving average: {smoothing_timesteps} timesteps",
        xlabel="Timestep across episodes",
        ylabel="Mutation probability",
    )
    ax.set_xlim(0, EPISODES * PLOT_TIMESTEPS_PER_EPISODE)
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

    try:
        seeds = discover_seeds()
    except FileNotFoundError as error:
        print(error)
        return

    traces = []
    labels = []
    for seed in seeds:
        try:
            generation, phase, checkpoint = latest_checkpoint(seed)
            team_id, fitness = final_best_agent(seed, generation)
            if not team_in_checkpoint(checkpoint, team_id):
                raise FileNotFoundError(
                    f"Team {team_id} is not in {checkpoint.name}"
                )
            traces.append(load_trace(replay(seed, team_id, generation, phase)))
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
