#!/usr/bin/env python3
"""Plot best-agent vector and observation sizes for one test setup."""

# Change this string to select experiments/<TEST_SETUP>.
TEST_SETUP = "gradient_test_perception_4_peaks"

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "experiments" / TEST_SETUP
CONFIG_FILE = REPO / "configs" / f"{TEST_SETUP}.yaml"
OUTPUT_PDF = REPO / "output" / "pdf" / f"{TEST_SETUP}_vector_sizes.pdf"

VECTOR_MEMORY_TYPE = 1  # MemoryEigen::kVectorType_


@dataclass(frozen=True)
class RunResult:
    seed: int
    generation: int
    team_id: int
    fitness: float
    vector_size: int
    observation_size: int


def selection_files_by_seed() -> dict[int, Path]:
    """Return one selection log per seed, preferring the highest process ID."""
    pattern = re.compile(r"^selection\.(\d+)\.(\d+)\.csv$")
    found: dict[int, tuple[int, Path]] = {}
    for path in (EXPERIMENT_DIR / "logs" / "selection").glob("selection.*.*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        seed, process_id = map(int, match.groups())
        if seed not in found or process_id > found[seed][0]:
            found[seed] = (process_id, path)
    return {seed: item[1] for seed, item in sorted(found.items())}


def best_agent(path: Path) -> tuple[int, int, float]:
    """Match replay selection: maximum fitness, then earliest generation."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Selection log is empty: {path}")
    row = min(
        rows,
        key=lambda item: (
            -float(item["best_fitness"]),
            int(float(item["generation"])),
        ),
    )
    return (
        int(float(row["generation"])),
        int(float(row["team_id"])),
        float(row["best_fitness"]),
    )


def latest_completed_checkpoint(seed: int) -> Path:
    pattern = re.compile(rf"^cp\.(\d+)\.{seed}\.(\d+)\.rslt$")
    candidates: list[tuple[int, int, Path]] = []
    checkpoint_dir = EXPERIMENT_DIR / "checkpoints"
    for path in checkpoint_dir.glob(f"cp.*.{seed}.*.rslt"):
        match = pattern.match(path.name)
        if match and path.read_text(encoding="utf-8", errors="ignore").rstrip().endswith("end"):
            generation, phase = map(int, match.groups())
            candidates.append((generation, phase, path))
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint for seed {seed}")
    return max(candidates)[2]


def vector_size_for_team(checkpoint: Path, team_id: int) -> int:
    """Read the sole program's vector dimension, as EvalGradient does."""
    team_programs: list[int] | None = None
    vector_sizes: dict[int, int] = {}
    for line in checkpoint.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split(":")
        if fields[0] == "team" and len(fields) >= 8 and int(fields[1]) == team_id:
            team_programs = [int(value) for value in fields[7:] if value]
        elif (
            fields[0] == "MemoryEigen"
            and len(fields) >= 5
            and int(fields[2]) == VECTOR_MEMORY_TYPE
        ):
            vector_sizes[int(fields[1])] = int(fields[4])

    if team_programs is None:
        raise ValueError(f"Team {team_id} is absent from {checkpoint.name}")
    if len(team_programs) != 1:
        raise ValueError(
            f"Team {team_id} has {len(team_programs)} programs; "
            "gradient evolving observations requires exactly one"
        )
    program_id = team_programs[0]
    if program_id not in vector_sizes:
        raise ValueError(f"Program {program_id} has no vector memory in {checkpoint.name}")
    return vector_sizes[program_id]


def config_flag(name: str) -> int:
    if not CONFIG_FILE.is_file():
        return 0
    match = re.search(
        rf"^\s*{re.escape(name)}\s*:\s*([01])(?:\s|#|$)",
        CONFIG_FILE.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match else 0


def load_results() -> list[RunResult]:
    files = selection_files_by_seed()
    if not files:
        raise FileNotFoundError(
            f"No selection logs found in {EXPERIMENT_DIR / 'logs' / 'selection'}"
        )

    extra_channels = config_flag("gradient_reward_obs") + config_flag(
        "gradient_obs_noise_cue"
    )
    results: list[RunResult] = []
    for seed, selection_file in files.items():
        try:
            generation, team_id, fitness = best_agent(selection_file)
            checkpoint = latest_completed_checkpoint(seed)
            vector_size = vector_size_for_team(checkpoint, team_id)
            results.append(
                RunResult(
                    seed=seed,
                    generation=generation,
                    team_id=team_id,
                    fitness=fitness,
                    vector_size=vector_size,
                    observation_size=vector_size + extra_channels,
                )
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping seed {seed}: {error}")
    if not results:
        raise RuntimeError("No runs had both a usable selection log and checkpoint")
    return results


def plot_summary(pdf: PdfPages, results: list[RunResult]) -> None:
    sizes = np.asarray([result.vector_size for result in results], dtype=float)
    median = float(np.median(sizes))
    standard_deviation = float(np.std(sizes))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    x = np.arange(len(results))
    ax.axhspan(
        median - standard_deviation,
        median + standard_deviation,
        color="#56B4E9",
        alpha=0.2,
        label="Median +/- 1 SD",
    )
    ax.axhline(median, color="#0072B2", linewidth=2.2, label=f"Median = {median:g}")
    ax.scatter(x, sizes, color="#D55E00", s=58, zorder=3, label="Best agent")
    ax.set_xticks(x, [str(result.seed) for result in results])
    ax.set_xlabel("Run (seed)")
    ax.set_ylabel("Vector-memory size")
    ax.set_title(f"{TEST_SETUP}: best-agent vector sizes")
    ax.text(
        0.99,
        0.02,
        f"Runs: {len(results)}   Median: {median:g}   SD: {standard_deviation:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_run_table(pdf: PdfPages, results: list[RunResult]) -> None:
    rows = [
        [
            str(result.seed),
            str(result.team_id),
            str(result.generation),
            f"{result.fitness:.6g}",
            str(result.vector_size),
            str(result.observation_size),
        ]
        for result in results
    ]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    ax.set_title(f"{TEST_SETUP}: best agent from each run", pad=18)
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Seed",
            "Best team",
            "Generation found",
            "Fitness",
            "Vector size",
            "Observation size",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.10, 0.19, 0.19, 0.17, 0.14, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, min(1.45, max(0.68, 22 / len(rows))))
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F7FA")
    ax.text(
        0.5,
        0.045,
        "Observation size = vector size + enabled reward/noise-cue channels.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    results = load_results()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        plot_summary(pdf, results)
        plot_run_table(pdf, results)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
