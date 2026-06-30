#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tpg_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MODES_LINE = re.compile(
    rf"TPG::MODES\s+t\s+(?P<generation>\d+)"
    rf"\s+pfASize\s+(?P<persistence_filter_size>\d+)"
    rf"\s+pfA_tSize\s+(?P<previous_persistence_filter_size>\d+)"
    rf"\s+change\s+(?P<change>\d+)"
    rf"\s+novelty\s+(?P<novelty>{NUMBER})"
    rf"\s+complexityRTC\s+(?P<complexity_runtime>{NUMBER})"
    rf"\s+complexityTeams\s+(?P<complexity_teams>{NUMBER})"
    rf"\s+complexityPrograms\s+(?P<complexity_programs>{NUMBER})"
    rf"\s+complexityInstructions\s+(?P<complexity_instructions>{NUMBER})"
    rf"\s+ecology\s+(?P<ecology>{NUMBER})"
)

PLOTS = (
    ("persistence_filter_size", "Persistent descendants"),
    ("change", "Change"),
    ("novelty", "Novelty"),
    ("ecology", "Ecology"),
    ("complexity_runtime", "Runtime complexity"),
    ("complexity_teams", "Active teams"),
    ("complexity_programs", "Active programs"),
    ("complexity_instructions", "Effective instructions"),
)

LOG_FILES = [
    Path("experiments/gradient_vector_evolve"),
]
OUTPUT_FILE = Path("modes_metrics.pdf")
FIGURE_TITLE = "MODES diagnostics"
LABELS = None  # Example: ["run 1", "run 2"]
LOG_SUFFIXES = {".std", ".err", ".out", ".log", ".txt"}

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    """Resolve paths from cwd first, then from the repository root."""
    if path.exists() or path.is_absolute():
        return path
    return REPO_ROOT / path


def discover_logs(path: Path):
    """Yield log-like files from a file or experiment directory."""
    path = resolve_path(path)
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        print(f"warning: {path} does not exist")
        return

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix in LOG_SUFFIXES:
            yield candidate


def load_modes(path: Path):
    """Return MODES records found in one stdout/stderr capture."""
    records = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = MODES_LINE.search(line)
            if not match:
                continue
            record = match.groupdict()
            record["generation"] = int(record["generation"])
            for key in record:
                if key != "generation":
                    record[key] = float(record[key])
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Plot TPG::MODES diagnostics from log files or experiment directories."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="Log files or experiment directories. Defaults to LOG_FILES in this script.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_FILE,
        help=f"Output figure path (default: {OUTPUT_FILE}).",
    )
    parser.add_argument("--title", default=FIGURE_TITLE, help="Figure title.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=LABELS,
        help="Optional labels, one per input path.",
    )
    args = parser.parse_args()

    logs = args.logs if args.logs else LOG_FILES
    output = args.output
    title = args.title
    labels = args.labels

    if not logs:
        raise SystemExit("Set at least one path in LOG_FILES at the top of this file.")

    if labels is not None and len(labels) != len(logs):
        raise SystemExit("LABELS/--labels must provide exactly one label per input path.")

    series = []
    for index, path in enumerate(logs):
        discovered = list(discover_logs(path))
        if not discovered:
            print(f"warning: no log files found for {path}")
            continue

        base_label = labels[index] if labels else resolve_path(path).stem
        found_records = False
        for log_file in discovered:
            records = load_modes(log_file)
            if not records:
                continue
            found_records = True
            label = base_label
            if len(discovered) > 1:
                label = f"{base_label}/{log_file.stem}"
            series.append((label, records))

        if not found_records:
            print(f"warning: no TPG::MODES records found in {resolve_path(path)}")

    if not series:
        raise SystemExit("none of the supplied logs contains a TPG::MODES record")

    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True,
                                constrained_layout=True)
    figure.suptitle(title)

    for axis, (field, title) in zip(axes.flat, PLOTS):
        for label, records in series:
            axis.plot([record["generation"] for record in records],
                      [record[field] for record in records],
                      marker="o", markersize=3, label=label)
        axis.set_title(title)
        axis.set_xlabel("Generation")
        axis.grid(alpha=0.25)

    axes.flat[0].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
