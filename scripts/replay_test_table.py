"""Replay selected checkpoints on Linux and export test scores for a paper table.

Usage from the repository root:
    python3 scripts/replay_test_table.py prepare
    python3 scripts/replay_test_table.py run --episodes 20
    python3 scripts/replay_test_table.py collect --episodes 20

The selection CSV row at the requested generation supplies the team ID. Test
episodes use the configs' separate test seed offset. Each job is saved as JSON
so interrupted runs can resume without repeating completed evaluations.
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "output" / "test_replay"
BINARY = ROOT / "build" / "release" / "experiments" / "TPGExperimentMPI"
GROUPS = (
    ("Mountain Car Continuous", 2000, "mountaincar_continuous_n_eval_test", (
        ("Modified mutation rates", "mountaincar_continuous_execution_modified_rates"),
        ("Fixed individual mutation rates", "mountaincar_continuous_fixed_individual"),
        ("Fixed mutation rates", "mountaincar_continuous_fixed_rates"),
    )),
    ("Pendulum", 1000, "pendulum_n_eval_test", (
        ("Modified mutation rates", "pendulum_execution_modified_rates_no_limit"),
        ("Fixed individual mutation rates", "pendulum_fixed_individual_all"),
        ("Fixed mutation rates", "pendulum_fixed_rates_no_limit"),
    )),
    ("Half Cheetah", 4000, "mj_n_eval_test", (
        ("Modified mutation rates", "halfcheetah_execution_modified_rates"),
        ("Fixed individual mutation rates", "halfcheetah_fixed_individual"),
        ("Fixed mutation rates", "halfcheetah_fixed_rates"),
    )),
    ("Reacher", 3000, "mj_n_eval_test", (
        ("Modified mutation rates", "reacher_execution_modified_rates"),
        ("Fixed individual mutation rates", "reacher_fixed_individual"),
        ("Fixed mutation rates", "reacher_fixed_rates"),
    )),
)
MANIFEST_FIELDS = ("environment", "method", "experiment", "generation", "seed", "team_id", "training_fitness", "checkpoint", "config", "test_eval_parameter")


def config_for(experiment):
    path = ROOT / "configs" / f"{experiment}.yaml"
    if not path.exists():
        path = ROOT / "configs" / "done" / f"{experiment}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config for {experiment}")
    return path


def selected_agent(log, generation):
    with log.open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if int(row["generation"]) == generation]
    if len(matches) != 1:
        raise ValueError(f"Expected one generation-{generation} row in {log}; found {len(matches)}")
    return int(matches[0]["team_id"]), float(matches[0]["best_fitness"])


def manifest():
    rows = []
    for environment, generation, test_parameter, experiments in GROUPS:
        for method, experiment in experiments:
            directory = ROOT / "experiments" / experiment
            config = config_for(experiment)
            logs = sorted((directory / "logs" / "selection").glob("selection.*.0.csv"))
            if len(logs) != 20:
                raise ValueError(f"Expected 20 selection logs for {experiment}; found {len(logs)}")
            for log in logs:
                seed = int(log.name.split(".")[1])
                checkpoint = directory / "checkpoints" / f"cp.{generation}.{seed}.0.rslt"
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                team_id, fitness = selected_agent(log, generation)
                rows.append(dict(environment=environment, method=method,
                                 experiment=experiment, generation=generation,
                                 seed=seed, team_id=team_id,
                                 training_fitness=fitness,
                                 checkpoint=str(checkpoint.relative_to(ROOT)),
                                 config=str(config.relative_to(ROOT)),
                                 test_eval_parameter=test_parameter))
    return rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def job_path(row, episodes):
    return RESULT_DIR / "jobs" / f"{row['experiment']}.g{row['generation']}.s{row['seed']}.n{episodes}.json"


def parse_outcomes(stdout, team_id, episodes):
    lines = stdout.splitlines()
    pattern = re.compile(r"Evaluation result team:(\d+) n_outcomes (\d+) mean ")
    matches = []
    for index, line in enumerate(lines[:-1]):
        match = pattern.search(line)
        if match and int(match.group(1)) == team_id:
            count = int(match.group(2))
            values = [float(value) for value in lines[index + 1].strip().split()]
            if count == episodes and len(values) == episodes and all(map(math.isfinite, values)):
                matches.append(values)
    if len(matches) != 1:
        raise ValueError(f"Expected one {episodes}-episode result for team {team_id}; found {len(matches)}")
    return matches[0]


def run_jobs(rows, episodes, experiment_filter=None):
    if platform.system() != "Linux":
        raise RuntimeError("Replay requires Linux: the checked-in TPGExperimentMPI is an ELF Linux binary")
    if not BINARY.is_file():
        raise FileNotFoundError(BINARY)
    for row in rows:
        if experiment_filter and row["experiment"] != experiment_filter:
            continue
        destination = job_path(row, episodes)
        if destination.exists():
            continue
        directory = ROOT / "experiments" / row["experiment"]
        command = ["mpirun", "-np", "1", str(BINARY),
                   f"parameters_file={ROOT / row['config']}",
                   f"seed_tpg={row['seed']}", "seed_aux=42",
                   "start_from_checkpoint=1", "checkpoint_in_phase=0",
                   f"checkpoint_in_t={row['generation']}", "replay=1", "animate=0",
                   f"id_to_replay={row['team_id']}", "task_to_replay=0",
                   "seed_with_episode_number=1",
                   f"{row['test_eval_parameter']}={episodes}"]
        print(f"Testing {row['experiment']} seed {row['seed']} at generation {row['generation']}", flush=True)
        completed = subprocess.run(command, cwd=directory, env={**os.environ, "TPG": str(ROOT)},
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_dir = RESULT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_stem = destination.stem
        (log_dir / f"{log_stem}.stdout.txt").write_text(completed.stdout)
        (log_dir / f"{log_stem}.stderr.txt").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Replay failed for {row['experiment']} seed {row['seed']}:\n{completed.stderr[-3000:]}")
        outcomes = parse_outcomes(completed.stdout, row["team_id"], episodes)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps({"selection": row, "episodes": episodes,
                                         "test_scores": outcomes}, indent=2) + "\n")
        temporary.replace(destination)


def collect(rows, episodes):
    episodes_rows = []
    seed_rows = []
    missing = []
    for row in rows:
        path = job_path(row, episodes)
        if not path.exists():
            missing.append(path.name)
            continue
        result = json.loads(path.read_text())
        if result["selection"] != row or result["episodes"] != episodes:
            raise ValueError(f"Result metadata does not match manifest: {path}")
        raw_scores = result["test_scores"]
        if len(raw_scores) != episodes:
            raise ValueError(f"Wrong episode count: {path}")
        # EvalControlViz increments n_prediction before AccumulateStepData,
        # so the replay logger fails to reset its reward accumulator between
        # classic-control episodes. Recover the individual episode returns.
        if row["test_eval_parameter"] in ("mountaincar_continuous_n_eval_test", "pendulum_n_eval_test"):
            scores = [raw_scores[0]] + [current - previous for previous, current in zip(raw_scores, raw_scores[1:])]
            adjustment = "difference_cumulative_replay_rewards"
        else:
            scores = raw_scores
            adjustment = "none"
        for index, score in enumerate(scores):
            episodes_rows.append({**row, "episode": index, "test_score": score,
                                  "score_adjustment": adjustment})
        seed_rows.append({**row, "episodes": episodes,
                          "test_mean": statistics.mean(scores),
                          "test_median": statistics.median(scores),
                          "test_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                          "score_adjustment": adjustment})
    if missing:
        print(f"{len(missing)} of {len(rows)} replays remain; first missing: {missing[0]}")
    write_csv(RESULT_DIR / "test_episode_scores.csv", MANIFEST_FIELDS + ("episode", "test_score", "score_adjustment"), episodes_rows)
    write_csv(RESULT_DIR / "test_seed_scores.csv", MANIFEST_FIELDS + ("episodes", "test_mean", "test_median", "test_std", "score_adjustment"), seed_rows)
    summary_rows = []
    for environment, generation, _, experiments in GROUPS:
        for method, experiment in experiments:
            means = [row["test_mean"] for row in seed_rows if row["experiment"] == experiment]
            if not means:
                continue
            median = statistics.median(means)
            std = statistics.stdev(means) if len(means) > 1 else float("nan")
            summary_rows.append(dict(environment=environment, method=method,
                                     experiment=experiment, generation=generation,
                                     n_seeds=len(means), episodes_per_seed=episodes,
                                     median_seed_mean=median, std_seed_mean=std,
                                     median_plus_minus_std=f"{median:.3f} ± {std:.3f}"))
    write_csv(RESULT_DIR / "test_table_summary.csv",
              ("environment", "method", "experiment", "generation", "n_seeds", "episodes_per_seed", "median_seed_mean", "std_seed_mean", "median_plus_minus_std"),
              summary_rows)
    print(f"Saved {len(seed_rows)} seed scores and {len(summary_rows)} table rows in {RESULT_DIR}")
    if missing:
        print("Table is partial until every requested replay completes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--experiment", help="Run only one experiment (run action only)")
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("--episodes must be at least 2")
    rows = manifest()
    write_csv(RESULT_DIR / "test_replay_manifest.csv", MANIFEST_FIELDS, rows)
    print(f"Prepared {len(rows)} replay selections at {RESULT_DIR / 'test_replay_manifest.csv'}")
    if args.action == "run":
        run_jobs(rows, args.episodes, args.experiment)
        collect(rows, args.episodes)
    elif args.action == "collect":
        collect(rows, args.episodes)


if __name__ == "__main__":
    main()
