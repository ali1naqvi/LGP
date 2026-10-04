"""Replay selected checkpoints on Linux and export test scores for a paper table.

Usage from the repository root:
    python3 scripts/replay_test_table.py prepare
    python3 scripts/replay_test_table.py run --episodes 20
    python3 scripts/replay_test_table.py collect --episodes 20

The selection CSV row at the requested generation supplies the team ID. For
Acrobot, each run's last logged generation and its matching champion are used.
Test episodes use the configs' separate test seed offset. Each job is saved as
JSON so interrupted runs can resume without repeating completed evaluations.
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
ACROBOT_EXPERIMENTS = (
    ("Modified mutation rates", "acrobot_execution_modified_rates"),
    ("Fixed individual mutation rates", "acrobot_fixed_individual"),
    ("Fixed mutation rates", "acrobot_fixed_rates"),
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


def manifest(environment_filter=None):
    rows = []
    for environment, generation, test_parameter, experiments in GROUPS:
        if environment_filter and environment.casefold() != environment_filter.casefold():
            continue
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
    # Evaluate the final available checkpoint for every standalone Acrobot
    # run. A few runs ended before generation 5,000, and their logs/checkpoints
    # are still useful for per-run final-agent testing.
    acrobot_experiments = ACROBOT_EXPERIMENTS if not environment_filter or environment_filter.casefold() == "acrobot" else ()
    for method, experiment in acrobot_experiments:
        directory = ROOT / "experiments" / experiment
        config = config_for(experiment)
        logs = sorted((directory / "logs" / "selection").glob("selection.*.0.csv"))
        if len(logs) != 20:
            raise ValueError(f"Expected 20 selection logs for {experiment}; found {len(logs)}")
        for log in logs:
            seed = int(log.name.split(".")[1])
            with log.open(newline="") as handle:
                rows_for_seed = list(csv.DictReader(handle))
            if not rows_for_seed:
                raise ValueError(f"No selection rows in {log}")
            latest = max(rows_for_seed, key=lambda row: int(row["generation"]))
            generation = int(latest["generation"])
            checkpoint = directory / "checkpoints" / f"cp.{generation}.{seed}.0.rslt"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            rows.append(dict(environment="Acrobot", method=method,
                             experiment=experiment, generation=generation,
                             seed=seed, team_id=int(latest["team_id"]),
                             training_fitness=float(latest["best_fitness"]),
                             checkpoint=str(checkpoint.relative_to(ROOT)),
                             config=str(config.relative_to(ROOT)),
                             test_eval_parameter="acrobot_n_eval_test"))
    if not rows:
        raise ValueError(f"No replay selections for environment {environment_filter!r}")
    return rows


def read_manifest(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("generation", "seed", "team_id"):
            row[field] = int(row[field])
        row["training_fitness"] = float(row["training_fitness"])
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
    observed_counts = []
    for index, line in enumerate(lines[:-1]):
        match = pattern.search(line)
        if match and int(match.group(1)) == team_id:
            count = int(match.group(2))
            observed_counts.append(count)
            values = [float(value) for value in lines[index + 1].strip().split()]
            if count == episodes and len(values) == episodes and all(map(math.isfinite, values)):
                matches.append(values)
    if len(matches) != 1:
        detail = f"; evaluator reported episode counts {observed_counts}" if observed_counts else ""
        hint = " Rebuild TPGExperimentMPI: cmake --build build --target TPGExperimentMPI --parallel 2" if observed_counts and episodes not in observed_counts else ""
        raise ValueError(f"Expected one {episodes}-episode result for team {team_id}; found {len(matches)}{detail}.{hint}")
    return matches[0]


def run_jobs(rows, episodes, experiment_filter=None, job_index=None):
    if experiment_filter:
        rows = [row for row in rows if row["experiment"] == experiment_filter]
    if job_index is not None:
        if job_index < 1 or job_index > len(rows):
            raise ValueError(f"--job-index must be between 1 and {len(rows)} after filtering")
        rows = [rows[job_index - 1]]
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
        launcher = (["srun", "--ntasks=1"] if os.environ.get("SLURM_JOB_ID")
                    else ["mpirun", "-np", "1"])
        command = launcher + [str(BINARY),
                   f"parameters_file={ROOT / row['config']}",
                   f"seed_tpg={row['seed']}", "seed_aux=42",
                   "start_from_checkpoint=1", "checkpoint_in_phase=0",
                   f"checkpoint_in_t={row['generation']}", "replay=1", "animate=0",
                   f"id_to_replay={row['team_id']}", "task_to_replay=0",
                   "seed_with_episode_number=1",
                   f"{row['test_eval_parameter']}={episodes}"]
        if row["environment"] == "Acrobot":
            command.append("replay_max_timesteps=500")
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
        if row["test_eval_parameter"] in ("mountaincar_continuous_n_eval_test", "pendulum_n_eval_test", "acrobot_n_eval_test"):
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
    for method, experiment in ACROBOT_EXPERIMENTS:
        acrobot_rows = [row for row in seed_rows if row["environment"] == "Acrobot" and row["experiment"] == experiment]
        means = [row["test_mean"] for row in acrobot_rows]
        if means:
            median = statistics.median(means)
            std = statistics.stdev(means) if len(means) > 1 else float("nan")
            summary_rows.append(dict(environment="Acrobot", method=method,
                                     experiment=experiment, generation="per-run final",
                                     n_seeds=len(means), episodes_per_seed=episodes,
                                     median_seed_mean=median, std_seed_mean=std,
                                     median_plus_minus_std=f"{median:.3f} ± {std:.3f}"))
    # Acrobot files can be copied back independently of the other tasks' jobs.
    # Collecting an Acrobot-only manifest leaves their existing score CSVs intact.
    for acrobot, prefix in ((False, "test"), (True, "acrobot_test")):
        if not any((row["environment"] == "Acrobot") == acrobot for row in rows):
            continue
        keep = lambda row: (row["environment"] == "Acrobot") == acrobot
        write_csv(RESULT_DIR / f"{prefix}_episode_scores.csv",
                  MANIFEST_FIELDS + ("episode", "test_score", "score_adjustment"),
                  [row for row in episodes_rows if keep(row)])
        write_csv(RESULT_DIR / f"{prefix}_seed_scores.csv",
                  MANIFEST_FIELDS + ("episodes", "test_mean", "test_median", "test_std", "score_adjustment"),
                  [row for row in seed_rows if keep(row)])
        write_csv(RESULT_DIR / f"{prefix}_table_summary.csv",
                  ("environment", "method", "experiment", "generation", "n_seeds", "episodes_per_seed", "median_seed_mean", "std_seed_mean", "median_plus_minus_std"),
                  [row for row in summary_rows if keep(row)])
    print(f"Saved {len(seed_rows)} seed scores and {len(summary_rows)} table rows in {RESULT_DIR}")
    if missing:
        print("Table is partial until every requested replay completes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--experiment", help="Run only one experiment (run action only)")
    parser.add_argument("--environment", help="Prepare, run, or collect only one environment")
    parser.add_argument("--job-index", type=int, help="Run one 1-based row after filtering; for Slurm arrays")
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("--episodes must be at least 2")
    manifest_path = RESULT_DIR / "test_replay_manifest.csv"
    if args.job_index is not None and not manifest_path.exists():
        parser.error("Prepare the manifest before starting array jobs")
    if args.action == "prepare" or not manifest_path.exists():
        rows = manifest(args.environment)
        write_csv(manifest_path, MANIFEST_FIELDS, rows)
        print(f"Prepared {len(rows)} replay selections at {manifest_path}")
    else:
        rows = read_manifest(manifest_path)
        print(f"Loaded {len(rows)} replay selections from {manifest_path}")
    if args.environment:
        rows = [row for row in rows if row["environment"].casefold() == args.environment.casefold()]
    if not rows:
        parser.error("No matching selections in the manifest; run prepare for this environment first")
    if args.action == "run":
        run_jobs(rows, args.episodes, args.experiment, args.job_index)
        if args.job_index is None:
            collect(rows, args.episodes)
    elif args.action == "collect":
        collect(rows, args.episodes)


if __name__ == "__main__":
    main()
