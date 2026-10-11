#!/usr/bin/env python3
"""Numerical sensitivity of mutation rates to observations.

Replays agents with rate_sensitivity=1 and summarizes per environment,
observation, rate and perturbation fraction. By default the agents are the best
individuals in best_rate_observation_dependencies (one per environment);
--all-seeds uses every seed's champion at the same generation. Run from the
repository root inside the Linux container:

    python3 scripts/run_rate_sensitivity.py              # best agents
    python3 scripts/run_rate_sensitivity.py --all-seeds  # every seed's champion
    python3 scripts/run_rate_sensitivity.py --summarize-only [--all-seeds]
    python3 scripts/run_rate_sensitivity.py --environment pendulum --force

Each replay evaluates as many episodes as one training evaluation. Step-level
files are summarized and deleted immediately, so a run only keeps the compact
tables. Completed champions are skipped on resume when the fractions and
carry horizon match output/rate_sensitivity/run.json. Best agents:
summary.csv. All seeds: summary_all_seeds.csv and population.csv. See
src/engine/RateSensitivity.h for the one-step and carried-over definitions.
"""
import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/plot'))
from plot_register_observation_links import analyze  # noqa: E402

BINARY = ROOT / 'build/release/experiments/TPGExperimentMPI'
STRUCTURAL = ROOT / 'output/structural_dependencies_all_seeds'
OUT = ROOT / 'output/rate_sensitivity'
VARIANT = 'execution_modified_rates'
PENDULUM_CONFIG = ROOT / 'output/structural_dependencies_requested_generations/pendulum_execution_modified_rates.archived.yaml'
# Environment -> prefix of its *_n_eval_train / *_n_eval_test parameters.
EVAL_PREFIX = {'mountaincar_continuous': 'mountaincar_continuous', 'pendulum': 'pendulum',
               'acrobot': 'acrobot', 'halfcheetah': 'mj', 'reacher': 'mj'}
RATES = (('S2', 'swap'), ('S3', 'delete'), ('S4', 'add'), ('S5', 'mutate'))


def agents(all_seeds):
    source = 'per_seed_observations.csv' if all_seeds else 'best_individuals.csv'
    with (STRUCTURAL / source).open(newline='') as handle:
        rows = {(r['environment'], int(r['generation']), int(r['seed']), int(r['root_team']))
                for r in csv.DictReader(handle) if r['variant'] == VARIANT}
    result = []
    for environment, generation, seed, team in sorted(rows):
        config = PENDULUM_CONFIG if environment == 'pendulum' else ROOT / 'configs' / f'{environment}_{VARIANT}.yaml'
        match = re.search(rf'^\s*{EVAL_PREFIX[environment]}_n_eval_train:\s*(\d+)', config.read_text(), re.M)
        if not match:
            raise ValueError(f'No {EVAL_PREFIX[environment]}_n_eval_train in {config}')
        result.append(dict(environment=environment, generation=generation, seed=seed, team=team,
                           config=config, episodes=int(match.group(1)),
                           directory=ROOT / 'experiments' / f'{environment}_{VARIANT}'))
    return result


def produced(agent):
    """Step-level files the replay writes beside the experiment logs."""
    stem = f"rate_sensitivity.{agent['seed']}.{agent['team']}"
    directory = agent['directory'] / 'logs/misc'
    return {kind: directory / f'{stem}.{kind}.csv' for kind in ('observations', 'one_step', 'carried')}


def settings(args):
    return dict(fractions=args.fractions, carry_horizon=args.carry_horizon, all_seeds=bool(args.all_seeds))


def completed_seeds(args):
    path = OUT / 'run.json'
    table = OUT / ('summary_all_seeds.csv' if args.all_seeds else 'summary.csv')
    if not path.exists() or not table.exists() or json.loads(path.read_text()) != settings(args):
        return set()
    with table.open(newline='') as handle:
        return {(r['environment'], int(r['seed'])) for r in csv.DictReader(handle)}


def replay(agent, args):
    command = ['mpirun', '--allow-run-as-root', '-np', '1', str(BINARY),
               f"parameters_file={agent['config']}", f"seed_tpg={agent['seed']}", 'seed_aux=42',
               'start_from_checkpoint=1', 'checkpoint_in_phase=0',
               f"checkpoint_in_t={agent['generation']}", 'replay=1', 'animate=0',
               f"id_to_replay={agent['team']}", 'task_to_replay=0', 'seed_with_episode_number=1',
               f"{EVAL_PREFIX[agent['environment']]}_n_eval_test={agent['episodes']}",
               'rate_sensitivity=1', f'rate_sensitivity_fractions={args.fractions}',
               f'rate_sensitivity_carry_horizon={args.carry_horizon}']
    log_dir = OUT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{agent['environment']}.seed{agent['seed']}"
    print(f"Replaying {agent['environment']} seed {agent['seed']} team {agent['team']} "
          f"({agent['episodes']} episodes)", flush=True)
    with (log_dir / f'{name}.stdout.txt').open('w') as out, \
         (log_dir / f'{name}.stderr.txt').open('w') as err:
        completed = subprocess.run(command, cwd=agent['directory'], stdout=out, stderr=err,
                                   env={**os.environ, 'TPG': str(ROOT), 'OMP_NUM_THREADS': '1'})
    if completed.returncode:
        raise RuntimeError(f"Replay failed for {name}; see {log_dir}")
    print(f'Finished replay {name}', flush=True)


def structural_labels(agent):
    """(register, observation) -> 'direct' | 'indirect' for this agent's program."""
    path = STRUCTURAL / 'per_seed' / f"{agent['environment']}.g{agent['generation']}.seed{agent['seed']}.json"
    graph, = json.loads(path.read_text())['graphs']
    labels = {}
    for record in analyze(graph):
        for obs in record['direct']:
            labels[record['register'], obs] = 'direct'
        for obs in record['indirect_only']:
            labels[record['register'], obs] = 'indirect'
    return labels


def quantile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def agent_rows(agent, paths):
    with paths['observations'].open(newline='') as handle:
        observations = list(csv.DictReader(handle))
    if any(int(r['baseline_mismatches']) or int(r['boundary_mismatches']) for r in observations):
        raise ValueError(f"{agent['environment']} seed {agent['seed']}: re-execution did not reproduce the replay")
    labels = structural_labels(agent)
    stds = {int(r['observation']): float(r['std']) for r in observations}
    one = defaultdict(list)
    with paths['one_step'].open(newline='') as handle:
        for r in csv.DictReader(handle):
            one[int(r['observation']), float(r['fraction'])].append(
                [abs(float(r[f'dp_{name}'])) for _, name in RATES])
    carried = defaultdict(list)
    with paths['carried'].open(newline='') as handle:
        for r in csv.DictReader(handle):
            carried[int(r['observation']), float(r['fraction'])].append(r)
    rows = []
    for (obs, fraction), values in sorted(one.items()):
        carry_rows = carried[obs, fraction]
        for r_index, (register, name) in enumerate(RATES):
            dp = [v[r_index] for v in values]
            changed = [d for d in dp if d > 0]
            persisted = [r for r in carry_rows if float(r[f'max_abs_dp_after_{name}']) > 0]
            final = [abs(float(r[f'final_dp_{name}'])) for r in carry_rows if r['status'] == 'end']
            rows.append(dict(
                environment=agent['environment'], generation=agent['generation'],
                seed=agent['seed'], team=agent['team'], observation=obs,
                observation_std=stds[obs], register=register, rate=name,
                structural=labels.get((register, obs), 'none'),
                fraction=fraction, n_perturbations=len(dp),
                fraction_changed=len(changed) / len(dp),
                median_abs_dp=statistics.median(dp), mean_abs_dp=statistics.fmean(dp),
                p95_abs_dp=quantile(dp, .95), max_abs_dp=max(dp),
                median_abs_dp_when_changed=statistics.median(changed) if changed else 0.0,
                carried_fraction_persisting=len(persisted) / len(carry_rows),
                carried_max_abs_dp_after=max((float(r[f'max_abs_dp_after_{name}']) for r in carry_rows), default=0.0),
                carried_median_last_lag=statistics.median(int(r['last_rate_change_lag']) for r in persisted) if persisted else 0,
                carried_max_last_lag=max((int(r['last_rate_change_lag']) for r in persisted), default=0),
                carried_fraction_censored=sum(r['status'] == 'horizon' for r in carry_rows) / len(carry_rows),
                final_fraction_changed=sum(f > 0 for f in final) / len(carry_rows),
                final_max_abs_dp=max(final, default=0.0)))
    return rows


def population(rows):
    """Across seed champions; seeds without a link or effect contribute zeros."""
    groups = defaultdict(list)
    for r in rows:
        groups[r['environment'], r['generation'], r['register'], r['rate'], r['observation'], r['fraction']].append(r)
    result = []
    for (environment, generation, register, rate, obs, fraction), members in sorted(groups.items()):
        linked = [r for r in members if r['structural'] != 'none']
        means = [r['mean_abs_dp'] for r in members]
        result.append(dict(
            environment=environment, generation=generation, register=register, rate=rate,
            observation=obs, fraction=fraction, n_seeds=len(members),
            n_direct=sum(r['structural'] == 'direct' for r in members),
            n_indirect=sum(r['structural'] == 'indirect' for r in members),
            n_affected=sum(r['max_abs_dp'] > 0 for r in members),
            mean_of_mean_abs_dp=statistics.fmean(means),
            median_of_mean_abs_dp=statistics.median(means),
            median_mean_abs_dp_linked=statistics.median(r['mean_abs_dp'] for r in linked) if linked else 0.0,
            median_fraction_changed_linked=statistics.median(r['fraction_changed'] for r in linked) if linked else 0.0,
            max_of_max_abs_dp=max(r['max_abs_dp'] for r in members),
            n_carried_persisting=sum(r['carried_fraction_persisting'] > 0 for r in members),
            max_carried_last_lag=max(r['carried_max_last_lag'] for r in members)))
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {path}')


def append_rows(path, rows, lock):
    with lock:
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)


def consume(agent, args, lock):
    """Replay one champion, append its summary, and delete the step-level files."""
    paths = produced(agent)
    table = OUT / ('summary_all_seeds.csv' if args.all_seeds else 'summary.csv')
    try:
        replay(agent, args)
        rows = agent_rows(agent, paths)
        if not rows:
            raise RuntimeError(f"No perturbations recorded for {agent['environment']} seed {agent['seed']}")
        OUT.mkdir(parents=True, exist_ok=True)
        append_rows(table, rows, lock)
        print(f"Summarized {agent['environment']} seed {agent['seed']}", flush=True)
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)


def best_summary():
    with (STRUCTURAL / 'best_individuals.csv').open(newline='') as handle:
        best = {(r['environment'], int(r['seed'])) for r in csv.DictReader(handle) if r['variant'] == VARIANT}
    with (OUT / 'summary_all_seeds.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if (r['environment'], int(r['seed'])) in best]
    write_csv(OUT / 'summary.csv', rows)


def summarize(args):
    table = OUT / ('summary_all_seeds.csv' if args.all_seeds else 'summary.csv')
    if not table.exists():
        print('No results to summarize')
        return
    with table.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('generation', 'seed', 'team', 'observation', 'n_perturbations'):
            row[key] = int(row[key])
        for key, value in row.items():
            if key not in ('environment', 'register', 'rate', 'structural') and not isinstance(value, int):
                row[key] = float(value)
    if args.all_seeds:
        write_csv(OUT / 'population.csv', population(rows))
        best_summary()
    print(f'{table.name}: {len({(r["environment"], r["seed"]) for r in rows})} champions')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all-seeds', action='store_true', help="Every seed's champion instead of the best agents")
    parser.add_argument('--environment', action='append', help='Limit to these environments (repeatable)')
    parser.add_argument('--fractions', default='0.01,0.05,0.1,0.15',
                        help='Perturbation sizes as fractions of each observation std')
    parser.add_argument('--carry-horizon', type=int, default=1000,
                        help='Maximum carried-over steps per perturbation (0 = to the end of the evaluation)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--force', action='store_true', help='Replay even if current results exist')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    selected = [a for a in agents(args.all_seeds) if not args.environment or a['environment'] in args.environment]
    if args.force and not args.summarize_only:
        for name in ('summary.csv', 'summary_all_seeds.csv', 'population.csv', 'run.json'):
            (OUT / name).unlink(missing_ok=True)
    done = completed_seeds(args)
    if not args.summarize_only:
        if platform.system() != 'Linux':
            raise SystemExit('Replay requires the Linux build; run inside the devcontainer')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'run.json').write_text(json.dumps(settings(args)) + '\n')
        pending, missing = [], []
        for agent in selected:
            if (agent['environment'], agent['seed']) in done:
                continue
            checkpoint = agent['directory'] / 'checkpoints' / f"cp.{agent['generation']}.{agent['seed']}.0.rslt"
            if not checkpoint.is_file():
                missing.append(str(checkpoint.relative_to(ROOT)))
                continue
            pending.append(agent)
        if missing:
            print(f'{len(missing)} champions skipped; checkpoint not on disk. First: {missing[0]}', flush=True)
        print(f'{len(pending)} champions to replay, {len(done)} already summarized', flush=True)
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for future in [pool.submit(consume, a, args, lock) for a in pending]:
                future.result()
    summarize(args)


if __name__ == '__main__':
    main()
