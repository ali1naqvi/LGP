#!/usr/bin/env python3
"""Evaluate each saved champion with continuous Gaussian sensor noise.

Use the agents selected by run_rate_sensitivity.py --latest-checkpoints.
Calibrate observation standard deviations and baseline fitness in clean episodes.
Then evaluate each channel/alpha with the same environment episode seeds.
The noise RNG is separate from environment randomness. Results resume per seed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/rate_sensitivity_gaussian_individual_latest'
OUT = SOURCE / 'fitness_rollouts_episode_returns'
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
BINARY = ROOT / 'build/release/experiments/TPGExperimentMPI'


def run(agent, channel=-1, sigma=0.0):
    env = agent['environment']
    prefix = 'mj' if env in ('reacher', 'halfcheetah') else env
    command = [str(BINARY), f"parameters_file={agent['config']}",
               f"seed_tpg={agent['seed']}", 'seed_aux=42',
               'start_from_checkpoint=1', 'checkpoint_in_phase=0',
               f"checkpoint_in_t={agent['generation']}", 'replay=1', 'animate=0',
               f"id_to_replay={agent['team']}", 'task_to_replay=0',
               'seed_with_episode_number=1', 'test_episode_seed_offset=20000',
               f"{prefix}_n_eval_test={agent['episodes']}", 'rate_sensitivity=0',
               f'replay_noise_calibrate={int(channel < 0)}',
               f'replay_noise_channel={channel}', f'replay_noise_sigma={sigma:.17f}',
               'replay_noise_seed=42']
    result = subprocess.run(command, cwd=agent['directory'], capture_output=True, text=True,
                            env={**os.environ, 'TPG': str(ROOT), 'OMP_NUM_THREADS': '1'})
    if result.returncode:
        raise RuntimeError(f'{env} seed {agent["seed"]}: {result.stderr[-3000:]}')
    parsed = {}
    for line in result.stdout.splitlines():
        if line.startswith(('noise_fitness,', 'noise_std,', 'noise_rates,', 'noise_final_rates,')):
            key, *values = line.split(',')
            parsed[key] = [float(v) for v in values]
    rewards = parsed.get('noise_fitness', [])
    if len(rewards) != agent['episodes'] or not all(math.isfinite(v) for v in rewards):
        raise ValueError(f'Missing/nonfinite rewards for {env} seed {agent["seed"]}')
    return parsed


def evaluate(agent):
    path = OUT / f"{agent['environment']}.seed{agent['seed']}.json"
    identity = {k: agent[k] for k in ('environment', 'seed', 'generation', 'team', 'episodes')}
    if path.exists():
        data = json.loads(path.read_text())
        if data['agent'] != identity:
            raise ValueError(f'Changed checkpoint identity: {path}')
    else:
        clean = run(agent)
        data = dict(agent=identity, noise_seed=42, baseline=clean['noise_fitness'],
                    observation_std=clean['noise_std'], conditions=[])
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(path)
    save()
    done = {(c['observation'], c['alpha']) for c in data['conditions']}
    for observation, std in enumerate(data['observation_std']):
        if not math.isfinite(std) or std < 0:
            raise ValueError(f'Invalid calibration in {path}')
        for alpha in ALPHAS:
            if (observation, alpha) in done:
                continue
            result = run(agent, observation, alpha * std)
            data['conditions'].append(dict(observation=observation, alpha=alpha,
                                           fitness=result['noise_fitness']))
            save()
    print(f"Completed {agent['environment']} seed {agent['seed']} ({len(data['conditions'])} conditions)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--environment')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    agents = json.loads((SOURCE / 'agents.json').read_text())
    agents = [a for a in agents if (not args.environment or a['environment'] == args.environment)
              and (args.seed is None or a['seed'] == args.seed)]
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(evaluate, agents):
            pass


if __name__ == '__main__':
    main()
