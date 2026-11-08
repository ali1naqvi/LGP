#!/usr/bin/env python3
"""Replay matched fitness conditions and retain actual mutation probabilities."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import run_observation_noise_fitness as replay

OUT = replay.SOURCE / 'absolute_rate_fitness'


def evaluate(agent, baseline_only):
    name = f"{agent['environment']}.seed{agent['seed']}.json"
    reference = json.loads((replay.OUT / name).read_text())
    path = OUT / name
    data = json.loads(path.read_text()) if path.exists() else dict(agent=reference['agent'], conditions=[])

    def measure(channel, sigma, expected):
        result = replay.run(agent, channel, sigma)
        discrepancy = max(abs(x - y) for x, y in zip(result['noise_fitness'], expected))
        for key in ['noise_rates', 'noise_final_rates']:
            values = result.get(key, [])
            if len(values) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(f'Missing/invalid probabilities: {name}')
        return dict(reference_fitness_max_difference=discrepancy, fitness=result['noise_fitness'], mean_rates=result['noise_rates'],
                    final_rates=result['noise_final_rates'])

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(path)

    if 'baseline' not in data:
        data['baseline'] = measure(-1, 0, reference['baseline'])
        save()
    if not baseline_only:
        completed = {(r['observation'], r['alpha']) for r in data['conditions']}
        for condition in reference['conditions']:
            channel, alpha = condition['observation'], condition['alpha']
            if (channel, alpha) in completed:
                continue
            values = measure(channel, alpha * reference['observation_std'][channel], condition['fitness'])
            data['conditions'].append(dict(observation=channel, alpha=alpha, **values))
            save()
    print(f"Completed {name}: {'baseline' if baseline_only else len(data['conditions'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agents = json.loads((replay.SOURCE / 'agents.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, agent, args.baseline_only) for agent in agents]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
