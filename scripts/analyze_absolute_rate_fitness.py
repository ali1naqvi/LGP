#!/usr/bin/env python3
"""Fitness versus actual mutation probabilities in clean champion evaluations."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/rate_sensitivity_gaussian_individual_latest/absolute_rate_fitness'
NAMES = ['mountaincar_continuous', 'pendulum', 'acrobot', 'reacher', 'halfcheetah']


def rank(values):
    # Ignore numerical roundoff below 1e-12 when assigning ties.
    values = np.round(values, 12)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def rho(x, y):
    x, y = rank(x), rank(y)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else float('nan')


def infer(x, y, rng):
    observed = rho(x, y)
    if not np.isfinite(observed):
        return observed, float('nan'), float('nan'), float('nan')
    a, b = rank(x), rank(y)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    exceed = 0
    for _ in range(10000):
        exceed += abs(a @ rng.permutation(b) / denominator) >= abs(observed) - 1e-12
    bootstrap = [rho(x[indices], y[indices]) for indices in rng.integers(len(x), size=(2000, len(x)))]
    low, high = np.nanquantile(bootstrap, [.025, .975])
    return observed, low, high, (exceed + 1) / 10001


def main():
    data = [json.loads(p.read_text()) for p in sorted(OUT.glob('*.json'))]
    assert len(data) == 100
    rng = np.random.default_rng(42)
    results = []
    for env in NAMES:
        agents = [a for a in data if a['agent']['environment'] == env]
        fitness = np.array([np.mean(a['baseline']['fitness']) for a in agents])
        for stage in ['final_rates', 'mean_rates']:
            rates = np.array([a['baseline'][stage] for a in agents])
            for i, metric in enumerate(['combined', 'swap', 'delete', 'add', 'mutate']):
                values = rates.mean(axis=1) if i == 0 else rates[:, i - 1]
                correlation, low, high, p = infer(fitness, values, rng)
                results.append(dict(environment=env,stage=stage,metric=metric,n=len(agents),rho=correlation,
                                    ci_low=low,ci_high=high,p_two_sided=p))
    primary = [r for r in results if r['metric'] == 'combined' and r['stage'] == 'final_rates']
    previous = 0
    for i, r in enumerate(sorted(primary, key=lambda r:r['p_two_sided'])):
        previous = max(previous, min(1, (5-i)*r['p_two_sided']))
        r['p_holm_5_environments'] = previous
    fields = list(results[0])
    if 'p_holm_5_environments' not in fields:
        fields.append('p_holm_5_environments')
    with (OUT/'clean_correlations.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(results)
    for r in primary:
        print(r)


if __name__ == '__main__':
    main()
