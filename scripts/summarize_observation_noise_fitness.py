#!/usr/bin/env python3
"""Combine observed noisy-rollout fitness with replay mutation-rate sensitivity."""
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/rate_sensitivity_gaussian_individual_latest'
NAMES = {'mountaincar_continuous': 'Mountain Car', 'pendulum': 'Pendulum',
         'acrobot': 'Acrobot', 'reacher': 'Reacher', 'halfcheetah': 'Half-Cheetah'}
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]


def main():
    agents = json.loads((OUT / 'agents.json').read_text())
    effects = defaultdict(list)
    with (OUT / 'summary_all_seeds.csv').open() as handle:
        for row in csv.DictReader(handle):
            key = row['environment'], int(row['seed']), int(row['observation']), float(row['alpha'])
            effects[key].append(float(row['mean_abs_dp']))
    assert all(len(values) == 4 for values in effects.values())
    groups = defaultdict(list)
    baselines = defaultdict(list)
    per_seed = []
    for agent in agents:
        env, seed = agent['environment'], agent['seed']
        data = json.loads((OUT / 'fitness_rollouts_episode_returns' / f'{env}.seed{seed}.json').read_text())
        assert data['agent']['team'] == agent['team']
        assert len(data['baseline']) == agent['episodes']
        assert len(data['conditions']) == len(data['observation_std']) * len(ALPHAS)
        baseline = stats.fmean(data['baseline'])
        baselines[env].append(baseline)
        seen = set()
        for condition in data['conditions']:
            obs, alpha = condition['observation'], condition['alpha']
            assert (obs, alpha) not in seen
            seen.add((obs, alpha))
            assert len(condition['fitness']) == agent['episodes']
            fitness = stats.fmean(condition['fitness'])
            effect = stats.fmean(effects[env, seed, obs, alpha])
            row = dict(environment=env, seed=seed, observation=obs, alpha=alpha,
                       clean_fitness=baseline, noisy_fitness=fitness,
                       fitness_change=fitness - baseline, combined_mutation_effect=effect)
            per_seed.append(row)
            groups[env, obs, alpha].append(row)
    summary = []
    for (env, obs, alpha), members in sorted(groups.items()):
        assert len(members) == len(baselines[env])
        summary.append(dict(environment=env, observation=obs, alpha=alpha,
                            n_seeds=len(members),
                            noisy_fitness_mean=stats.fmean(r['noisy_fitness'] for r in members),
                            noisy_fitness_sd=stats.stdev(r['noisy_fitness'] for r in members),
                            clean_fitness_mean=stats.fmean(r['clean_fitness'] for r in members),
                            fitness_change_mean=stats.fmean(r['fitness_change'] for r in members),
                            combined_mutation_effect=stats.fmean(r['combined_mutation_effect'] for r in members)))
    for filename, rows in [('combined_effect_noisy_fitness.csv', summary),
                           ('combined_effect_noisy_fitness_per_seed.csv', per_seed)]:
        with (OUT / filename).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    tex = [r'\documentclass{article}', r'\usepackage[margin=1in]{geometry}',
           r'\usepackage{booktabs,longtable,amsmath}', r'\begin{document}',
           r'\noindent Each row reports fitness evaluated with Gaussian noise applied to one observation at every timestep. Fitness is averaged over 20 episodes per champion and then equally across champions. The reported SD is across champion means. $\Delta F$ is noisy fitness minus the matching clean-evaluation fitness. Positive values indicate improvement.',
           r'\[ E_{i,\alpha}=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{4}\sum_{r=1}^{4}\frac{1}{T_s}\sum_{t=1}^{T_s}|p^{\mathrm{noisy}}_{s,r,t}(i,\alpha)-p^{\mathrm{clean}}_{s,r,t}|. \]',
           r'\noindent $E$ combines swap, delete, add, and one-point mutation into one mean absolute probability change on the 0--1 scale. The mutation effect uses the recorded clean trajectory with noisy working memory carried forward. Fitness uses new environment rollouts, allowing actions and subsequent states to change. Both use the clean observation standard deviation to scale noise. Environment episode seeds are matched across conditions. A separate noise RNG with seed 42 is reset per episode for the fitness evaluations, with draws paired across channels and alphas.',
           r'\begin{table}[ht]\centering',
           r'\caption{Clean-evaluation fitness for the generation-5000 champions.}',
           r'\begin{tabular}{lrr}\toprule Environment & Champions & Clean fitness (mean $\pm$ SD) \\\midrule']
    for env, name in NAMES.items():
        values = baselines[env]
        tex.append(f'{name} & {len(values)} & ${stats.fmean(values):.2f} \\pm {stats.stdev(values):.2f}$ '+r'\\')
    tex.extend([r'\bottomrule\end{tabular}\end{table}'])
    for env, name in NAMES.items():
        tex.extend([r'\begin{longtable}{rrrrr}',
                    r'\caption{'+name+r': fitness under each noise condition and combined mutation effect.}\\',
                    r'\toprule $\alpha$ & Observation & Noisy fitness (mean $\pm$ SD) & $\Delta F$ & $E$ \\',
                    r'\midrule\endfirsthead',
                    r'\toprule $\alpha$ & Observation & Noisy fitness (mean $\pm$ SD) & $\Delta F$ & $E$ \\',
                    r'\midrule\endhead'])
        for alpha in ALPHAS:
            members = sorted((r for r in summary if r['environment'] == env and r['alpha'] == alpha),
                             key=lambda r: r['observation'])
            for row in members:
                effect = row['combined_mutation_effect']
                effect_text = '$<0.001$' if 0 < effect < .001 else f'{effect:.3f}'
                tex.append(f"{alpha:.2f} & $o_{{{row['observation']}}}$ & "
                           f"${row['noisy_fitness_mean']:.2f} \\pm {row['noisy_fitness_sd']:.2f}$ & "
                           f"{row['fitness_change_mean']:+.2f} & {effect_text} "+r'\\')
            if alpha != ALPHAS[-1]:
                tex.append(r'\addlinespace')
        tex.extend([r'\bottomrule\end{longtable}'])
    tex.extend([r'\noindent Effects are rounded to three decimals. Nonzero effects below $0.001$ are shown as $<0.001$. Fitness values are rounded to two decimals. Each noise condition uses one realization per episode.',
                r'\end{document}'])
    (OUT / 'effects_with_fitness.tex').write_text('\n'.join(tex) + '\n')
    print(f'Wrote {len(summary)} table rows from {len(per_seed)} champion-condition evaluations.')


if __name__ == '__main__':
    main()
