#!/usr/bin/env python3
"""Clean fitness versus final mutation probability for saved champions."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/rate_sensitivity_gaussian_individual_latest/absolute_rate_fitness'
NAMES = {'mountaincar_continuous': 'Mountain Car', 'pendulum': 'Pendulum',
         'acrobot': 'Acrobot', 'reacher': 'Reacher', 'halfcheetah': 'Half-Cheetah'}
agents = [json.loads(path.read_text()) for path in OUT.glob('*.json')]
with (OUT / 'clean_correlations.csv').open() as handle:
    correlations = {row['environment']: row for row in csv.DictReader(handle)
                    if row['stage'] == 'final_rates' and row['metric'] == 'combined'}
fig, axes = plt.subplots(2, 3, figsize=(10, 6.1))
for ax, (environment, name) in zip(axes.flat, NAMES.items()):
    selected = [a for a in agents if a['agent']['environment'] == environment]
    fitness = [np.mean(a['baseline']['fitness']) for a in selected]
    rates = [np.mean(a['baseline']['final_rates']) for a in selected]
    ax.scatter(fitness, rates, s=28, color='#226789', alpha=.8)
    correlation = float(correlations[environment]['rho'])
    ax.set_title(f'{name} · ρ = {correlation:+.2f} · n = {len(selected)}', fontsize=10)
    ax.set_xlabel('Clean evaluation fitness', fontsize=9)
    ax.set_ylabel('Mean final mutation probability', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.15)
axes.flat[-1].axis('off')
axes.flat[-1].text(.05,.7,'Each point is one generation-5000 champion.\n\nThe scalar averages the four output probabilities.\n\nThe hypothesis predicts a negative association.\n\nAxes vary by environment.',
                    transform=axes.flat[-1].transAxes, fontsize=9, va='top')
fig.tight_layout()
for extension in ['pdf', 'png']:
    fig.savefig(OUT / f'clean_fitness_vs_actual_rates.{extension}', dpi=180)
