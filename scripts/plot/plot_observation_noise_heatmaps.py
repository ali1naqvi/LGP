#!/usr/bin/env python3
"""Paired heatmaps of noisy fitness changes and replay mutation-rate effects."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/rate_sensitivity_gaussian_individual_latest'
NAMES = [('mountaincar_continuous', 'Mountain Car'), ('pendulum', 'Pendulum'),
         ('acrobot', 'Acrobot'), ('reacher', 'Reacher'), ('halfcheetah', 'Half-Cheetah')]
ALPHAS = [.05, .1, .15, .2, .25, .5]
rows = list(csv.DictReader((OUT / 'combined_effect_noisy_fitness.csv').open()))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.titlesize': 10})
fig, axes = plt.subplots(5, 2, figsize=(10, 9.2))
fig.subplots_adjust(left=.075, right=.94, bottom=.115, top=.91, hspace=.73, wspace=.4)
maximum_effect = max(float(r['combined_mutation_effect']) for r in rows)
effect_norm = Normalize(0, maximum_effect)
for index, (environment, name) in enumerate(NAMES):
    selected = [r for r in rows if r['environment'] == environment]
    observations = sorted({int(r['observation']) for r in selected})
    lookup = {(float(r['alpha']), int(r['observation'])): r for r in selected}
    fitness = np.array([[float(lookup[a, o]['fitness_change_mean']) for o in observations] for a in ALPHAS])
    effect = np.array([[float(lookup[a, o]['combined_mutation_effect']) for o in observations] for a in ALPHAS])
    limit = float(np.max(np.abs(fitness)))
    fitness_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for column, values in enumerate([fitness, effect]):
        ax = axes[index, column]
        artist = ax.imshow(values, aspect='auto', interpolation='nearest',
                           cmap='RdBu' if column == 0 else 'viridis',
                           norm=fitness_norm if column == 0 else effect_norm)
        ax.set_xticks(range(len(observations)), [f'$o_{{{o}}}$' for o in observations], fontsize=8)
        ax.set_yticks(range(6), [f'{a:g}' for a in ALPHAS], fontsize=8)
        ax.set_ylabel(r'$\alpha$', rotation=0, labelpad=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if column == 0:
            ax.set_title(name, loc='left', pad=5, fontweight='bold')
            bar_ax = ax.inset_axes([1.025, 0, .035, 1])
            bar = fig.colorbar(artist, cax=bar_ax)
            bar.set_ticks([-limit, 0, limit])
            bar.ax.set_yticklabels([f'{-limit:.1f}', '0', f'{limit:.1f}'])
            bar.ax.tick_params(labelsize=7, length=2)
            bar.outline.set_visible(False)
        else:
            effect_artist = artist
fig.text(.245, .967, 'Fitness change from clean baseline', ha='center', fontsize=12, weight='bold')
fig.text(.245, .944, 'Red = lower reward · Blue = higher reward', ha='center', fontsize=9, color='#555555')
fig.text(.755, .967, 'Combined mutation-rate effect', ha='center', fontsize=12, weight='bold')
fig.text(.755, .944, 'Mean absolute change across four rates', ha='center', fontsize=9, color='#555555')
bar_ax = fig.add_axes([.61, .055, .30, .012])
bar = fig.colorbar(effect_artist, cax=bar_ax, orientation='horizontal')
bar.locator = MaxNLocator(4)
bar.update_ticks()
bar.outline.set_visible(False)
bar.ax.tick_params(labelsize=8)
bar.set_label('Mutation probability change (0–1) · shared scale', fontsize=9)
fig.text(.075, .052, 'Observations are perturbed individually.\nFitness colour scales differ by environment.', fontsize=9, color='#555555')
for extension in ['pdf', 'png']:
    fig.savefig(OUT / f'fitness_mutation_heatmaps.{extension}', dpi=200)
print('Saved paired heatmaps.')
