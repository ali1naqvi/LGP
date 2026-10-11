#!/usr/bin/env python3
"""best_rate_observation_dependencies with numerical effect sizes as marker area.

Marker style is the structural link (filled: direct, open: indirect only).
Marker area is |change in mutation probability| when that observation is
shifted by --fraction of its replay std (scripts/run_rate_sensitivity.py).
Page 1 uses the mean over replayed steps, page 2 the maximum.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from plot_dependency_matrices import ROOT, CASES, OBSERVATIONS
from plot_register_observation_links import analyze

LOG_MIN, LOG_MAX = -6, 0
SIZE_MIN, SIZE_MAX = 4, 330


def area(value):
    scaled = (math.log10(max(value, 10 ** LOG_MIN)) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * min(max(scaled, 0.0), 1.0)


def page(effects, metric, fraction, label, base, summary):
    fig = plt.figure(figsize=(14, 7.5), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.065, right=.98, top=.94, bottom=.17,
                            hspace=.60, wspace=.16, height_ratios=[1, 1, 1])
    specs = [(CASES[0][0], 'Mountain Car Continuous', CASES[0][2]),
             CASES[1], CASES[4], CASES[3], CASES[2]]
    roles = [('S2', 'Swap'), ('S3', 'Delete'), ('S4', 'Add'), ('S5', 'One-point')]
    for panel, (env, name, generation) in enumerate(specs):
        ax = fig.add_subplot(grid[panel // 2, panel % 2] if panel < 4 else grid[2, :])
        seed = summary[env, 'S2']['best_seed']
        data = json.loads((base / 'per_seed' / f'{env}.g{generation}.seed{seed}.json').read_text())
        graph, = data['graphs']
        records = {r['register']: r for r in analyze(graph)}
        count = graph['observation_dimension']
        ax.set_xlim(-2.0, count + .7)
        ax.set_ylim(3.5, -1.15)
        ax.axis('off')
        ax.set_title(f'({chr(97 + panel)}) Best agent of {name}', fontsize=10, loc='left', pad=8)
        for obs in range(count):
            ax.text(obs, -.8, f'$o_{{{obs}}}$', ha='center', fontsize=9)
            ax.plot([obs, obs], [-.45, 3.4], color='#E3E3E3', lw=.6, zorder=0)
        for y, (register, operation) in enumerate(roles):
            ax.text(-.6, y, operation, ha='right', va='center', fontsize=9)
            ax.plot([0, count - .6], [y, y], color='#E3E3E3', lw=.6, zorder=0)
            row = records[register]
            for obs in row['direct'] + row['indirect_only']:
                value = effects[env, register, obs][metric]
                if value == 0:
                    ax.scatter(obs, y, s=22, marker='x', c='#B22222', lw=1.0, zorder=3)
                elif obs in row['direct']:
                    ax.scatter(obs, y, s=area(value), c='#262626', zorder=3)
                else:
                    ax.scatter(obs, y, s=area(value), facecolors='white',
                               edgecolors='#262626', lw=1.1, zorder=3)
    handles = [Line2D([], [], ls='', marker='o', ms=math.sqrt(area(10 ** e)), color='#262626',
                      label=f'$10^{{{e}}}$') for e in (-6, -4, -2, 0)]
    handles += [Line2D([], [], ls='', marker='o', ms=7, mfc='white', mec='#262626', label='Indirect only'),
                Line2D([], [], ls='', marker='x', ms=6, color='#B22222', mew=1.0, label='Structural link, no effect')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.05, .065), ncol=6, frameon=False,
               fontsize=9, title=f'{label} |Δ mutation probability|  (filled: direct input)',
               title_fontsize=9, alignment='left')
    fig.text(.055, .035, f'Each observation shifted by ±{fraction:g} × its replay standard deviation, one step at a time; '
             f'area is log-scaled (≤$10^{{{LOG_MIN}}}$ shown at minimum size). Blank cells: no structural dependency.',
             fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fraction', type=float, action='append',
                        help='Perturbation fraction to draw (repeatable; default: every fraction in the summary)')
    args = parser.parse_args()
    base = ROOT / 'output/structural_dependencies_all_seeds'
    out = ROOT / 'output/rate_sensitivity'
    with (base / 'summary_mean_and_best.csv').open() as handle:
        summary = {(r['environment'], r['output']): r for r in csv.DictReader(handle)}
    effects = {}
    with (out / 'summary.csv').open() as handle:
        for r in csv.DictReader(handle):
            fraction = float(r['fraction'])
            if args.fraction and not any(math.isclose(fraction, chosen) for chosen in args.fraction):
                continue
            effects[r['environment'], r['register'], int(r['observation']), fraction] = dict(
                mean=float(r['mean_abs_dp']), max=float(r['max_abs_dp']))
    fractions = sorted({key[3] for key in effects})
    if not fractions:
        raise SystemExit(f'No matching rows in {out / "summary.csv"}')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'text.color': '#262626', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    prefix = out / 'best_rate_observation_sensitivity'
    with PdfPages(str(prefix) + '.pdf') as pdf:
        for fraction in fractions:
            selected = {key[:3]: value for key, value in effects.items() if math.isclose(key[3], fraction)}
            for metric, label in (('mean', 'Mean'), ('max', 'Maximum')):
                fig = page(selected, metric, fraction, label, base, summary)
                pdf.savefig(fig, facecolor='white')
                fig.savefig(f'{prefix}.{metric}.{fraction:g}.png', dpi=200, facecolor='white')
                plt.close(fig)
    print(f'Created {prefix}.pdf ({len(fractions)} perturbation sizes)')


if __name__ == '__main__':
    main()
