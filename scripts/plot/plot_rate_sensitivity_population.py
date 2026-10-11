#!/usr/bin/env python3
"""Population sensitivity of the per-seed champions in best_rate_observation_dependencies.

By default, show perturbation size 0.1 in black at single-column width (3.5 in).
Area represents the mean across seed champions of each champion's mean absolute
rate change, including zero-effect seeds and both direct and indirect dependencies.
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_dependency_matrices import ROOT, CASES
from plot_rate_sensitivity_comparison import area

OUT = ROOT / 'output/rate_sensitivity'
ROLES = [('S2', 'Swap'), ('S3', 'Delete'), ('S4', 'Add'), ('S5', 'One-point')]


def load(fractions):
    cells = {}
    with (OUT / 'population.csv').open() as handle:
        for row in csv.DictReader(handle):
            fraction = float(row['fraction'])
            if fractions and not any(math.isclose(fraction, chosen) for chosen in fractions):
                continue
            cells[row['environment'], row['register'], int(row['observation']), fraction] = row
    return cells


def page(cells, fractions):
    specs = [(CASES[0][0], 'Mountain Car', CASES[0][2]),
             CASES[1], CASES[4], CASES[3], CASES[2]]
    specs = [case for case in specs if any(key[0] == case[0] for key in cells)]
    fig, axes = plt.subplots(len(specs), len(fractions),
                             figsize=(3.5 * len(fractions), 1.22 * len(specs) + 1.1), squeeze=False)
    fig.subplots_adjust(left=.20, right=.985, top=.89, bottom=.15,
                        hspace=.95, wspace=.22)
    # Match the best-agent physical sizes when reduced from 15 to 7.2 inches.
    scale = (7.2 / 15) ** 2
    for i, (env, name, generation) in enumerate(specs):
        count = max(key[2] for key in cells if key[0] == env) + 1
        if env == 'mountaincar_continuous':
            count = 2  # Show physical state inputs; omit the two added noise channels.
        for j, fraction in enumerate(fractions):
            ax = axes[i, j]
            ax.grid(False)
            color = 'black'
            # Keep the two physical Mountain Car inputs centred and compact.
            padding = max(0, 4-count) / 2
            ax.set_xlim(-.65-padding, count-.35+padding)
            ax.set_ylim(4.0, -.8)
            ax.set_xticks(range(count), [f'$o_{{{k}}}$' for k in range(count)],
                          fontsize=6.5 if count > 10 else 10)
            ax.xaxis.tick_top()
            ax.set_yticks(range(4), [role for _, role in ROLES] if j == 0 else [], fontsize=9)
            ax.tick_params(length=0, pad=3)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for obs in range(count):
                ax.plot([obs, obs], [-.35, 3.35], color='#E5E5E5', lw=.35, zorder=0)
            for y, (register, _) in enumerate(ROLES):
                ax.plot([0, count-1], [y, y], color='#E5E5E5', lw=.35, zorder=0)
                for obs in range(count):
                    row = cells.get((env, register, obs, fraction))
                    if row is None:
                        continue
                    linked = int(row['n_direct']) + int(row['n_indirect'])
                    if not linked and not int(row['n_affected']):
                        continue
                    value = float(row['mean_of_mean_abs_dp'])
                    if value == 0:
                        ax.scatter(obs, y, s=10, marker='x', color=color, linewidths=.8)
                    else:
                        size = area(value) * scale
                        ax.scatter(obs, y, s=size, marker='o', facecolors='black',
                                   edgecolors='black', linewidths=.75)
            centre = ax.get_position().x0 + ax.get_position().width/2
            fig.text(centre, ax.get_position().y1 + .036,
                     f'({chr(97+i*len(fractions)+j)}) {name}', ha='center', fontsize=10)
            if i == 0:
                fig.text(centre, .972, rf'$\epsilon = {fraction:g}$', ha='center', va='top', fontsize=11)
    handles = [Line2D([], [], ls='', marker='o', ms=math.sqrt(area(value)*scale),
                      mew=.75, color='black', label='≤0.001' if value == .001 else f'{value:g}')
               for value in (.001, .01, .1, .2)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.55, .044), ncol=4,
               frameon=False, fontsize=9,
               title='Mean absolute change in mutation rate\n(0–1 scale)', title_fontsize=9)
    no_effect = Line2D([], [], ls='', marker='x', ms=5, color='black', label='No effect')
    fig.legend(handles=[no_effect], loc='lower center', bbox_to_anchor=(.55, .006),
               frameon=False, fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fraction', type=float, action='append',
                        help='Column perturbation sizes (default: 0.1)')
    args = parser.parse_args()
    cells = load(args.fraction or [.1])
    fractions = sorted({key[3] for key in cells})
    if not fractions:
        raise SystemExit(f'No rows in {OUT / "population.csv"}')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'text.color': '#262626', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    prefix = OUT / 'population_rate_observation_sensitivity'
    fig = page(cells, fractions)
    fig.savefig(str(prefix) + '.pdf', facecolor='white')
    fig.savefig(str(prefix) + '.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'Created {prefix}.pdf and .png ({len(fractions)} columns)')


if __name__ == '__main__':
    main()
