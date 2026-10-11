#!/usr/bin/env python3
"""Paper figure comparing best-agent mean sensitivity at 0.01 and 0.1."""
import csv
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_dependency_matrices import ROOT, CASES


def area(value):
    """Linear effect-to-area mapping; keep tiny nonzero effects visible."""
    return max(3.0, 3000.0 * value)


def main():
    out = ROOT / 'output/rate_sensitivity'
    with (out / 'summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    # A partial all-seed aggregation can temporarily replace summary.csv.
    # Recover absent best-agent cells from saved replays without rewriting it.
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_rate_sensitivity import agents, agent_rows
    existing = {(r['environment'], r['register'], int(r['observation']), float(r['fraction']))
                for r in rows}
    for agent in agents(False):
        if all((agent['environment'], 'S2', 0, f) in existing for f in (0.01, 0.1)):
            continue
        stem = f"rate_sensitivity.{agent['seed']}.{agent['team']}"
        paths = {kind: out / 'raw' / agent['environment'] / f'{stem}.{kind}.csv'
                 for kind in ('observations', 'one_step', 'carried')}
        if all(path.exists() for path in paths.values()):
            rows.extend(r for r in agent_rows(agent, paths)
                        if (r['environment'], r['register'], int(r['observation']), float(r['fraction']))
                        not in existing)
    cells = {(r['environment'], r['register'], int(r['observation']), float(r['fraction'])): r
             for r in rows}
    columns = [(0.01, '#0072B2'), (0.1, '#8B5AA5')]
    cases = [CASES[i] for i in (0, 1, 4, 3, 2)]
    roles = [('S2', 'Swap'), ('S3', 'Delete'), ('S4', 'Add'), ('S5', 'One-point')]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.color': '#262626'})
    fig, axes = plt.subplots(5, 2, figsize=(15, 12))
    fig.subplots_adjust(left=.12, right=.985, top=.89, bottom=.13, hspace=.90, wspace=.20)
    for i, (env, name, generation) in enumerate(cases):
        count = max(int(r['observation']) for r in rows if r['environment'] == env) + 1
        for j, (fraction, color) in enumerate(columns):
            ax = axes[i, j]
            ax.set_xlim(-.65, count-.35)
            ax.set_ylim(4.0, -.8)
            ax.set_xticks(range(count), [f'$o_{{{k}}}$' for k in range(count)],
                          fontsize=15 if count > 10 else 17)
            ax.xaxis.tick_top()
            ax.set_yticks(range(4), [role for _, role in roles] if j == 0 else [], fontsize=15)
            ax.tick_params(length=0, pad=5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for k in range(count):
                ax.plot([k, k], [-.35, 3.35], color='#E5E5E5', lw=.6, zorder=0)
            for y, (register, _) in enumerate(roles):
                ax.plot([0, count-1], [y, y], color='#E5E5E5', lw=.6, zorder=0)
                for obs in range(count):
                    row = cells[env, register, obs, fraction]
                    if row['structural'] == 'none':
                        continue
                    value = float(row['mean_abs_dp'])
                    if value == 0:
                        ax.scatter(obs, y, s=40, marker='x', color=color, linewidths=1.5)
                    else:
                        ax.scatter(obs, y, s=area(value),
                                   facecolors=color if row['structural'] == 'direct' else 'white',
                                   edgecolors=color, linewidths=1.5)
            fig.text(ax.get_position().x0 + ax.get_position().width / 2,
                     ax.get_position().y1 + .033,
                     f'({chr(97+2*i+j)}) {"Mountain Car Continuous" if i == 0 else name}',
                     ha='center', fontsize=17, color='black')
            if i == 0:
                fig.text(ax.get_position().x0 + ax.get_position().width/2, .965,
                         rf'$\epsilon = {fraction:g}$', color='black',
                         ha='center', fontsize=17)
    handles = [Line2D([], [], linestyle='', marker='o', markersize=math.sqrt(area(value)),
                      markeredgewidth=1.5, color='#555555',
                      label='≤0.001' if value == .001 else f'{value:g}')
               for value in (.001, .01, .1, .2)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .05), ncol=4,
               frameon=False, fontsize=14, title='Mean absolute change in mutation rate (0–1 scale)',
               title_fontsize=15)
    structure = [Line2D([], [], linestyle='', marker='o', markersize=9, color='#555555', label='Direct'),
                 Line2D([], [], linestyle='', marker='o', markersize=9, markerfacecolor='white',
                        markeredgecolor='#555555', label='Indirect'),
                 Line2D([], [], linestyle='', marker='x', markersize=9, color='#555555', label='No effect')]
    fig.legend(handles=structure, loc='lower center', bbox_to_anchor=(.5, .015), ncol=3,
               frameon=False, fontsize=14)
    prefix = out / 'best_rate_observation_sensitivity.comparison.mean'
    fig.savefig(str(prefix) + '.pdf', facecolor='white')
    fig.savefig(str(prefix) + '.png', dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Created {prefix}.pdf and .png')


if __name__ == '__main__':
    main()
