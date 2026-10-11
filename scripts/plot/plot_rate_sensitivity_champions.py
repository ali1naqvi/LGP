#!/usr/bin/env python3
"""Mean sensitivity across seed champions, with equal weight per seed."""
import argparse
import csv
from collections import defaultdict
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

from plot_dependency_matrices import ROOT, CASES
from plot_rate_sensitivity_comparison import area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', default='acrobot', choices=[c[0] for c in CASES])
    args = parser.parse_args()
    out = ROOT / 'output/rate_sensitivity'
    with (out / 'summary_all_seeds.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['environment'] == args.environment
                and float(r['fraction']) in (.01, .1)]
    if not rows:
        raise SystemExit('No matching champion results available.')
    seeds = {int(r['seed']) for r in rows}
    groups = defaultdict(list)
    unique = set()
    for row in rows:
        key = (row['register'], int(row['observation']), float(row['fraction']))
        identity = (int(row['seed']),) + key
        if identity in unique:
            raise ValueError(f'Duplicate champion result: {identity}')
        unique.add(identity)
        groups[key].append(row)
    count = max(int(r['observation']) for r in rows) + 1
    roles = [('S2', 'Swap'), ('S3', 'Delete'), ('S4', 'Add'), ('S5', 'One-point')]
    records = []
    for fraction in (.01, .1):
        for register, role in roles:
            for obs in range(count):
                members = groups[register, obs, fraction]
                if {int(r['seed']) for r in members} != seeds:
                    raise ValueError(f'Incomplete seed coverage: {register}, {obs}, {fraction}')
                records.append(dict(environment=args.environment, register=register, rate=role,
                    observation=obs, fraction=fraction, n_seeds=len(seeds),
                    mean_abs_change=statistics.fmean(float(r['mean_abs_dp']) for r in members),
                    median_seed_mean=statistics.median(float(r['mean_abs_dp']) for r in members),
                    n_affected=sum(float(r['max_abs_dp']) > 0 for r in members),
                    n_linked=sum(r['structural'] != 'none' for r in members)))
    lookup = {(r['register'], r['observation'], r['fraction']): r for r in records}
    name = next(c[1] for c in CASES if c[0] == args.environment)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    fig.subplots_adjust(left=.13, right=.97, top=.72, bottom=.30, wspace=.25)
    for j, (fraction, color) in enumerate(((.01, '#0072B2'), (.1, '#8B5AA5'))):
        ax = axes[j]
        ax.set_xlim(-.65, count-.35)
        ax.set_ylim(4, -.8)
        ax.set_xticks(range(count), [f'$o_{{{k}}}$' for k in range(count)], fontsize=17)
        ax.xaxis.tick_top()
        ax.set_yticks(range(4), [role for _, role in roles] if j == 0 else [], fontsize=15)
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for obs in range(count):
            ax.plot([obs, obs], [-.35, 3.35], color='#E5E5E5', lw=.6, zorder=0)
        for y, (register, _) in enumerate(roles):
            ax.plot([0, count-1], [y, y], color='#E5E5E5', lw=.6, zorder=0)
            for obs in range(count):
                row = lookup[register, obs, fraction]
                if row['mean_abs_change'] > 0:
                    ax.scatter(obs, y, s=area(row['mean_abs_change']), color=color, linewidths=1.5)
                elif row['n_linked']:
                    ax.scatter(obs, y, s=40, marker='x', color=color, linewidths=1.5)
        centre = ax.get_position().x0 + ax.get_position().width/2
        fig.text(centre, .92, rf'$\epsilon = {fraction:g}$', ha='center', fontsize=17)
        fig.text(centre, .82, f'({chr(97+j)}) {name}', ha='center', fontsize=17)
    handles = [Line2D([], [], linestyle='', marker='o', markersize=math.sqrt(area(v)),
                       markeredgewidth=1.5, color='#555555',
                       label='≤0.001' if v == .001 else f'{v:g}') for v in (.001, .01, .1, .2)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105), ncol=4,
               frameon=False, fontsize=14, title='Mean absolute change in mutation rate (0–1 scale)',
               title_fontsize=15)
    fig.text(.5, .035, f'One champion per seed · {len(seeds)} seeds · equal weight per seed',
             ha='center', fontsize=12)
    prefix = out / f'{args.environment}_champions_rate_sensitivity.comparison.mean'
    fig.savefig(str(prefix)+'.pdf', facecolor='white')
    fig.savefig(str(prefix)+'.png', dpi=200, facecolor='white')
    with open(str(prefix)+'.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.close(fig)
    print(f'Created {prefix}.pdf and .png')
    for fraction in (.01, .1):
        selected = [r for r in records if r['fraction'] == fraction]
        largest = max(selected, key=lambda r: r['mean_abs_change'])
        print(f'fraction={fraction}: {sum(r["n_affected"] > 0 for r in selected)}/{len(selected)} active cells; '
              f'largest={largest}')


if __name__ == '__main__':
    main()
