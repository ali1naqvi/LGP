#!/usr/bin/env python3
"""Plot all-seed structural dependency means beside the best individual."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
CASES = [('mountaincar_continuous', 'Mountain Car', 2000),
         ('pendulum', 'Pendulum', 1000), ('halfcheetah', 'Half-Cheetah', 4000),
         ('reacher', 'Reacher', 3000), ('acrobot', 'Acrobot', 5000)]
OUTPUTS = [('S2', 'Swap'), ('S3', 'Delete'), ('S4', 'Add'), ('S5', 'Mutation')]
MEAN, BEST = '#3379B9', '#D27B26'


def save(fig, prefix):
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(str(prefix) + '.' + suffix, dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'output/structural_dependencies_all_seeds/summary_mean_and_best.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/structural_dependencies_all_seeds/figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    data = {(r['environment'], r['output']): r for r in rows if r['mode'] == 'runtime' and r['observation'] == 'all'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'axes.labelcolor': '#263446', 'text.color': '#263446',
                         'xtick.color': '#526171', 'ytick.color': '#526171'})
    fig, axes = plt.subplots(5, 3, figsize=(14, 13))
    fig.subplots_adjust(left=.17, right=.98, top=.88, bottom=.09, hspace=.65, wspace=.28)
    fig.suptitle('Observation-to-mutation-rate structural dependencies', x=.52, y=.97, fontsize=19, fontweight='bold')
    fig.text(.52, .941, 'execution_modified_rates · One execution · All current observations combined', ha='center', fontsize=11)
    fig.legend(handles=[Patch(facecolor=MEAN, label='Mean across seed champions'),
                        Patch(facecolor=BEST, label='Highest-fitness individual')],
               loc='upper center', bbox_to_anchor=(.52, .929), ncol=2, frameon=False)
    metrics = [('size', 'Size: unique dependent instructions'),
               ('depth', 'Depth: longest instruction chain'),
               ('paths', 'Paths: operand/component routes')]
    for i, (environment, name, generation) in enumerate(CASES):
        first = data[environment, 'S2']
        axes[i, 0].text(-.51, .58, f'{name}\nGeneration {generation}\nn = {first["n_seeds"]}\nBest seed {first["best_seed"]}',
                         transform=axes[i, 0].transAxes, ha='left', va='center', fontsize=11, fontweight='bold')
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            means = [float(data[environment, r][f'mean_{metric}']) for r, _ in OUTPUTS]
            bests = [int(data[environment, r][f'best_{metric}']) for r, _ in OUTPUTS]
            maximum = max(means + bests + [1])
            for x, (mean, best) in enumerate(zip(means, bests)):
                for offset, value, color, text in [(-.19, mean, MEAN, f'{mean:.2f}'),
                                                   (.19, best, BEST, str(best))]:
                    ax.bar(x + offset, value, width=.34, color=color, zorder=3)
                    ax.text(x + offset, value + maximum*.045, text,
                            ha='center', va='bottom', fontsize=8, color=color)
            ax.set_ylim(0, maximum * 1.30)
            ax.set_xlim(-.7, 3.7)
            ax.set_xticks(range(4), [f'{r}\n{role}' for r, role in OUTPUTS], fontsize=9)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(axis='y', color='#E5EAF0', linewidth=.7, zorder=0)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(label, fontsize=11, pad=16)
    fig.text(.17, .034, 'Zeros are included in the means. Acrobot: 18 available seeds; seeds 3 and 18 did not reach generation 5000.\n'
             'S6 has zero observation dependencies in every analyzed agent. Panel scales vary; matrix reductions can have many component paths.', fontsize=9, color='#526171')
    save(fig, args.output_dir / 'dependency_comparison')

    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.065, hspace=.35)
    fig.suptitle('Structural dependency overview', x=.5, y=.98, fontsize=20, fontweight='bold')
    fig.text(.5, .95, 'Node values: size / depth / paths · Branches organize the summary; they are not individual computational paths.',
             ha='center', fontsize=11)
    for ax, (environment, name, generation) in zip(axes, CASES):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        first = data[environment, 'S2']
        ax.text(0, .94, f'{name} · Generation {generation}', fontsize=13, fontweight='bold', va='top')
        ax.text(1, .94, f'{first["n_seeds"]} seeds · Best individual: seed {first["best_seed"]}', ha='right', va='top', fontsize=10, color='#526171')
        ax.text(.5, .77, 'All observations', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.plot([.5, .5], [.69, .58], color='#B3BFCC', lw=1.3)
        centers = [.12, .375, .625, .88]
        ax.plot([centers[0], centers[-1]], [.58, .58], color='#B3BFCC', lw=1.3)
        for x, (register, role) in zip(centers, OUTPUTS):
            ax.plot([x, x], [.58, .45], color='#B3BFCC', lw=1.3)
            ax.add_patch(FancyBboxPatch((x-.112, .03), .224, .42,
                                       boxstyle='round,pad=0.008,rounding_size=0.02',
                                       facecolor='#F2F5F9', edgecolor='none'))
            row = data[environment, register]
            mean = ' / '.join(f'{float(row[f"mean_{m}"]):.2f}' for m in ('size','depth','paths'))
            best = ' / '.join(row[f'best_{m}'] for m in ('size','depth','paths'))
            ax.text(x, .35, f'{register} · {role}', ha='center', fontsize=11, fontweight='bold')
            ax.text(x, .22, f'Mean  {mean}', ha='center', fontsize=10, color=MEAN)
            ax.text(x, .09, f'Best   {best}', ha='center', fontsize=10, color=BEST)
    fig.text(.035, .026, 'S6: 0 / 0 / 0 throughout. This measures structural dependence on current observations; incoming state is reported separately in the data.', fontsize=10, color='#526171')
    save(fig, args.output_dir / 'dependency_tree_overview')
    print(args.output_dir / 'dependency_comparison.png')
    print(args.output_dir / 'dependency_tree_overview.png')


if __name__ == '__main__':
    main()
