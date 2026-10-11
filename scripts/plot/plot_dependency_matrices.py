#!/usr/bin/env python3
"""Source-by-destination matrices, with size/depth/paths in every cell."""
import argparse
import csv
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
CASES = [('mountaincar_continuous', 'Mountain Car', 2000), ('pendulum', 'Pendulum', 1000),
         ('halfcheetah', 'Half-Cheetah', 4000), ('reacher', 'Reacher', 3000), ('acrobot', 'Acrobot', 5000)]
TARGETS = [('S2', 'Swap'), ('S3', 'Delete'), ('S4', 'Add'), ('S5', 'Mutation'), ('S6', 'Decoy')]
OBSERVATIONS = {
    'mountaincar_continuous': ['Position', 'Velocity', 'Noise channel 1', 'Noise channel 2'],
    'pendulum': ['cos(theta)', 'sin(theta)', 'Angular velocity'],
    'acrobot': ['theta1', 'theta2', 'theta1 velocity', 'theta2 velocity'],
    'reacher': ['cos(theta1)', 'cos(theta2)', 'sin(theta1)', 'sin(theta2)',
                'Target position x', 'Target position y', 'theta1 velocity', 'theta2 velocity',
                'Fingertip - target x', 'Fingertip - target y'],
    'halfcheetah': [f'qpos[{i}]' for i in range(1, 9)] + [f'qvel[{i}]' for i in range(9)],
}


def format_value(value, mean):
    return f'{float(value):.2f}' if mean else str(int(value))


def matrix(rows, sources, labels, title, description, footnote):
    count = len(sources)
    fig, axes = plt.subplots(1, 2, figsize=(17, max(4.2, count * .46 + 2.8)))
    fig.subplots_adjust(left=.14, right=.98, top=1 - 1.25 / fig.get_figheight(),
                        bottom=.85 / fig.get_figheight(), wspace=.35)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1 - .15 / fig.get_figheight())
    fig.text(.5, 1 - .53 / fig.get_figheight(), description, ha='center', fontsize=10)
    first = next(iter(rows.values()))
    for ax, stat, color, fill in zip(axes, ('mean', 'best'), ('#28679E', '#A65C1C'), ('#E5F0FA', '#FFF0E1')):
        ax.set_title(f'Mean across {first["n_seeds"]} seeds' if stat == 'mean'
                     else f'Best individual · seed {first["best_seed"]}', fontsize=13, pad=33)
        ax.set_xlim(-.5, 4.5); ax.set_ylim(count - .5, -.5)
        ax.set_xticks(range(5), [f'{register}\n{role}' for register, role in TARGETS], fontsize=10)
        ax.xaxis.tick_top()
        ax.set_yticks(range(count), labels, fontsize=10)
        ax.tick_params(length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, source in enumerate(sources):
            for x, (target, _) in enumerate(TARGETS):
                row = rows[source, target]
                connected = float(row[f'{stat}_paths']) > 0
                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor=fill if connected else '#F6F7F9',
                                       edgecolor='white', linewidth=2))
                text = ' / '.join(format_value(row[f'{stat}_{metric}'], stat == 'mean')
                                  for metric in ('size', 'depth', 'paths'))
                ax.text(x, y, text, va='center', ha='center', fontsize=8.7,
                        color=color if connected else '#85909C')
        ax.set_aspect('auto')
    fig.text(.14, .18 / fig.get_figheight(), footnote, fontsize=9, va='bottom', color='#526171')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'output/structural_dependencies_all_seeds')
    args = parser.parse_args()
    out = args.input_dir / 'matrices'
    out.mkdir(parents=True, exist_ok=True)
    with (args.input_dir / 'observations_mean_and_best.csv').open() as f:
        observations = list(csv.DictReader(f))
    with (args.input_dir / 'registers_mean_and_best.csv').open() as f:
        registers = list(csv.DictReader(f))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#263446'})
    exported = []
    with PdfPages(out / 'all_dependency_matrices.pdf') as pdf:
        for environment, name, generation in CASES:
            obs = [r for r in observations if r['environment'] == environment and r['observation'] != 'all']
            sources = sorted({r['observation'] for r in obs}, key=int)
            labels = [f'o{i} · {OBSERVATIONS[environment][int(i)]}' for i in sources]
            lookup = {(r['observation'], r['output']): r for r in obs}
            specs = [('observations', sources, labels, lookup,
                      'Observation channel → final mutation-rate register',
                      'Cell = size / depth / paths. Shading marks a structural connection. Zeros are included in means.\n'
                      'S6 has no observation paths; it is an inherited-value drift reference. Matrix/vector paths count component routes.')]
            for scope, slug, heading in [('incoming_value', 'incoming_registers', 'Incoming register value → final mutation-rate register'),
                                         ('written_values', 'written_registers', 'Intermediate register values → final mutation-rate register')]:
                selected = [r for r in registers if r['environment'] == environment and r['source_scope'] == scope]
                lookup = {(r['source_register'], r['output']): r for r in selected}
                all_sources = {r['source_register'] for r in selected}
                sources = sorted([s for s in all_sources if any(float(lookup[s, t]['mean_paths']) > 0
                                  or int(lookup[s, t]['best_paths']) > 0 for t, _ in TARGETS)],
                                 key=lambda s: ('SVM'.index(s[0]), int(s[1:])))
                specs.append((slug, sources, sources, lookup, heading,
                    'Cell = size / depth / paths. S = scalar, V = vector, M = matrix. All-zero source rows are omitted.\n' +
                    ('Incoming state is separate from current observations. An unchanged value has 0 / 0 / 1 (identity), including S6 → S6.'
                     if scope == 'incoming_value' else
                     'Source writes are excluded from size/depth. Paths sum over written versions; diagonal entries can include a final-value identity.')))
            for slug, sources, labels, lookup, heading, footnote in specs:
                fig = matrix(lookup, sources, labels, f'{name} · Generation {generation}',
                             heading + ' · execution_modified_rates', footnote)
                prefix = out / f'{environment}.{slug}'
                for suffix in ('png', 'svg', 'pdf'):
                    fig.savefig(str(prefix) + '.' + suffix, dpi=180, facecolor='white')
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)
                # Wide CSV preserves each cell's three separate numeric metrics.
                flat = []
                for source, label in zip(sources, labels):
                    row = {'source': source, 'label': label}
                    for target, _ in TARGETS:
                        for stat in ('mean', 'best'):
                            for metric in ('size', 'depth', 'paths'):
                                key = f'{stat}_{metric}'
                                row[f'{target}_{key}'] = lookup[source, target][key]
                    flat.append(row)
                with Path(str(prefix) + '.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
                exported.append(dict(environment=environment, matrix=slug, source_rows=sources,
                                     targets=[r for r, _ in TARGETS], file=str(prefix)))
    (out / 'manifest.json').write_text(json.dumps(exported, indent=2) + '\n')
    print(f'Wrote {len(exported)} matrices; combined PDF: {out / "all_dependency_matrices.pdf"}')


if __name__ == '__main__':
    main()
