#!/usr/bin/env python3
"""Render actual best-individual operand trees, with shared-node references."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_dependency_matrices import CASES, TARGETS, OBSERVATIONS, ROOT


def operand_tree(nodes, root, environment):
    lookup = {node['id']: node for node in nodes}
    seen = set()
    lines = []
    def visit(node_id, prefix='', connector=''):
        node = lookup[node_id]
        kind = node['kind']
        component = node.get('component', [0, 0])
        register = node.get('register', '')
        if register.startswith('V'):
            register += f'[{component[0]}]'
        elif register.startswith('M'):
            register += f'[{component[0]},{component[1]}]'
        if kind == 'instruction':
            label = f"i{node['index']}  {register} <- {node['operation'].removesuffix('_OP')}"
            color = '#27384B'
        elif kind == 'observation':
            index = node['observation']
            label = f'o{index}: {OBSERVATIONS[environment][index]}'
            color = '#126A9C'
        elif kind == 'initial':
            label = f"{register} at entry ({node.get('origin', 'initial value')})"
            color = '#9259A0'
        else:
            label = f"{register or node_id} ({kind})"
            color = '#78818A'
        repeated = node_id in seen and bool(node['parents'])
        if repeated:
            label += '  [shared: expanded above]'
        lines.append((prefix + connector + label, color, node_id))
        if repeated:
            return
        seen.add(node_id)
        child_prefix = prefix + ('    ' if connector == '└── ' else '│   ' if connector else '')
        for i, parent in enumerate(node['parents']):
            visit(parent, child_prefix, '└── ' if i == len(node['parents']) - 1 else '├── ')
    visit(root)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'output/structural_dependencies_all_seeds')
    args = parser.parse_args()
    out = args.input_dir / 'trees'
    out.mkdir(parents=True, exist_ok=True)
    with (args.input_dir / 'summary_mean_and_best.csv').open() as handle:
        summary = {(r['environment'], r['output']): r for r in csv.DictReader(handle)}
    manifest = []
    with PdfPages(out / 'all_best_dependency_trees.pdf') as combined:
        for environment, name, generation in CASES:
            seed = summary[environment, 'S2']['best_seed']
            data = json.loads((args.input_dir / 'per_seed' / f'{environment}.g{generation}.seed{seed}.json').read_text())
            graph, = data['graphs']
            with PdfPages(out / f'{environment}.best_dependency_trees.pdf') as environment_pdf:
                for target, role in TARGETS:
                    row = summary[environment, target]
                    lines = operand_tree(graph['nodes'], graph['output_nodes'][target], environment)
                    width = max(12, max(len(line[0]) for line in lines) * .087 + 1.2)
                    height = max(4.5, 3.0 + len(lines) * .235)
                    fig = plt.figure(figsize=(width, height))
                    fig.text(.035, 1-.30/height, f'{name} · generation {generation} · best seed {seed} · {target} {role}',
                             fontsize=16, weight='bold', va='top')
                    triple = lambda stat: ' / '.join(f"{float(row[f'{stat}_{metric}']):.2f}" if stat == 'mean'
                                                     else row[f'{stat}_{metric}'] for metric in ('size', 'depth', 'paths'))
                    fig.text(.035, 1-.72/height, f'Observation dependencies: size / depth / paths = {triple("best")}    |    Mean ({row["n_seeds"]} seeds): {triple("mean")}',
                             fontsize=11, va='top')
                    source_rows = [r for r in data['metrics'] if r['output'] == target and r['observation'] != 'all' and r['paths']]
                    source_text = ';  '.join(f"o{r['observation']}: {r['size']} / {r['depth']} / {r['paths']}" for r in source_rows)
                    # Wrap long source lists without truncating channels.
                    import textwrap
                    source_text = '\n'.join(textwrap.wrap('Connected observations (size / depth / paths): ' + (source_text or 'none'), width=max(90, int(width*11))))
                    fig.text(.035, 1-1.05/height, source_text, fontsize=9, color='#126A9C', va='top')
                    y_start = 1.65 + .14 * (source_text.count('\n'))
                    for i, (line, color, _) in enumerate(lines):
                        fig.text(.035, 1-(y_start+i*.235)/height, line, fontsize=9.3,
                                 family='DejaVu Sans Mono', color=color, va='top')
                    fig.text(.035, .25/height,
                             'Read from the output down to its operands. i = original zero-based instruction index. Blue = observation; purple = incoming state.\n'
                             'Shared references preserve multiple uses. Size counts unique instructions; depth counts instructions along the longest observation route.\n'
                             'Paths count operand/component routes. Gray constants and incoming-state branches do not contribute to observation metrics.',
                             fontsize=8, color='#5B6672', va='bottom')
                    prefix = out / f'{environment}.{target}.tree'
                    for extension in ('png', 'svg', 'pdf'):
                        fig.savefig(str(prefix)+'.'+extension, dpi=160, facecolor='white')
                    combined.savefig(fig)
                    environment_pdf.savefig(fig)
                    plt.close(fig)
                    Path(str(prefix)+'.txt').write_text('\n'.join(line for line, _, _ in lines)+'\n')
                    manifest.append({'environment': environment, 'generation': generation, 'seed': seed,
                                     'target': target, 'rendered_rows': len(lines), 'file': str(prefix)})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Wrote {len(manifest)} actual dependency trees to {out}')


if __name__ == '__main__':
    main()
