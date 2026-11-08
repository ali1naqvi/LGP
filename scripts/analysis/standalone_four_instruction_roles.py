"""Reconstruct disjoint structural instruction roles from exact endpoint checkpoints.

Run from any directory: python3 scripts/analysis/standalone_four_instruction_roles.py
No earlier-generation roles are inferred from endpoint checkpoints.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from extract_best_agent_instructions import load_instruction_definitions
from extract_generation_agent_instructions import INSTRUCTION_RE, load_parameters, output_registers
from structural_dependency_graph import engine_effective_indices

OUT = ROOT / 'output/standalone_four_instruction_roles'
TASKS = [('mountaincar_continuous', 'Mountain Car'), ('pendulum', 'Pendulum'),
         ('acrobot', 'Acrobot'), ('reacher', 'Reacher'), ('halfcheetah', 'Half-Cheetah')]
KEYS = ['structural_introns', 'action_only', 'mutation_only', 'shared']
LABELS = ['Structural introns', 'Action-only', 'Mutation-only', 'Both action and mutation']
COLORS = ['#A8BFA3', '#98B5C9', '#D6A77E', '#A491B5']
GENERATION = 5000


def read_checkpoint(path):
    # Memory constants are irrelevant to this structural dependency analysis.
    teams, programs = {}, {}
    with path.open() as handle:
        for line in handle:
            fields = line.strip().split(':')
            if fields[0] == 'team':
                teams[int(fields[1])] = [int(v) for v in fields[7:] if v]
            elif fields[0] == 'RegisterMachine':
                programs[int(fields[1])] = dict(action=int(fields[3]), memory_slots=int(fields[10]),
                                               instructions=fields[12:])
    return teams, programs


def classify(instructions, slots, definitions, params, sm):
    action_roots = output_registers(params, slots, False)
    mutation_roots = {('S', i) for i in range(2, min(6, slots))} if sm else set()
    def dependencies(roots):
        return set(engine_effective_indices(instructions, slots, definitions, roots, sm))
    action, mutation = dependencies(action_roots), dependencies(mutation_roots)
    full = dependencies(action_roots | mutation_roots)
    if full != action | mutation:
        raise ValueError('Combined dependency analysis does not equal the union')
    parts = [set(range(len(instructions))) - full, action - mutation, mutation - action, action & mutation]
    if sum(map(len, parts)) != len(instructions):
        raise ValueError('Role partition does not cover instructions exactly once')
    return np.array(list(map(len, parts)))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    definitions = load_instruction_definitions()
    records, coverage, summaries = [], [], []
    for env, title in TASKS:
        name = env + '_execution_modified_rates' + ('_vector' if env in {'reacher', 'halfcheetah'} else '')
        directory = ROOT / 'experiments' / name
        config = next(p for p in [ROOT/'configs'/f'{name}.yaml', ROOT/'configs/done'/f'{name}.yaml'] if p.exists())
        params = load_parameters(config)
        group = []
        for log in sorted((directory / 'logs/selection').glob('selection.*.0.csv')):
            seed = int(log.name.split('.')[1])
            with log.open() as handle:
                selected = [r for r in csv.DictReader(handle) if int(r['generation']) == GENERATION]
            checkpoint = directory / 'checkpoints' / f'cp.{GENERATION}.{seed}.0.rslt'
            reason = 'missing endpoint selection' if not selected else 'missing checkpoint' if not checkpoint.exists() else ''
            if len(selected) > 1:
                raise ValueError(f'Duplicate endpoint selection: {log}')
            audit = dict(environment=env, seed=seed, generation=GENERATION, included=not reason, reason=reason)
            coverage.append(audit)
            if reason:
                continue
            row = selected[0]
            teams, programs = read_checkpoint(checkpoint)
            team = int(row['team_id'])
            if team not in teams:
                raise ValueError(f'Selected team {team} missing from {checkpoint}')
            counts = np.zeros(4, dtype=int)
            # Match team::policyInstructions: visit teams once, count each membership.
            pending, visited = [team], set()
            while pending:
                tid = pending.pop()
                if tid in visited:
                    continue
                visited.add(tid)
                if tid not in teams:
                    raise ValueError(f'Missing reachable team {tid}: {checkpoint}')
                for pid in teams[tid]:
                    program = programs[pid]
                    raw = program['instructions']
                    instructions = [v for v in raw if INSTRUCTION_RE.fullmatch(v)]
                    flag = next((v for v in raw if v in ('SM0', 'SM1')), None)
                    sm = flag == 'SM1' if flag else bool(params.get('self_modifying', 0))
                    counts += classify(instructions, program['memory_slots'], definitions, params, sm)
                    if program['action'] >= 0:
                        pending.append(program['action'])
            total = int(counts.sum())
            actual = (total, total-int(counts[0]), int(counts[2]))
            expected = tuple(int(row[k]) for k in ['program_instruction_count', 'effective_program_instruction_count',
                                                   'self_modification_only_instruction_count'])
            if actual != expected:
                raise ValueError(f'Log mismatch {env} seed {seed}: reconstructed {actual}, logged {expected}')
            record = dict(environment=env, seed=seed, generation=GENERATION, team_id=team,
                          total=total, log_counts_match=True, checkpoint=str(checkpoint.relative_to(ROOT)))
            record.update({key+'_count': int(value) for key, value in zip(KEYS, counts)})
            record.update({key+'_fraction': float(value/total) for key, value in zip(KEYS, counts)})
            records.append(record)
            group.append(record)
        if not group:
            raise ValueError(f'No validated endpoints for {env}')
        summary = dict(environment=env, generation=GENERATION, n_seeds=len(group))
        summary.update({key+'_fraction': float(np.mean([r[key+'_fraction'] for r in group])) for key in KEYS})
        summaries.append(summary)
        print(title, 'validated seeds:', len(group), {k: round(summary[k+'_fraction'], 3) for k in KEYS}, flush=True)
    write_csv('roles_by_seed.csv', records)
    write_csv('endpoint_summary.csv', summaries)
    write_csv('seed_coverage.csv', coverage)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#BBBBBB'})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=.18, right=.98, bottom=.23, top=.96)
    left = np.zeros(len(TASKS))
    for key, label, color in zip(KEYS, LABELS, COLORS):
        values = np.array([r[key+'_fraction'] for r in summaries])
        ax.barh(range(len(TASKS)), values, left=left, color=color, edgecolor='white', linewidth=1,
                label=label, height=.58)
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= .02:
                ax.text(start+value/2, y, f'{value:.0%}', ha='center', va='center', fontsize=9)
        left += values
    ax.set(yticks=range(len(TASKS)), yticklabels=[f'{title} (n={s["n_seeds"]})' for (_, title), s in zip(TASKS, summaries)],
           xlim=(0, 1), xlabel='Instructions (%)', xticks=np.linspace(0, 1, 6))
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis='y', length=0)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.025),
               ncol=2, frameon=False, fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'four_instruction_roles_generation_5000.{ext}', dpi=220, facecolor='white')
    plt.close(fig)
    (OUT/'README.md').write_text('''# Four disjoint structural instruction roles

Exact generation-5000 training checkpoints, paired with that seed's training-selected team in the selection log. Fractions are computed per selected policy and averaged equally across seeds. This is an endpoint comparison, not a reconstructed training trajectory.

Action-only = dependencies of action/bid outputs minus dependencies of mutation-rate outputs. Mutation-only = mutation dependencies minus action dependencies. Both = intersection. Structural introns = instructions outside their union. The four categories sum to 100%; displayed percentages are rounded independently. Shared instructions are counted once.

Uses the repository engine_effective_indices implementation of MarkIntrons (conservative structural dependencies, including fixed-point expansion and identical S6 decoy handling in both analyses). These categories describe structural dependencies, not measured causal effects or experimental neutrality. Each reconstructed policy is checked against all three logged counts: total, effective, mutation-only. Any mismatch aborts the analysis. See roles_by_seed.csv and seed_coverage.csv for provenance and exclusions.

Only endpoint checkpoints are available locally. Four-way trajectories cannot be inferred from the existing three-count selection logs. Original three-category figures remain separate.
''')


if __name__ == '__main__':
    main()
