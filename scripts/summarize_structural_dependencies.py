#!/usr/bin/env python3
"""Average each seed's generation champion and report the best-fitness champion.

Example:
  python3 scripts/summarize_structural_dependencies.py \
    --case mountaincar_continuous:2000 --case pendulum:1000 \
    --case halfcheetah:4000 --case reacher:3000 --case acrobot:5000
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

from extract_generation_agent_instructions import ROOT


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                          for k, v in row.items()} for row in rows)


def aggregate(rows, grouping, best_seeds):
    """Equal seed weights, zeros included; best means fitness, not metric maximum."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in grouping)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        seeds = [row['seed'] for row in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Multiple programs/rows per seed for group {key}; explicit agent aggregation required')
        best_seed = best_seeds[members[0]['environment']]
        best = next(row for row in members if row['seed'] == best_seed)
        record = dict(zip(grouping, key))
        record.update(n_seeds=len(members), seeds=sorted(seeds),
                      best_seed=best_seed, best_team=best['root_team'],
                      best_program=best['program'])
        for metric in ('size', 'depth', 'paths'):
            values = [row[metric] for row in members]
            record[f'mean_{metric}'] = statistics.mean(values)
            record[f'sd_{metric}'] = statistics.stdev(values) if len(values) > 1 else 0
            record[f'best_{metric}'] = best[metric]
        record['n_connected_seeds'] = sum(row['paths'] > 0 for row in members)
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, help='ENVIRONMENT:GENERATION; repeat for each environment')
    parser.add_argument('--variant', default='execution_modified_rates')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/structural_dependencies_all_seeds')
    parser.add_argument('--pendulum-config', type=Path, default=ROOT / 'output/structural_dependencies_requested_generations/pendulum_execution_modified_rates.archived.yaml')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    (out / 'per_seed').mkdir(parents=True, exist_ok=True)
    cases = []
    jobs = []
    missing = []
    discovered_counts = {}
    for item in args.case:
        environment, generation = item.rsplit(':', 1)
        generation = int(generation)
        if generation < 0 or environment in [env for env, _ in cases]:
            parser.error('Use one nonnegative generation per environment')
        cases.append((environment, generation))
        logs = sorted((ROOT / 'experiments' / f'{environment}_{args.variant}' / 'logs/selection').glob('selection.*.0.csv'))
        if not logs:
            parser.error(f'No training selection logs for {environment}')
        discovered_counts[environment] = len(logs)
        for log in logs:
            seed = int(log.name.split('.')[1])
            with log.open() as handle:
                logged_generations = [int(row['generation']) for row in csv.DictReader(handle)]
            if generation not in logged_generations:
                missing.append(dict(environment=environment, seed=seed, requested_generation=generation,
                                    last_logged_generation=max(logged_generations, default=None),
                                    reason='No selection row at requested generation'))
                continue
            prefix = out / 'per_seed' / f'{environment}.g{generation}.seed{seed}'
            cmd = [sys.executable, str(ROOT / 'scripts/analyze_observation_rate_dependencies.py'),
                   '--environment', environment, '--variant', args.variant,
                   '--generation', str(generation), '--seed', str(seed),
                   '--exact-checkpoint', '--rate-registers-only', '--output-prefix', str(prefix)]
            if environment == 'pendulum' and not (ROOT / 'configs' / f'pendulum_{args.variant}.yaml').exists():
                cmd += ['--config', str(args.pendulum_config.resolve())]
            jobs.append((environment, generation, seed, prefix, cmd))
    def run(job):
        environment, generation, seed, prefix, cmd = job
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise ValueError(f'{environment} seed {seed}: {completed.stderr.strip()}')
        payload = json.loads(Path(str(prefix) + '.json').read_text())
        if len(payload['graphs']) != 1:
            raise ValueError(f'{environment} seed {seed}: expected one program per champion')
        graph = payload['graphs'][0]
        with open(payload['metadata']['selection_log']) as handle:
            selection = next(row for row in csv.DictReader(handle) if int(row['generation']) == generation)
        if len(graph['engine_effective_indices']) != int(selection['effective_program_instruction_count']):
            raise ValueError(f'{environment} seed {seed}: engine-effective count differs from selection log')
        if payload['metadata']['checkpoint_generation'] != generation:
            raise ValueError('Unexpected non-exact checkpoint')
        seen = set()
        for node in graph['nodes']:
            if not all(parent in seen for parent in node['parents']):
                raise ValueError('Dependency graph is not topologically ordered')
            seen.add(node['id'])
        return payload
    payloads, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                payloads.append(future.result())
            except Exception as error:
                errors.append(str(error))
            done = len(payloads) + len(errors)
            if done % 20 == 0 or done == len(jobs):
                print(f'Analyzed {done}/{len(jobs)} seed champions ({len(errors)} errors)', flush=True)
    if errors:
        (out / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
        raise SystemExit('Incomplete seed coverage; no new aggregate report written. See ' + str(out / 'errors.json'))
    (out / 'errors.json').write_text('[]\n')
    if set(env for env, _ in cases) != set(p['metrics'][0]['environment'] for p in payloads):
        raise SystemExit('At least one environment has no usable seed champions')
    observations = [row for payload in payloads for row in payload['metrics']]
    registers = [row for payload in payloads for row in payload['register_metrics']]
    by_env = defaultdict(list)
    manifests = []
    for payload in payloads:
        champion = payload['metrics'][0]
        by_env[champion['environment']].append(champion)
        manifests.append(dict(environment=champion['environment'], generation=champion['generation'],
            seed=champion['seed'], team=champion['root_team'], fitness=champion['fitness'],
            checkpoint=payload['metadata']['checkpoint'], config=payload['metadata']['config']))
    best_seeds = {env: min(rows, key=lambda row: (-row['fitness'], row['seed']))['seed'] for env, rows in by_env.items()}
    obs_keys = ['environment', 'variant', 'generation', 'mode', 'output', 'role', 'observation']
    reg_keys = ['environment', 'variant', 'generation', 'mode', 'source_scope', 'source_register', 'output']
    obs_aggregate = aggregate(observations, obs_keys, best_seeds)
    reg_aggregate = aggregate(registers, reg_keys, best_seeds)
    expected_counts = {env: len(rows) for env, rows in by_env.items()}
    for row in obs_aggregate + reg_aggregate:
        if row['n_seeds'] != expected_counts[row['environment']]:
            raise ValueError('A metric group has incomplete seed coverage')
    summary = [row for row in obs_aggregate if row['observation'] == 'all']
    write_csv(out / 'summary_mean_and_best.csv', summary)
    write_csv(out / 'observations_mean_and_best.csv', obs_aggregate)
    write_csv(out / 'registers_mean_and_best.csv', reg_aggregate)
    write_csv(out / 'per_seed_observations.csv', sorted(observations, key=lambda r: (r['environment'], r['seed'], r['output'], r['observation'])))
    write_csv(out / 'per_seed_registers.csv', sorted(registers, key=lambda r: (r['environment'], r['seed'], r['source_scope'], r['source_register'], r['output'])))
    best_individuals = [row for row in observations if row['seed'] == best_seeds[row['environment']]]
    write_csv(out / 'best_individuals.csv', best_individuals)
    (out / 'manifest.json').write_text(json.dumps(dict(variant=args.variant, cases=cases,
        averaging='arithmetic mean with equal weight per seed champion at the exact generation; disconnected zeros included',
        sd='sample standard deviation across seed champions',
        best='highest training fitness at that generation, ties use lowest seed; same best individual for every metric',
        best_seeds=best_seeds, n_seeds=expected_counts, discovered_seed_counts=discovered_counts,
        excluded_seeds=missing,
        agents=sorted(manifests, key=lambda row: (row['environment'], row['seed']))), indent=2) + '\n')
    lines = ['# Structural dependencies: all-seed means and best individuals', '',
             'Variant: `' + args.variant + '`. Means use one best individual per available seed at the specified generation, including zero dependencies. Best means highest training fitness across those individuals, not the largest dependency metric.', '',
             'Each triple is **size / depth / paths** from all current observations to the final register. Mean triples are rounded to two decimals here; CSVs include means, sample standard deviations, seed counts and best-individual metrics.', '',
             '| Environment | Generation | Seeds | Row | S2 swap | S3 delete | S4 add | S5 point mutation |',
             '|---|---:|---:|---|---|---|---|---|']
    for env, generation in cases:
        rows = {r['output']: r for r in summary if r['environment'] == env}
        for stat in ('mean', 'best'):
            cells = []
            for output in ('S2', 'S3', 'S4', 'S5'):
                values = [rows[output][f'{stat}_{metric}'] for metric in ('size', 'depth', 'paths')]
                cells.append(' / '.join(f'{v:.2f}' if stat == 'mean' else str(v) for v in values))
            label = 'Mean' if stat == 'mean' else f'Best (seed {best_seeds[env]})'
            lines.append('| ' + ' | '.join([env, str(generation), str(expected_counts[env]), label] + cells) + ' |')
    lines += ['', 'S6 observation dependencies are zero by construction; S6 is the inherited-value drift reference. Incoming state is separate. These metrics cover one execution and describe structural connections, not numerical sensitivity.', '',
              'Vector/matrix paths are component operand routes; repeated observation channels can produce multiple paths. Size counts unique instruction indices. Register source versions are aggregated separately as incoming values or written values.', '',
              'All seed champions were analyzed from exact checkpoints, and their engine-effective instruction counts were checked against their selection logs. All graph edges were checked for topological order.', '',
              'Missing runs (excluded without substituting earlier generations): ' +
              ('; '.join(f"{row['environment']} seed {row['seed']} last logged generation {row['last_logged_generation']}" for row in missing) if missing else 'none') + '.', '',
              'Files: `summary_mean_and_best.csv`, `observations_mean_and_best.csv`, `registers_mean_and_best.csv`, `best_individuals.csv`, raw per-seed CSVs, per-seed graphs and provenance in `manifest.json`.', '',
              'Reproduce:', '', '```bash', 'python3 scripts/summarize_structural_dependencies.py ' + ' '.join(f'--case {env}:{generation}' for env, generation in cases), '```', '']
    (out / 'README.md').write_text('\n'.join(lines))
    archive = Path(str(out) + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name != 'errors.json':
                handle.write(path, arcname=str(Path(out.name) / path.relative_to(out)))
    print('\n'.join(lines[6:18]))
    print(f'Wrote {out}\nArchive: {archive}')


if __name__ == '__main__':
    main()
