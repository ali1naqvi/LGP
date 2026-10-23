"""Seed-level comparison of current Mountain Car transfer training runs.

Uses NumPy exact paired sign-flip tests; no SciPy dependency is needed.
"""
import csv
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/mountaincar_transfer_statistics'
VARIANTS = ('execution_modified_rates', 'fixed_individual', 'fixed_rates')
LABELS = dict(zip(VARIANTS, ('Runtime-modified', 'Gene-encoded', 'Static')))
THRESHOLDS = {'pendulum': -200., 'acrobot': -100.}
METRICS = ('mean_transfer_score', 'final_score')
RNG = np.random.default_rng(20261017)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_flip(differences):
    """Two-sided paired randomization of the sum, enumerating every sign pattern."""
    sums = np.array([0.])
    for d in differences:
        sums = np.concatenate((sums + d, sums - d))
    observed = abs(float(np.sum(differences)))
    tolerance = 1e-10 * max(1., observed)
    return float(np.mean(np.abs(sums) >= observed - tolerance))


def bootstrap_mean_ci(values):
    indices = RNG.integers(0, len(values), size=(50000, len(values)))
    return np.quantile(np.mean(values[indices], axis=1), [.025, .975])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for task, threshold in THRESHOLDS.items():
        for variant in VARIANTS:
            for seed in range(1, 21):
                path = ROOT / f'experiments/mountaincar_continuous_to_{task}_{variant}/seed_{seed}/{task}/logs/selection/selection.{seed}.0.csv'
                if not path.exists():
                    missing.append(f'{task}, {LABELS[variant]}, seed {seed}')
                    continue
                with path.open() as f:
                    rows = [r for r in csv.DictReader(f) if 2001 <= int(r['generation']) <= 7000]
                generations = np.array([int(r['generation']) for r in rows])
                if not np.array_equal(generations, np.arange(2001, 7001)):
                    raise ValueError(f'Incomplete or duplicated generations: {path}')
                scores = np.array([float(r['best_fitness']) for r in rows])
                if not np.isfinite(scores).all():
                    raise ValueError(f'Nonfinite fitness: {path}')
                hits = np.flatnonzero(scores >= threshold)
                records.append(dict(task=task, variant=variant, seed=seed,
                                    mean_transfer_score=float(scores.mean()),
                                    final_score=float(scores[-1]),
                                    reached_threshold=int(bool(len(hits))),
                                    first_threshold_step=int(hits[0] + 1) if len(hits) else '',
                                    nonzero_validation_rows=sum(float(r['validation_fitness']) != 0 for r in rows)))
    common = {task: sorted(set.intersection(*[
        {r['seed'] for r in records if r['task'] == task and r['variant'] == v}
        for v in VARIANTS])) for task in THRESHOLDS}
    for r in records:
        r['in_complete_case_analysis'] = int(r['seed'] in common[r['task']])
    write_csv('per_seed_metrics.csv', records)
    tests, summaries = [], []
    for task in THRESHOLDS:
        for variant in VARIANTS:
            selected = [r for r in records if r['task'] == task and r['variant'] == variant and r['in_complete_case_analysis']]
            for metric in METRICS:
                values = np.array([r[metric] for r in selected])
                lo, hi = bootstrap_mean_ci(values)
                summaries.append(dict(task=task, variant=variant, metric=metric, n=len(values),
                                      mean=float(values.mean()), median=float(np.median(values)),
                                      sd=float(values.std(ddof=1)), mean_ci_low=float(lo), mean_ci_high=float(hi)))
        for metric in METRICS:
            for a, b in combinations(VARIANTS, 2):
                lookup = {(r['variant'], r['seed']): r[metric] for r in records if r['task'] == task}
                d = np.array([lookup[a, seed] - lookup[b, seed] for seed in common[task]])
                lo, hi = bootstrap_mean_ci(d)
                tests.append(dict(task=task, metric=metric, a=a, b=b, n=len(d),
                                  mean_difference=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                                  paired_dz=float(d.mean()/d.std(ddof=1)) if d.std(ddof=1) else '',
                                  a_wins=int((d>0).sum()), ties=int((d==0).sum()), b_wins=int((d<0).sum()),
                                  p_raw=exact_sign_flip(d)))
    # One Holm family includes every task, metric and pair (12 tests).
    ordered = sorted(range(len(tests)), key=lambda i: tests[i]['p_raw'])
    previous = 0.
    for rank, i in enumerate(ordered):
        previous = min(1., max(previous, (len(tests) - rank) * tests[i]['p_raw']))
        tests[i]['p_holm'] = previous
    write_csv('complete_case_summary.csv', summaries)
    write_csv('paired_comparisons.csv', tests)
    lines = ['# Mountain Car transfer: statistical comparison', '',
             'Current raw selection logs; generations 2001–7000 (5,000 transfer generations). Higher scores are better. No generations are filtered.', '',
             '## Method', '',
             '- Experimental unit: one seed/run, not a generation. Primary comparisons use the same seeds present in all three mechanisms within each task.',
             '- Learning performance: arithmetic mean of best training fitness across the 5,000 transfer generations (discrete normalized AUC). Final performance: best training fitness at generation 7000.',
             '- Two-sided exact paired sign-flip randomization tests of mean differences; Holm correction across all 12 tests. Pairing uses matching seed IDs and assumes runs across mechanisms form meaningful seed blocks; sign flipping assumes within-block exchangeability under the null.',
             '- Effect sizes: paired mean difference in reward units, paired standardized mean difference (dz), and paired win/tie/loss counts. Confidence intervals are 50,000-resample percentile bootstraps over seeds; they are pointwise 95% intervals, not multiplicity-adjusted.',
             '- Threshold success is descriptive: Pendulum ≥ −200; Acrobot ≥ −100. First crossing uses the recorded best training score and does not imply sustained or held-out success.', '',
             '## Seed coverage', '']
    for task in THRESHOLDS:
        lines.append(f'- {task.capitalize()}: {len(common[task])} matched seeds: {common[task]}.')
    lines += ['', 'Missing logs: ' + '; '.join(missing) + '.', '', '## Complete-case results', '',
              '| Task | Mechanism | n | Mean learning score | Mean final score | Threshold reached |',
              '|---|---|---:|---:|---:|---:|']
    for task in THRESHOLDS:
        for v in VARIANTS:
            ss = [r for r in records if r['task']==task and r['variant']==v and r['in_complete_case_analysis']]
            lines.append(f"| {task.capitalize()} | {LABELS[v]} | {len(ss)} | {np.mean([r['mean_transfer_score'] for r in ss]):.2f} | {np.mean([r['final_score'] for r in ss]):.2f} | {sum(r['reached_threshold'] for r in ss)}/{len(ss)} |")
    lines += ['', '## Paired comparisons', '', 'Positive differences favor the first mechanism. A large adjusted p-value does not establish equivalence.', '',
              '| Task | Metric | Comparison | Mean difference [95% CI] | Wins/ties/losses | Raw p | Holm p |',
              '|---|---|---|---:|---:|---:|---:|']
    for t in tests:
        lines.append(f"| {t['task'].capitalize()} | {t['metric']} | {LABELS[t['a']]} − {LABELS[t['b']]} | {t['mean_difference']:.2f} [{t['ci_low']:.2f}, {t['ci_high']:.2f}] | {t['a_wins']}/{t['ties']}/{t['b_wins']} | {t['p_raw']:.5g} | {t['p_holm']:.5g} |")
    lines += ['', '## Scope and limitations', '',
              'These are exploratory comparisons of best training fitness, not held-out evaluation. The transfer configurations disable validation; the loaded transfer rows contain '
              f"{sum(r['nonzero_validation_rows'] for r in records)} nonzero validation fitness values. Zero values alone do not prove how a binary was executed.", '',
              'The mechanisms also evolved different Mountain Car source populations. Differences therefore describe the entire source-training-plus-transfer pipeline; they do not isolate the causal effect of changing the mutation mechanism at transfer. Complete-case analysis assumes the missing runs do not systematically change the comparison; unavailable runs are not treated as failures. No claim of equivalence follows from a nonsignificant result.', '',
              'Per-seed metrics include all available runs and exclusions so alternative summaries can be reproduced. Bootstrap intervals and tests were selected after inspecting the plots; treat the results as exploratory rather than a preregistered confirmatory study.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, task in enumerate(THRESHOLDS):
        for j, metric in enumerate(METRICS):
            ax = axes[i,j]
            for seed in common[task]:
                rr = [next(r[metric] for r in records if r['task']==task and r['variant']==v and r['seed']==seed) for v in VARIANTS]
                ax.plot(range(3), rr, color='gray', alpha=.2, linewidth=.7)
            for k,v in enumerate(VARIANTS):
                rr = [r[metric] for r in records if r['task']==task and r['variant']==v and r['in_complete_case_analysis']]
                ax.scatter(np.full(len(rr),k), rr, s=18, alpha=.65, color=('#2563a6','#df6b20','#666666')[k])
                stat = next(s for s in summaries if s['task']==task and s['variant']==v and s['metric']==metric)
                ax.errorbar(k+.12,stat['mean'],yerr=[[stat['mean']-stat['mean_ci_low']],[stat['mean_ci_high']-stat['mean']]],fmt='D',color='black',capsize=4)
            ax.set_xticks(range(3), [LABELS[v] for v in VARIANTS])
            ax.set_title(f"{task.capitalize()} · {'Mean transfer score' if j==0 else 'Final score'} (n={len(common[task])})")
            ax.set_ylabel('Training reward (higher is better)')
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('Matched-seed transfer performance\nDots: individual runs; diamonds: means with pointwise 95% bootstrap CIs')
    fig.savefig(OUT/'comparison.png',dpi=180)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    print('\n'.join(lines[lines.index('## Complete-case results'):lines.index('## Scope and limitations')]))
    print(f'Wrote report, metrics, tests and figures to {OUT}')


if __name__ == '__main__':
    main()
