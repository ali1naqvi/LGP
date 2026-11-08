"""Plot within-run training fitness improvement over a trailing generation window."""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_training_requested_ranges import PANELS
from plot_mutation_validation_comparison import ROOT, LABELS, COLORS, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=int, default=100)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    window = args.window
    out = ROOT / 'output' / 'training_rate_of_improvement' / f'window_{window}'
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 5), layout='constrained')
    rows = []
    for index, (ax, (environment, endpoint, experiments)) in enumerate(zip(axes, PANELS)):
        if window > endpoint:
            raise ValueError(f'Window exceeds endpoint for {environment}')
        for experiment, label, color in zip(experiments, LABELS, COLORS):
            runs, total = load_experiment(experiment, 'best_fitness')
            # A fixed cohort prevents changing run membership from creating apparent progress.
            complete = [run for run in runs if all(g in run for g in range(endpoint + 1))]
            if not complete:
                raise ValueError(f'No complete training histories for {experiment}')
            fitness = np.array([[run[g] for g in range(endpoint + 1)] for run in complete])
            rates = (fitness[:, window:] - fitness[:, :-window]) / window
            # The windowed change equals the trailing mean of one-generation changes.
            cumulative = np.cumsum(np.diff(fitness, axis=1), axis=1)
            cumulative = np.pad(cumulative, ((0, 0), (1, 0)))
            np.testing.assert_allclose(rates, (cumulative[:, window:] - cumulative[:, :-window]) / window,
                                       atol=1e-10, rtol=1e-9)
            q25, median, q75 = np.quantile(rates, [.25, .5, .75], axis=0)
            generations = np.arange(window, endpoint + 1)
            ax.plot(generations, median, color=color, linewidth=1.5, label=label)
            ax.fill_between(generations, q25, q75, color=color, alpha=.14)
            rows.extend(dict(environment=environment, variant=label, experiment=experiment,
                             generation=int(g), window=window, n_runs=len(complete),
                             total_logs=total, median=m, q25=lo, q75=hi)
                        for g, m, lo, hi in zip(generations, median, q25, q75))
            print(f'{environment}: {label}: {len(complete)}/{total} complete runs through {endpoint}')
        ax.axhline(0, color='#555555', linewidth=.8, linestyle='--')
        ax.set(title=f'({chr(97 + index)}) {environment.replace(" Continuous", "")}',
               xlim=(0, endpoint), xlabel='Generation')
        ax.set_xscale('function', functions=(
            lambda x: np.where(x <= 500, x, 500 + (x - 500) / 9),
            lambda x: np.where(x <= 500, x, 500 + (x - 500) * 9)))
        ax.set_xticks([0, 250, 500, 2750, 5000])
        ax.set_xticklabels(['0', '250', '500', '2,750', '5,000'])
        ax.axvline(500, color='#777777', linewidth=.8, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Training fitness improvement / generation')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False,
               title=f'Trailing {window}-generation rate · Median and 25th–75th percentile · X-axis: equal widths for 0–500 and 500–5,000')
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'training_rate_of_improvement.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    with (out / 'rate_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'README.md').write_text(
        f'# Training rate of improvement\n\n'
        f'Uses the same five environments, three variants, selection-log best_fitness metric, '
        f'and generation endpoints as plot_training_requested_ranges.py.\n\n'
        f'For each run, rate(g) = [fitness(g) - fitness(g-{window})] / {window}. '
        f'This is the trailing mean of {window} consecutive-generation changes, in reward units per generation. '
        f'The first value is at generation {window}; there is no padding, interpolation, or future data. '
        f'Only runs complete from generation 0 through the panel endpoint contribute, keeping each cohort fixed. '
        f'Counts are recorded in rate_summary.csv.\n\n'
        f'The x-axis is piecewise linear: 0–500 occupies the left half and 500–5,000 '
        f'occupies the right half, with a dotted divider at generation 500.\n\n'
        f'Lines show the median of within-run rates; shading shows their 25th–75th percentiles, '
        f'not a confidence interval. Positive values mean increasing training fitness. '
        f'Training scores may vary with evaluation conditions and the selected individual; '
        f'this is descriptive progress, not a causal estimate. Reward scales and y axes differ between environments.\n')
    print(out / 'training_rate_of_improvement.png')


if __name__ == '__main__':
    main()
