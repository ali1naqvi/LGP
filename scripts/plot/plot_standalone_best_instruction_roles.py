"""Best-policy instruction composition at requested standalone budgets."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/standalone_best_instruction_roles'
TASKS = [('mountaincar_continuous', 'Mountain Car', 5000), ('pendulum', 'Pendulum', 5000),
         ('acrobot', 'Acrobot', 5000), ('reacher', 'Reacher', 5000), ('halfcheetah', 'Half-Cheetah', 5000)]
KEYS = ['structural_introns', 'action_effective', 'mutation_only']
LABELS = ['Structural introns', 'Action effective', 'Mutation-only effective']
COLORS = ['#A8BFA3', '#98B5C9', '#D6A77E']


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=220, facecolor='white')
    fig.savefig(OUT / f'{name}.pdf', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    endpoints, summaries, coverage, trajectories, data = [], [], [], [], {}
    for env, title, end in TASKS:
        all_seeds = {}
        suffix = '_vector' if env in {'reacher', 'halfcheetah'} else ''
        folder = ROOT / 'experiments' / f'{env}_execution_modified_rates{suffix}' / 'logs/selection'
        for path in sorted(folder.glob('selection.*.0.csv')):
            seed = int(path.name.split('.')[1]); records = {}
            with path.open() as f:
                for row in csv.DictReader(f):
                    g = int(row['generation'])
                    if g > end: continue
                    total = int(row['program_instruction_count'])
                    effective = int(row['effective_program_instruction_count'])
                    mutation = int(row['self_modification_only_instruction_count'])
                    if not 0 <= mutation <= effective <= total or total == 0:
                        raise ValueError(f'Invalid counts: {path}, generation {g}')
                    record = dict(environment=env, seed=seed, generation=g, team_id=row['team_id'], total=total)
                    for key, count in zip(KEYS, (total-effective, effective-mutation, mutation)):
                        record[key+'_count'] = count; record[key+'_fraction'] = count/total
                    if g in records: raise ValueError(f'Duplicate generation: {path}: {g}')
                    records[g] = record
            complete = set(range(end+1)).issubset(records)
            coverage.append(dict(environment=env, seed=seed, requested_generation=end,
                                 endpoint_available=end in records, complete_trajectory=complete))
            if end in records:
                endpoints.append(records[end]); all_seeds[seed] = records
        if not all_seeds: raise ValueError(f'No endpoint data for {env}')
        # Keep a fixed cohort throughout each trajectory; never fill missing generations.
        cohort = {s:r for s,r in all_seeds.items() if set(range(end+1)).issubset(r)}
        if not cohort: raise ValueError(f'No complete trajectories for {env}')
        means = np.array([[np.mean([r[g][k+'_fraction'] for r in cohort.values()]) for k in KEYS]
                          for g in range(end+1)])
        if not np.allclose(means.sum(axis=1), 1): raise ValueError('Composition does not sum to one')
        data[env] = means, len(cohort)
        for g, values in enumerate(means):
            trajectories.append(dict(environment=env,generation=g,n_seeds=len(cohort),
                                     **{k+'_fraction':v for k,v in zip(KEYS,values)}))
        group = [r[end] for r in all_seeds.values()]
        record = dict(environment=env, generation=end, n_seeds=len(group))
        for key in KEYS:
            for unit in ['count', 'fraction']:
                values = [r[key+'_'+unit] for r in group]
                record['mean_'+key+'_'+unit] = float(np.mean(values))
                record['median_'+key+'_'+unit] = float(np.median(values))
        summaries.append(record)
        print(title, end, 'endpoint seeds', len(group), 'trajectory seeds', len(cohort), flush=True)
    write('best_individuals_at_requested_generations.csv', endpoints)
    write('endpoint_summary.csv', summaries)
    write('seed_coverage.csv', coverage)
    write('mean_trajectories.csv', trajectories)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#BBBBBB','pdf.fonttype':42})
    for style in ['stacked', 'lines', 'lines_smoothed']:
        fig, axes = plt.subplots(1,5,figsize=(12.8,3.2),sharey=True)
        fig.subplots_adjust(left=.055,right=.975,bottom=.29,top=.88,wspace=.32)
        for index,(ax,(env,title,end)) in enumerate(zip(axes,TASKS)):
            means,n = data[env]
            if style == 'stacked':
                ax.stackplot(np.arange(end+1), means.T, colors=COLORS, labels=LABELS, linewidth=0)
            else:
                for values, color, label in zip(means.T, COLORS, LABELS):
                    if style == 'lines_smoothed':
                        kernel = np.ones(101)
                        values = np.convolve(values, kernel, mode='same') / np.convolve(np.ones(len(values)), kernel, mode='same')
                    ax.plot(np.arange(end+1), values, color=color, label=label, linewidth=1.5 if style == 'lines_smoothed' else .8)
            ax.set(xlim=(0,end), ylim=(0,1), xlabel='Generation')
            ax.set_title(f'({chr(97+index)}) {title}',loc='left',fontsize=9,pad=8)
            ax.tick_params(axis='both',labelsize=8)
            ax.xaxis.label.set_size(8)
            ax.set_xticks([0,end//2,end]); ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',color='#E5E5E5' if style != 'stacked' else 'white',linewidth=.7)
        axes[0].set_ylabel('Instructions (%)',fontsize=8)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=8)
        save(fig,'best_instruction_composition_trajectories' + ('_' + style if style != 'stacked' else ''))
    # Relative composition with explicit band boundaries and endpoint shares.
    fig, axes = plt.subplots(1, 5, figsize=(13.8, 3.8), sharey=True)
    fig.subplots_adjust(left=.055, right=.985, bottom=.34, top=.88, wspace=.25)
    window = np.ones(101)
    for col, (ax, (env, title, end)) in enumerate(zip(axes, TASKS)):
        means, n = data[env]
        smooth = np.column_stack([np.convolve(v, window, mode='same') /
                                 np.convolve(np.ones(len(v)), window, mode='same') for v in means.T])
        ax.stackplot(np.arange(end+1), smooth.T, colors=COLORS, labels=LABELS, linewidth=0)
        for boundary in np.cumsum(smooth, axis=1)[:, :-1].T:
            ax.plot(np.arange(end+1), boundary, color='white', linewidth=.9)
        lower = 0
        for value in smooth[-1]:
            ax.text(end*.985, lower+value/2, f'{value:.0%}', ha='right', va='center', fontsize=8,
                    color='#1F2933', bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
            lower += value
        ax.set(xlim=(0,end), ylim=(0,1), xlabel='Generation', xticks=[0,end//2,end], yticks=[0,.25,.5,.75,1])
        ax.set_title(f'({chr(97+col)}) {title}', loc='left', fontsize=10, pad=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel('Instructions (%)', fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.13),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5,.075,'Band thickness = share of instructions; all three bands total 100%.', ha='center', fontsize=9)
    fig.text(.5,.025,'101-generation moving average; labels show the smoothed final shares.', ha='center', fontsize=8, color='#555555')
    save(fig, 'best_instruction_composition_relative_clear')
    # Separate components to avoid crossing lines; smooth only for display.
    fig, axes = plt.subplots(3, 5, figsize=(12.8, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.145, right=.98, bottom=.13, top=.93, wspace=.18, hspace=.22)
    window = np.ones(101)
    for col, (env, title, end) in enumerate(TASKS):
        means, n = data[env]
        for row, (label, color) in enumerate(zip(LABELS, ['#65865D', '#487F9F', '#B7753E'])):
            ax = axes[row, col]
            values = means[:, row]
            smooth = np.convolve(values, window, mode='same') / np.convolve(np.ones(len(values)), window, mode='same')
            ax.plot(np.arange(end+1), smooth, color=color, linewidth=1.6)
            ax.set(xlim=(0, end), ylim=(0, 1), yticks=[0, .25, .5, .75, 1], xticks=[0, end//2, end])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.tick_params(labelsize=8)
            ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
            if row == 0:
                ax.set_title(f'({chr(97+col)}) {title}', loc='left', fontsize=9, pad=9)
            if col == 0:
                ax.annotate(label.replace(' effective', '\neffective'), xy=(-.36, .5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=9, color=color)
    fig.supylabel('Instructions (%)', x=.012, fontsize=10)
    fig.supxlabel('Generation', y=.06, fontsize=10)
    fig.text(.56, .015, '101-generation centered moving average of seed means', ha='center', fontsize=8, color='#555555')
    save(fig, 'best_instruction_composition_trajectories_lines_clear')
    fig,ax = plt.subplots(figsize=(10,4.3),layout='constrained')
    names=[]
    for i,((env,title,end),summary) in enumerate(zip(TASKS,summaries)):
        left=0
        names.append(f'{title} · g={end:,}')
        for key,color,label in zip(KEYS,COLORS,LABELS):
            value=summary['mean_'+key+'_fraction']
            ax.barh(i,value,left=left,height=.55,color=color,label=label if i==0 else None)
            left+=value
    ax.set(yticks=range(5),yticklabels=names,xlim=(0,1),xlabel='Mean best-individual instruction composition')
    ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='y',length=0)
    ax.set_title('Standalone · runtime-modified · requested generations',pad=12)
    fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    save(fig,'best_instruction_composition_endpoints')
    (OUT/'README.md').write_text('''# Best-individual instruction composition

Uses the training-selected best individual in each seed at each generation, from standalone runtime-modified selection CSVs. These are not population means or population medians. Each seed has equal weight; fractions are computed per individual before averaging. The selected best individual can change over time.

Action effective = effective minus mutation-only effective; structural introns = total minus effective. Shared dependencies belong to action effective. Counts follow the engine's structural definition and do not measure experimental neutrality.

Stacked plots use means across seeds to preserve a 100% composition. Endpoint component medians are saved separately in endpoint_summary.csv; they need not sum to 100% and are not normalized or stacked. Trajectories use every logged generation, without smoothing, and only seeds with a complete history through the requested endpoint. Seed inclusion is recorded in seed_coverage.csv. No missing endpoint is replaced with an earlier generation.

Requested budget: generation 5000 for every environment. Reacher and Half-Cheetah use the vector experiment folders. See seed_coverage.csv for current endpoint availability and complete trajectory inclusion; incomplete runs are excluded from trajectory means.
''')


if __name__ == '__main__': main()
