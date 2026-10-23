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
TASKS = [('mountaincar_continuous', 'Mountain Car', 2000), ('pendulum', 'Pendulum', 1000),
         ('acrobot', 'Acrobot', 5000), ('reacher', 'Reacher', 3000), ('halfcheetah', 'Half-Cheetah', 4000)]
KEYS = ['action_effective', 'mutation_only', 'structural_introns']
LABELS = ['Action effective', 'Mutation-only effective', 'Structural introns']
COLORS = ['#98B5C9', '#D6A77E', '#E3E3E3']


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
        folder = ROOT / 'experiments' / f'{env}_execution_modified_rates' / 'logs/selection'
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
                    for key, count in zip(KEYS, (effective-mutation, mutation, total-effective)):
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
    fig, axes = plt.subplots(1,5,figsize=(12.8,3.2),sharey=True)
    fig.subplots_adjust(left=.055,right=.975,bottom=.29,top=.88,wspace=.32)
    for index,(ax,(env,title,end)) in enumerate(zip(axes,TASKS)):
        means,n = data[env]
        ax.stackplot(np.arange(end+1), means.T, colors=COLORS, labels=LABELS, linewidth=0)
        ax.set(xlim=(0,end), ylim=(0,1), xlabel='Generation')
        ax.set_title(f'({chr(97+index)}) {title}',loc='left',fontsize=9,pad=8)
        ax.tick_params(axis='both',labelsize=8)
        ax.xaxis.label.set_size(8)
        ax.set_xticks([0,end//2,end]); ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',color='white',linewidth=.7)
    axes[0].set_ylabel('Mean best-individual composition',fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=8)
    save(fig,'best_instruction_composition_trajectories')
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

Requested budgets: Mountain Car 2000, Pendulum 1000, Half-Cheetah 4000, Reacher 3000, Acrobot 5000. Acrobot seeds 3 and 18 lack generation 5000; the other four environments each have 20 seeds at their requested endpoints.
''')


if __name__ == '__main__': main()
