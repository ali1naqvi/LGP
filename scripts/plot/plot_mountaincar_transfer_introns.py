"""Minimal intron plots of matched-seed Mountain Car transfer policies."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/mountaincar_transfer_introns'
VARIANTS = ('execution_modified_rates', 'fixed_individual', 'fixed_rates')
LABELS = ('Runtime-modified', 'Gene-encoded', 'Static')
COLORS = ('#3574A0', '#C17A45', '#686868')
TASKS = ('pendulum', 'acrobot')
FIELDS = ('program_instruction_count', 'effective_program_instruction_count',
          'self_modification_only_instruction_count')


def read_counts(path, start, end):
    values = np.full((end-start+1, 3), np.nan)
    seen = set()
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            g = int(row['generation'])
            if not start <= g <= end:
                continue
            if g in seen:
                raise ValueError(f'Duplicate generation {g}: {path}')
            seen.add(g)
            values[g-start] = [float(row[key]) for key in FIELDS]
    if not np.isfinite(values).all():
        raise ValueError(f'Incomplete counts: {path}')
    total, effective, self_only = values.T
    if not ((total > 0) & (effective >= self_only) & (self_only >= 0) & (effective <= total)).all():
        raise ValueError(f'Invalid instruction counts: {path}')
    return values


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=220, facecolor='white')
    fig.savefig(OUT/f'{name}.pdf', facecolor='white')
    plt.close(fig)


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelcolor':'#444444', 'xtick.color':'#555555',
                         'ytick.color':'#555555', 'axes.edgecolor':'#BBBBBB',
                         'pdf.fonttype':42})
    data, seedsets, milestones, aggregates = {}, {}, [], []
    for task in TASKS:
        paths = {v:{seed: ROOT/f'experiments/mountaincar_continuous_to_{task}_{v}/seed_{seed}/{task}/logs/selection/selection.{seed}.0.csv'
                    for seed in range(1,21)} for v in VARIANTS}
        seeds = [s for s in range(1,21) if all(paths[v][s].exists() and
                 (ROOT/f'experiments/mountaincar_continuous_to_{task}_{v}/seed_{s}/mountaincar_continuous/logs/selection/selection.{s}.0.csv').exists()
                 for v in VARIANTS)]
        seedsets[task] = seeds
        print(f'{task}: {len(seeds)} matched seeds: {seeds}')
        for v in VARIANTS:
            runs = []
            for seed in seeds:
                source = ROOT/f'experiments/mountaincar_continuous_to_{task}_{v}/seed_{seed}/mountaincar_continuous/logs/selection/selection.{seed}.0.csv'
                manifest = json.loads((source.parents[3]/'transfer.json').read_text())
                if manifest['source_mode'] != 'fresh_mountaincar_continuous':
                    raise ValueError(f'Unexpected source mode: {source}')
                run = np.concatenate((read_counts(source,0,2000),read_counts(paths[v][seed],2001,7000)))
                runs.append(run)
                for g in (0,2000,2001,3000,7000):
                    t,e,s = run[g]
                    milestones.append(dict(task=task,variant=v,seed=seed,generation=g,
                                           total=int(t),effective=int(e),introns=int(t-e),
                                           intron_fraction=float((t-e)/t),self_modification_only=int(s)))
            values = np.stack(runs)
            total,effective,self_only = values.transpose(2,0,1)
            introns = total-effective
            data[task,v] = dict(count=introns,fraction=introns/total,
                                composition=np.stack(((effective-self_only)/total,self_only/total,introns/total),axis=-1))
            for metric in ('count','fraction'):
                q1,med,q3 = np.quantile(data[task,v][metric],[.25,.5,.75],axis=0)
                for g in range(7001):
                    aggregates.append(dict(task=task,variant=v,metric=metric,generation=g,n=len(seeds),
                                           q1=float(q1[g]),median=float(med[g]),q3=float(q3[g])))
    write_csv('milestones_by_seed.csv', milestones)
    write_csv('trajectories_median_iqr.csv', aggregates)
    x = np.arange(7001)
    for metric,name,ylabel in (('fraction','intron_fraction','Intron fraction'),('count','intron_count','Intron instructions')):
        fig,axes = plt.subplots(1,2,figsize=(10,3.5),sharey=True,layout='constrained')
        for ax,task in zip(axes,TASKS):
            for v,label,color in zip(VARIANTS,LABELS,COLORS):
                q1,med,q3 = np.quantile(data[task,v][metric],[.25,.5,.75],axis=0)
                ax.fill_between(x,q1,q3,color=color,alpha=.07,linewidth=0)
                ax.plot(x,med,color=color,lw=1.3,label=label)
            ax.axvline(2000.5,color='#999999',ls='--',lw=.8)
            ax.text(2000.5,1.015,'transfer',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#777777')
            ax.set_title(f'Mountain Car → {task.capitalize()}  ·  n={len(seedsets[task])}',pad=22,fontsize=11)
            ax.set_xlabel('Generation')
            ax.set_xticks([0,2000,4000,6000],['0','2,000','4,000','6,000'])
            ax.set_xlim(0,7000)
            ax.set_ylim(bottom=0)
            if metric=='fraction':
                ax.set_ylim(0,1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y',color='#EEEEEE',linewidth=.6)
        axes[0].set_ylabel(ylabel)
        handles,labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        save(fig,name)
    fig,axes = plt.subplots(1,2,figsize=(9,3.5),sharey=True,layout='constrained')
    composition_rows=[]
    categories=('Action-effective','Self-modification only','Introns')
    for ax,task in zip(axes,TASKS):
        means=np.stack([data[task,v]['composition'][:,-1,:].mean(axis=0) for v in VARIANTS])
        bottom=np.zeros(3)
        for k,(label,color) in enumerate(zip(categories,('#464646','#98B5C9','#E5E5E5'))):
            ax.bar(range(3),means[:,k],bottom=bottom,width=.55,color=color,label=label)
            bottom+=means[:,k]
        for i,v in enumerate(VARIANTS):
            composition_rows.append(dict(task=task,variant=v,n=len(seedsets[task]),
                                        action_effective_fraction=float(means[i,0]),
                                        self_modification_only_fraction=float(means[i,1]),
                                        intron_fraction=float(means[i,2])))
        ax.set_xticks(range(3),['Runtime','Genes','Static'])
        ax.set_title(f'{task.capitalize()}  ·  n={len(seedsets[task])}',fontsize=11)
        ax.set_ylim(0,1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x',length=0)
    axes[0].set_ylabel('Mean instruction fraction')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Final policy composition · generation 7,000',fontsize=12)
    save(fig,'final_instruction_composition')
    write_csv('final_composition.csv',composition_rows)
    (OUT/'README.md').write_text('''# Mountain Car transfer introns

Figures use current raw selection logs for the logged best training policy, not population-average instructions. Mountain Car history is generations 0–2000; transfer is 2001–7000. Curves are unsmoothed seed medians, with light interquartile bands. Each task uses only seeds available for all three mechanisms, consistently before and after transfer.

- Introns = total program instructions − effective instructions, using the engine’s structural dependency analysis. This is not a measurement of executed frequency or experimentally demonstrated neutrality.
- Effective instructions include self-modification dependencies in runtime-modified runs. Self-modification-only instructions are therefore shown separately from introns.
- Final composition shows the mean of per-seed fractions at generation 7000; the three parts sum to 100%. Action-effective = effective − self-modification-only (the action/bid dependency slice under the engine definition).
- Source and transfer histories both come from the same mountaincar_continuous_to_<task>_<variant>/seed_<seed> run. Earlier figures used unrelated standalone Mountain Car histories; these have been corrected.
- These are descriptive structural plots, not statistical tests or held-out fitness results. Complete-case exclusions can affect representativeness.

See milestones_by_seed.csv for individual counts, trajectories_median_iqr.csv for all plotted curve values, and final_composition.csv for bar values. Regenerate with scripts/plot/plot_mountaincar_transfer_introns.py.
'''+ '\nMatched seeds:\n'+''.join(f'- {t}: {seedsets[t]}\n' for t in TASKS))
    print(f'Wrote plots and data to {OUT}')
    for r in composition_rows:print(r)


if __name__=='__main__':
    main()
