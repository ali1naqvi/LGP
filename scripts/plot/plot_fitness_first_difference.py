"""Consecutive-generation changes in standalone runtime best training fitness."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
from plot_standalone_best_instruction_roles import TASKS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/standalone_fitness_first_difference'


def write(name, rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={}; per_seed=[]; aggregates=[]; coverage=[]
    for env,title,end in TASKS:
        curves=[]
        folder=ROOT/'experiments'/f'{env}_execution_modified_rates'/'logs/selection'
        for path in sorted(folder.glob('selection.*.0.csv')):
            seed=int(path.name.split('.')[1]); values={}
            with path.open() as f:
                for row in csv.DictReader(f):
                    g=int(row['generation'])
                    if g>end: continue
                    if g in values: raise ValueError(f'Duplicate generation: {path}, {g}')
                    values[g]=float(row['best_fitness'])
            valid=all(g in values and np.isfinite(values[g]) for g in range(end+1))
            coverage.append(dict(environment=env,seed=seed,included=valid,end_generation=end))
            if not valid: continue
            fitness=np.array([values[g] for g in range(end+1)])
            differences=np.diff(fitness)
            assert np.isclose(differences.sum(),fitness[-1]-fitness[0])
            curves.append(differences)
            per_seed.extend(dict(environment=env,seed=seed,generation=g,fitness=fitness[g],
                                 previous_fitness=fitness[g-1],delta_fitness=differences[g-1]) for g in range(1,end+1))
        if not curves: raise ValueError(f'No complete histories: {env}')
        changes=np.array(curves); mean=changes.mean(axis=0)
        smooth=np.convolve(mean,np.ones(50)/50,mode='valid')
        data[env]=(mean,smooth)
        for i,value in enumerate(mean):
            aggregates.append(dict(environment=env,generation=i+1,n_seeds=len(curves),mean_delta_fitness=value,
                                   median_delta_fitness=float(np.median(changes[:,i])),
                                   trailing_50_mean_delta=smooth[i-49] if i>=49 else ''))
        print(f'{title}: {len(curves)} complete seeds, generations 0–{end}',flush=True)
    write('per_seed_first_differences.csv',per_seed)
    write('aggregate_first_differences.csv',aggregates)
    write('seed_coverage.csv',coverage)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#BBBBBB','pdf.fonttype':42})
    for smoothed in [False,True]:
        fig,axes=plt.subplots(1,5,figsize=(12.8,3.2))
        fig.subplots_adjust(left=.065,right=.975,bottom=.25,top=.86,wspace=.48)
        for i,(ax,(env,title,end)) in enumerate(zip(axes,TASKS)):
            mean,smooth=data[env]
            x=np.arange(50,end+1) if smoothed else np.arange(1,end+1)
            y=smooth if smoothed else mean
            ax.axhline(0,color='#999999',lw=.7)
            ax.plot(x,y,color='#507F9D',lw=1 if smoothed else .65)
            ax.set_title(f'({chr(97+i)}) {title}',loc='left',fontsize=9,pad=8)
            ax.set(xlim=(0,end),xlabel='Generation')
            ax.set_xticks([0,end//2,end]);ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.yaxis.set_major_locator(MaxNLocator(5));ax.tick_params(labelsize=7)
            ax.grid(axis='y',color='#E6E6E6',lw=.6)
        axes[0].set_ylabel('Change in best training fitness / generation')
        label='50-generation trailing mean of first differences' if smoothed else 'First differences · mean across runs'
        fig.text(.5,.04,label,ha='center',fontsize=9)
        name='fitness_first_difference_50_generation' if smoothed else 'fitness_first_difference_raw'
        for extension in ['png','pdf']:
            fig.savefig(OUT/f'{name}.{extension}',dpi=220,facecolor='white')
        plt.close(fig)
    (OUT/'README.md').write_text('''# Fitness first differences

Standalone runtime-modified experiments, same environment order and generation budgets as the instruction-role figures. Uses best_fitness (training), not validation or test fitness. Compute delta_fitness(g) = best_fitness(g) - best_fitness(g-1) independently for each seed, then average those changes across complete runs. A fixed cohort is used through each environment's endpoint: 20 runs per environment except Acrobot, with 18. There is no interpolation, missing-generation bridging, or running-best transformation.

Positive differences indicate an increase in reported fitness; negative differences indicate a decrease. These describe changes in the training-selected score and can reflect evaluation variability and changes in the selected individual as well as evolutionary progress. Reward units differ across environments; axes are independently scaled.

The raw plot shows consecutive-generation differences. The smoothed plot shows their trailing 50-generation arithmetic mean, equivalent to (mean_fitness(g)-mean_fitness(g-50))/50 for this fixed cohort. It starts at generation 50, uses no padding or future values, and remains in fitness units per generation. This averaging is descriptive, not a confidence interval. Endpoint telescoping checks validate every seed's difference series. CSVs preserve raw seed differences, aggregate means/medians, and coverage.
''')


if __name__=='__main__': main()
