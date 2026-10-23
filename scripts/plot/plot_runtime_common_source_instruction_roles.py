"""Common Mountain Car reference in both panels, with separate target cohorts."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/mountaincar_runtime_population_instruction_roles'
OUT=ROOT/'output/mountaincar_runtime_elite_instruction_roles'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'population_roles_by_seed.csv').open() as f:
        rows=[{k:v if k=='task' else int(v) if k in ('seed','generation','n_policies') else float(v)
               for k,v in r.items()} for r in csv.DictReader(f)]
    reference=[r for r in rows if r['task']=='pendulum' and r['generation']<=2000]
    displayed=[]
    for task in ('pendulum','acrobot'):
        displayed += [dict(r,task=task,cohort='common_mountaincar_reference',original_pipeline='pendulum') for r in reference]
        displayed += [dict(r,cohort='target',original_pipeline=task) for r in rows if r['task']==task and r['generation']>2000]
    with (OUT/'common_source_display_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(displayed[0]));w.writeheader();w.writerows(displayed)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#BBBBBB','pdf.fonttype':42})
    def axes_setup(ax,task):
        target_n=len({r['seed'] for r in displayed if r['task']==task and r['cohort']=='target'})
        ax.set_title(f'Mountain Car (n=20) → {task.capitalize()} (n={target_n})',fontsize=11,pad=21)
        ax.axvline(2000,color='#999999',ls='--',lw=.8)
        ax.text(2000,1.015,'transfer',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#777777')
        ax.set_xlim(0,7000);ax.set_ylim(bottom=0)
        ax.set_xticks([0,2000,4000,6000],['0','2,000','4,000','6,000']);ax.set_xlabel('Generation')
        ax.grid(axis='y',color='#EEEEEE',lw=.6)
    def cohorts(task):
        for name in ('common_mountaincar_reference','target'):
            group=[r for r in displayed if r['task']==task and r['cohort']==name]
            if name=='target':
                # Join the displayed target region to the common reference at
                # the boundary; this anchor is a display value, not a target sample.
                group += [dict(r,generation=2000) for r in reference if r['generation']==2000]
            yield sorted({r['generation'] for r in group}),group
    def save(fig,name):
        fig.savefig(OUT/f'{name}_common_source.png',dpi=220,facecolor='white')
        fig.savefig(OUT/f'{name}_common_source.pdf',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
    for ax,task in zip(axes,('pendulum','acrobot')):
        for segment,(generations,group) in enumerate(cohorts(task)):
            parts=[[np.mean([r[f'mean_{key}_fraction'] for r in group if r['generation']==g]) for g in generations]
                   for key in ('action_effective','mutation_only','structural_introns')]
            assert np.allclose(np.sum(parts,axis=0),1)
            ax.stackplot(generations,*parts,colors=['#98B5C9','#D6A77E','#E3E3E3'],lw=0,
                         labels=['Action effective (includes shared)','Mutation-only effective','Structural introns'] if segment==0 else [])
        axes_setup(ax,task);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Mean elite composition')
    fig.suptitle('Runtime-modified · common Mountain Car reference',fontsize=12)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False,fontsize=8.5)
    save(fig,'runtime_instruction_composition')
    for unit in ('fraction','count'):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
        for ax,task in zip(axes,('pendulum','acrobot')):
            for segment,(generations,group) in enumerate(cohorts(task)):
                for key,label,color,style in (
                    ('structural_introns','Structural introns','#777777','-'),
                    ('mutation_only','Mutation-only effective','#C17A45','-'),
                    ('outside_action','Outside action slice','#3574A0','--')):
                    qs=np.array([np.quantile([r[f'mean_{key}_{unit}'] for r in group if r['generation']==g],[.25,.5,.75]) for g in generations]).T
                    lo,med,hi=qs
                    ax.fill_between(generations,lo,hi,color=color,alpha=.065,lw=0)
                    ax.plot(generations,med,color=color,ls=style,lw=1.4,marker='.',ms=4,label=label if segment==0 else None)
            axes_setup(ax,task)
            if unit=='fraction':ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set_ylabel('Elite mean instruction fraction' if unit=='fraction' else 'Mean instructions per elite policy')
        fig.suptitle('Runtime-modified · common Mountain Car reference',fontsize=12)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        save(fig,f'runtime_instruction_roles_{unit}')
    # Verify both displayed source cohorts are exactly identical seed-by-seed.
    a=[{k:v for k,v in r.items() if k!='task'} for r in displayed if r['task']=='pendulum' and r['cohort']=='common_mountaincar_reference']
    b=[{k:v for k,v in r.items() if k!='task'} for r in displayed if r['task']=='acrobot' and r['cohort']=='common_mountaincar_reference']
    assert a==b and len({r['seed'] for r in reference})==20
    (OUT/'COMMON_SOURCE.md').write_text('''# Common Mountain Car reference display

Both panels use exactly the same 20-seed Mountain Car reference from the Pendulum transfer pipeline through generation 2000. After that, each panel uses its actual available target cohort: 20 Pendulum seeds and 19 Acrobot seeds. Each observation remains a population mean over 1000 engine-elite survivors; the stacked composition averages across seeds and the line figures show medians and IQRs across seeds.

This is a shared-reference visualization, not a claim that the Acrobot trajectories started from these exact source populations. The actual Acrobot source histories differ in the saved experiments. The target region is visually anchored to the common reference at generation 2000 to avoid a blank gap. Its first measured target checkpoint remains generation 2100; the connecting region is a display interpolation between different cohorts, not a measured target trajectory. The displayed data and provenance are in common_source_display_data.csv. Original actual-history plots are retained alongside these files.
''')
    print('Verified identical 20-seed Mountain Car distributions in both panels; target n=20 / n=19.')


if __name__=='__main__':main()
