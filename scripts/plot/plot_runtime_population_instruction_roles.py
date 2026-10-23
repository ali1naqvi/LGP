"""Separate action dependencies, mutation-only dependencies and structural introns.

Uses every available complete runtime seed and the same sampled checkpoints as before.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/plot'))
from plot_mountaincar_population_introns import (GENERATIONS, INS, effective_count,
                                                load_parameters, output_registers)
OUT=ROOT/'output/mountaincar_runtime_population_instruction_roles'
VARIANT='execution_modified_rates'


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def population(path,parameters):
    if parameters.get('self_modifying')!=1:
        raise ValueError(f'Expected runtime-modified parameters: {path}')
    programs,teams={},{}
    with path.open() as f:
        for line in f:
            if line.startswith('RegisterMachine:'):
                p=line.strip().split(':');pid=int(p[1]);slots=int(p[10])
                instructions=tuple(i for i in p[11:] if INS.fullmatch(i))
                full_roots=tuple(sorted(output_registers(parameters,slots,True)))
                action_roots=tuple(sorted(output_registers(parameters,slots,False)))
                # Keep runtime mode True in BOTH analyses so S6 decoy treatment
                # is identical. Only the live output roots change.
                full=effective_count(instructions,slots,full_roots,True)
                action=effective_count(instructions,slots,action_roots,True)
                total=len(instructions)
                if not 0<=action<=full<=total or total==0:
                    raise ValueError(f'Invalid role decomposition: {path}, program {pid}')
                programs[pid]=(total,full,action,int(p[3]))
            elif line.startswith('team:'):
                p=line.strip().split(':');teams[int(p[1])]=[int(i) for i in p[7:]]
    members=[p for group in teams.values() for p in group]
    if (not members or any(len(g)!=1 for g in teams.values()) or
        len(members)!=len(set(members)) or set(members)!=set(programs) or
        any(p[3]>=0 for p in programs.values())):
        raise ValueError(f'Expected unique atomic program per policy: {path}')
    total,full,action=np.array([programs[p][:3] for p in members],dtype=float).T
    introns=total-full;mutation_only=full-action;non_action=total-action
    metrics=dict(n_policies=len(members),mean_total=float(total.mean()))
    for key,values in [('action_effective',action),('mutation_only',mutation_only),
                       ('structural_introns',introns),('outside_action',non_action)]:
        metrics[f'mean_{key}_count']=float(values.mean())
        metrics[f'mean_{key}_fraction']=float((values/total).mean())
    if not np.allclose(action/total+mutation_only/total+introns/total,1):
        raise ValueError('Fractions do not sum to one')
    return programs,teams,metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-csv',action='store_true',help='Redraw from validated population metrics')
    args=parser.parse_args()
    if args.from_csv:
        with (OUT/'population_roles_by_seed.csv').open() as f:
            records=[{k:v if k=='task' else int(v) if k in ('seed','generation','n_policies') else float(v)
                      for k,v in r.items()} for r in csv.DictReader(f)]
        with (OUT/'validation.csv').open() as f:
            audits=list(csv.DictReader(f))
        seedsets={task:sorted({r['seed'] for r in records if r['task']==task})
                  for task in ('pendulum','acrobot')}
        render(records,audits,seedsets)
        return
    baseline=ROOT/'output/mountaincar_population_introns/population_by_seed.csv'
    with baseline.open() as f:
        prior=[r for r in csv.DictReader(f) if r['variant']==VARIANT]
    seedsets, coverage = {}, []
    for task in ('pendulum','acrobot'):
        seedsets[task] = []
        for seed in range(1,21):
            base=ROOT/f'experiments/mountaincar_continuous_to_{task}_{VARIANT}/seed_{seed}'
            required=[base/'transfer.json', base/f'{task}/checkpoints/cp.2000.{seed}.0.rslt']
            for stage in ('mountaincar_continuous',task):
                required.extend([base/f'{stage}/parameters.yaml',
                                 base/f'{stage}/logs/selection/selection.{seed}.0.csv'])
            for g in GENERATIONS:
                stage='mountaincar_continuous' if g<=2000 else task
                required.append(base/f'{stage}/checkpoints/cp.{g}.{seed}.0.rslt')
            missing=[str(path.relative_to(base)) for path in required if not path.exists()]
            coverage.append(dict(task=task, seed=seed, included=not missing,
                                 missing_files='; '.join(missing)))
            if not missing:seedsets[task].append(seed)
        print(f'{task}: using all {len(seedsets[task])} complete runtime seeds: {seedsets[task]}',flush=True)
    write_csv('seed_coverage.csv',coverage)
    lookup={(r['task'],int(r['seed']),int(r['generation'])):r for r in prior}
    records,audits=[],[]
    for task,seeds in seedsets.items():
        for seed in seeds:
            base=ROOT/f'experiments/mountaincar_continuous_to_{task}_{VARIANT}/seed_{seed}'
            if json.loads((base/'transfer.json').read_text())['source_mode']!='fresh_mountaincar_continuous':
                raise ValueError(f'Unexpected source mode: {base}')
            a=base/f'mountaincar_continuous/checkpoints/cp.2000.{seed}.0.rslt'
            b=base/f'{task}/checkpoints/cp.2000.{seed}.0.rslt'
            if hashlib.sha256(a.read_bytes()).digest()!=hashlib.sha256(b.read_bytes()).digest():
                raise ValueError(f'Transfer population mismatch: {base}')
            stage_data={}
            for stage in ('mountaincar_continuous',task):
                directory=base/stage
                params=load_parameters(directory/'parameters.yaml')
                with (directory/f'logs/selection/selection.{seed}.0.csv').open() as f:
                    selected={int(r['generation']):r for r in csv.DictReader(f)
                              if int(r['generation']) in GENERATIONS}
                stage_data[stage]=params,selected
            effective_count.cache_clear()
            for generation in GENERATIONS:
                stage='mountaincar_continuous' if generation<=2000 else task
                params,selected=stage_data[stage]
                cp=base/f'{stage}/checkpoints/cp.{generation}.{seed}.0.rslt'
                programs,teams,metrics=population(cp,params)
                champ=selected[generation];pid=teams[int(champ['team_id'])][0]
                total,full,action,_=programs[pid]
                if (total!=int(champ['program_instruction_count']) or
                    full!=int(champ['effective_program_instruction_count']) or
                    full-action!=int(champ['self_modification_only_instruction_count'])):
                    raise ValueError(f'Engine role counts disagree: {cp}')
                old=lookup.get((task,seed,generation))
                if old is not None and not (np.isclose(metrics['mean_structural_introns_fraction'],float(old['mean_intron_fraction'])) and
                        np.isclose(metrics['mean_structural_introns_count'],float(old['mean_introns']))):
                    raise ValueError(f'Population totals disagree with previous reconstruction: {cp}')
                records.append(dict(task=task,seed=seed,generation=generation,**metrics))
                audits.append(dict(task=task,seed=seed,generation=generation,
                                   champion_total=total,champion_effective=full,
                                   champion_mutation_only=full-action,engine_counts_match=True))
            print(f'{task} seed {seed}: {len(GENERATIONS)} full populations validated',flush=True)
        write_csv('population_roles_by_seed.csv',records)
        write_csv('validation.csv',audits)
    source_comparison=[]
    for seed in sorted(set(seedsets['pendulum']) & set(seedsets['acrobot'])):
        row={'seed':seed}
        for task in ('pendulum','acrobot'):
            source=ROOT/f'experiments/mountaincar_continuous_to_{task}_{VARIANT}/seed_{seed}/mountaincar_continuous'
            checkpoint=source/f'checkpoints/cp.2000.{seed}.0.rslt'
            row[f'{task}_checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            row[f'{task}_parameters_sha256']=hashlib.sha256((source/'parameters.yaml').read_bytes()).hexdigest()
            r=next(r for r in records if r['task']==task and r['seed']==seed and r['generation']==2000)
            for metric in ('mean_total','mean_structural_introns_fraction','mean_mutation_only_fraction'):
                row[f'{task}_{metric}']=r[metric]
        row['same_checkpoint']=row['pendulum_checkpoint_sha256']==row['acrobot_checkpoint_sha256']
        row['same_parameters']=row['pendulum_parameters_sha256']==row['acrobot_parameters_sha256']
        source_comparison.append(row)
    write_csv('source_pipeline_comparison.csv',source_comparison)
    render(records,audits,seedsets)


def render(records,audits,seedsets,cohort="population"):
    title_prefix="Runtime-modified" if cohort=="population" else "Runtime-modified · engine elites"
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#BBBBBB',
                         'xtick.color':'#555555','ytick.color':'#555555','pdf.fonttype':42})
    curves=(('structural_introns','Structural introns','#777777','-'),
            ('mutation_only','Mutation-only effective','#C17A45','-'),
            ('outside_action','Outside action slice','#3574A0','--'))
    aggregate=[]
    for unit in ('fraction','count'):
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True,layout='constrained')
        for ax,task in zip(axes,seedsets):
            for key,label,color,ls in curves:
                metric=f'mean_{key}_{unit}'
                vals=np.array([[next(r[metric] for r in records if r['task']==task and r['seed']==s and r['generation']==g)
                                for g in GENERATIONS] for s in seedsets[task]])
                lo,med,hi=np.quantile(vals,[.25,.5,.75],axis=0)
                ax.fill_between(GENERATIONS,lo,hi,color=color,alpha=.065,lw=0)
                ax.plot(GENERATIONS,med,color=color,ls=ls,lw=1.4,marker='.',ms=4,label=label)
                for g,l,m,h in zip(GENERATIONS,lo,med,hi):
                    aggregate.append(dict(task=task,metric=metric,generation=g,n_seeds=len(seedsets[task]),q1=l,median=m,q3=h))
            format_axis(ax,task,seedsets)
            if unit=='fraction':ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set_ylabel(('Population mean instruction fraction' if cohort=='population' else 'Elite mean instruction fraction') if unit=='fraction' else 'Mean instructions per policy')
        fig.suptitle(title_prefix+' · instruction roles',fontsize=12)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        save(fig,f'runtime_instruction_roles_{unit}')
    # Component MEANS across seeds preserve the additive 100% decomposition.
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True,layout='constrained')
    for ax,task in zip(axes,seedsets):
        parts=[]
        for key in ('action_effective','mutation_only','structural_introns'):
            parts.append([np.mean([r[f'mean_{key}_fraction'] for r in records if r['task']==task and r['generation']==g]) for g in GENERATIONS])
        if not np.allclose(np.sum(parts,axis=0),1):raise ValueError('Stack does not sum to one')
        ax.stackplot(GENERATIONS,*parts,colors=['#98B5C9','#D6A77E','#E3E3E3'],
                     labels=['Action effective (includes shared)','Mutation-only effective','Structural introns'],lw=0)
        format_axis(ax,task,seedsets)
        ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Mean population composition' if cohort=='population' else 'Mean elite composition')
    fig.suptitle(title_prefix+' · instruction composition',fontsize=12)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False,fontsize=8.5)
    save(fig,'runtime_instruction_composition')
    write_csv('roles_median_iqr.csv',aggregate)
    (OUT/'README.md').write_text('''# Runtime-modified population instruction roles

For each full program, compute A = engine-effective instructions rooted at action outputs (normally S0/S1), and E = engine-effective instructions rooted at action plus mutation outputs S2–S5. Both analyses retain runtime mode, so S6 decoy handling is identical. The mutation-register roots are removed only from the action analysis root set, not disabled at runtime.

Disjoint categories:
- Action effective: A. Includes instructions shared with mutation-rate dependencies.
- Mutation-only effective: E − A. Effective because of mutation outputs, but not needed by the action slice.
- Structural introns: total − E. These are the introns under the actual runtime mechanism.
- Outside action slice: total − A = structural introns + mutation-only effective. This is a counterfactual action-focused intron measure, not a claim that mutation-only instructions are useless or unexecuted.

Dependency analysis includes indirect dependencies, not just direct writes to S2–S5. Keeping runtime mode in both masks preserves the engine's special exclusion of S6 decoy accesses. These are structural dependency measurements, not causal ablations.

All 1,000 policies in each saved population receive equal weight. The fraction/count plots show medians of population means across seeds and interquartile bands. Componentwise medians are not necessarily additive, so the blue curve need not equal the sum of the two plotted median curves. The stacked composition instead uses means across seeds, so its components sum to 100%.

Uses every complete runtime seed, without requiring matching gene-encoded/static runs. See seed_coverage.csv for included seeds and missing files. Actual fresh Mountain Car source histories are taken from each target pipeline. These source populations can differ between pipelines even for matching seed IDs, so the Mountain Car portions need not coincide. source_pipeline_comparison.csv compares the source checkpoint hashes, saved parameter hashes, and population means for common seed IDs. The cause of divergence between separately rerun same-seed source populations is not diagnosed here. Generations are sampled every 500, plus 2100; lines connect checkpoints. The first post-transfer point includes 100 generations of transfer evolution.

Validation: every source/target generation-2000 checkpoint pair is byte-identical; at every sample, reconstructed champion total, effective, and mutation-only counts match all three corresponding engine log fields. Population structural intron means also match the previous full-population reconstruction. See validation.csv and population_roles_by_seed.csv.
'''+f'\nSeed counts: '+', '.join(f'{task}: {len(seeds)}' for task,seeds in seedsets.items())+f'. Validated {len(audits)} full populations, each with 1,000 policies.\n')
    print(f'Wrote runtime-only plots, data and validation to {OUT}',flush=True)


def format_axis(ax,task,seedsets):
    ax.axvline(2000,color='#999999',ls='--',lw=.8)
    ax.text(2000,1.015,'transfer',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#777777')
    ax.set_title(f'Mountain Car → {task.capitalize()}  ·  n={len(seedsets[task])}',pad=21,fontsize=11)
    ax.set_xlim(0,7000);ax.set_ylim(bottom=0)
    ax.set_xticks([0,2000,4000,6000],['0','2,000','4,000','6,000'])
    ax.set_xlabel('Generation');ax.grid(axis='y',color='#EEEEEE',lw=.6)


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=220,facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
