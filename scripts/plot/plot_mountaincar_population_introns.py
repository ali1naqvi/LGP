"""Population introns reconstructed from full, matched transfer checkpoints."""
import csv
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from extract_best_agent_instructions import load_instruction_definitions
from extract_generation_agent_instructions import output_registers, load_parameters
from structural_dependency_graph import engine_effective_indices

OUT = ROOT/'output/mountaincar_population_introns'
VARIANTS = ('execution_modified_rates','fixed_individual','fixed_rates')
LABELS = ('Runtime-modified','Gene-encoded','Static')
COLORS = ('#3574A0','#C17A45','#686868')
GENERATIONS = sorted(set(range(0,7001,500)) | {2100})
INS = re.compile(r'(?:\d+_){8}')
DEFINITIONS = load_instruction_definitions()


@lru_cache(maxsize=50000)
def effective_count(instructions,slots,outputs,self_modifying):
    return len(engine_effective_indices(instructions,slots,DEFINITIONS,set(outputs),self_modifying))


def parse(path,parameters):
    programs,teams = {},{}
    with path.open() as stream:
        for line in stream:
            if line.startswith('RegisterMachine:'):
                parts=line.strip().split(':')
                pid=int(parts[1]); slots=int(parts[10])
                instructions=tuple(p for p in parts[11:] if INS.fullmatch(p))
                self_modifying=bool(parameters.get('self_modifying',0))
                outputs=tuple(sorted(output_registers(parameters,slots,self_modifying)))
                total=len(instructions)
                effective=effective_count(instructions,slots,outputs,self_modifying)
                programs[pid]=(total,effective,int(parts[3]))
            elif line.startswith('team:'):
                parts=line.strip().split(':')
                teams[int(parts[1])]=[int(p) for p in parts[7:]]
    # These experiments are single-program atomic policies; enforce this instead
    # of silently weighting a hierarchical/shared population incorrectly.
    members=[p for group in teams.values() for p in group]
    if (not teams or any(len(group)!=1 for group in teams.values()) or
        len(members)!=len(set(members)) or set(members)!=set(programs) or
        any(p[2]>=0 for p in programs.values())):
        raise ValueError(f'Expected one unique atomic program per policy: {path}')
    values=np.array([programs[p][:2] for p in members],dtype=float)
    total,effective=values.T
    if not ((total>0)&(effective>=0)&(effective<=total)).all():
        raise ValueError(f'Invalid counts: {path}')
    return programs,teams,dict(n_policies=len(teams),mean_total=float(total.mean()),
                              mean_effective=float(effective.mean()),
                              mean_introns=float((total-effective).mean()),
                              mean_intron_fraction=float(((total-effective)/total).mean()),
                              pooled_intron_fraction=float((total-effective).sum()/total.sum()))


def log_rows(path):
    with path.open() as f:
        return {int(r['generation']):r for r in csv.DictReader(f) if int(r['generation']) in GENERATIONS}


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records,audits,seedsets=[],[],{}
    for task in ('pendulum','acrobot'):
        seeds=[s for s in range(1,21) if all(
            (ROOT/f'experiments/mountaincar_continuous_to_{task}_{v}/seed_{s}/{stage}/logs/selection/selection.{s}.0.csv').exists()
            for v in VARIANTS for stage in ('mountaincar_continuous',task))]
        seedsets[task]=seeds
        for v in VARIANTS:
            for seed in seeds:
                base=ROOT/f'experiments/mountaincar_continuous_to_{task}_{v}/seed_{seed}'
                manifest=json.loads((base/'transfer.json').read_text())
                if manifest['source_mode']!='fresh_mountaincar_continuous':
                    raise ValueError(f'Unexpected source mode {base}')
                a=base/f'mountaincar_continuous/checkpoints/cp.2000.{seed}.0.rslt'
                b=base/f'{task}/checkpoints/cp.2000.{seed}.0.rslt'
                digest_a=hashlib.sha256(a.read_bytes()).hexdigest()
                digest_b=hashlib.sha256(b.read_bytes()).hexdigest()
                if digest_a!=digest_b:
                    raise ValueError(f'Transferred checkpoint differs from actual source: {base}')
                audit=dict(task=task,variant=v,seed=seed,checkpoint_2000_identical=True,
                           source_sha256=digest_a,validated_champions=0)
                stage_data={}
                for stage in ('mountaincar_continuous',task):
                    directory=base/stage
                    stage_data[stage]=(load_parameters(directory/'parameters.yaml'),
                                       log_rows(directory/f'logs/selection/selection.{seed}.0.csv'))
                effective_count.cache_clear()
                for generation in GENERATIONS:
                    stage='mountaincar_continuous' if generation<=2000 else task
                    cp=base/f'{stage}/checkpoints/cp.{generation}.{seed}.0.rslt'
                    if not cp.exists():
                        raise ValueError(f'Missing exact checkpoint: {cp}')
                    params,selection=stage_data[stage]
                    programs,teams,metrics=parse(cp,params)
                    champion=selection[generation]
                    pid=teams[int(champion['team_id'])][0]
                    total,effective,_=programs[pid]
                    if (total!=int(champion['program_instruction_count']) or
                        effective!=int(champion['effective_program_instruction_count'])):
                        raise ValueError(f'Reconstructed champion disagrees with engine log: {cp}: {total}/{effective}')
                    audit['validated_champions']+=1
                    records.append(dict(task=task,variant=v,seed=seed,generation=generation,**metrics))
                audits.append(audit)
            print(f'{task} {v}: {len(seeds)} seeds × {len(GENERATIONS)} population checkpoints',flush=True)
    write_csv('population_by_seed.csv',records)
    write_csv('transfer_audit.csv',audits)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#BBBBBB',
                         'xtick.color':'#555555','ytick.color':'#555555','pdf.fonttype':42})
    aggregates=[]
    for metric,name,ylabel in (
        ('mean_intron_fraction','population_intron_fraction','Population mean intron fraction'),
        ('mean_introns','population_intron_count','Mean introns per policy'),
        ('mean_total','population_program_size','Mean instructions per policy')):
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True,layout='constrained')
        for ax,task in zip(axes,('pendulum','acrobot')):
            for v,label,color in zip(VARIANTS,LABELS,COLORS):
                lookup={(r['seed'],r['generation']):r[metric] for r in records if r['task']==task and r['variant']==v}
                values=np.array([[lookup[s,g] for g in GENERATIONS] for s in seedsets[task]])
                q1,median,q3=np.quantile(values,[.25,.5,.75],axis=0)
                ax.fill_between(GENERATIONS,q1,q3,color=color,alpha=.08,lw=0)
                ax.plot(GENERATIONS,median,color=color,lw=1.4,marker='.',ms=4,label=label)
                for g,lo,med,hi in zip(GENERATIONS,q1,median,q3):
                    aggregates.append(dict(task=task,variant=v,metric=metric,generation=g,
                                           n_seeds=len(seedsets[task]),q1=lo,median=med,q3=hi))
            ax.axvline(2000,color='#999999',ls='--',lw=.8)
            ax.text(2000,1.015,'transfer',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#777777')
            ax.set_title(f'Mountain Car → {task.capitalize()}  ·  n={len(seedsets[task])}',pad=22,fontsize=11)
            ax.set_xlabel('Generation');ax.set_xlim(0,7000);ax.set_ylim(bottom=0)
            ax.set_xticks([0,2000,4000,6000],['0','2,000','4,000','6,000'])
            if metric=='mean_intron_fraction':
                ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y',color='#EEEEEE',lw=.6)
        axes[0].set_ylabel(ylabel)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
        fig.savefig(OUT/f'{name}.png',dpi=220,facecolor='white')
        fig.savefig(OUT/f'{name}.pdf',facecolor='white')
        plt.close(fig)
    write_csv('population_median_iqr.csv',aggregates)
    (OUT/'README.md').write_text('''# Population introns in Mountain Car transfer

Source histories are the actual fresh Mountain Car runs inside each transfer experiment, identified by transfer.json. Every source generation-2000 checkpoint is SHA256-identical to the checkpoint copied into its target directory; see transfer_audit.csv.

Full populations are reconstructed at generations 0, 500, 1000, 1500, 2000, 2100, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, and 7000. Lines connect sampled checkpoints; they do not imply measurements at intermediate generations. The first plotted post-transfer checkpoint is generation 2100; changes between 2000 and 2100 include 100 generations of transfer evolution.

Checkpoint programs retain all instructions (TPG::WriteCheckpoint(false)). We reuse structural_dependency_graph.engine_effective_indices, which mirrors the engine’s MarkIntrons. At EVERY sampled checkpoint, both total and effective counts for the logged champion are checked against the selection log; any mismatch aborts. One atomic program per policy, unique program membership, and complete population membership are enforced.

Introns = total − effective. Effective instructions include self-modification dependencies for runtime-modified programs. Within each seed, intron fraction is averaged across all policies with equal policy weight. Count and size are also population means. Plotted lines are the median of those seed means; light bands are their interquartile range. No champion is used to summarize the population. Long-program-weighted pooled fractions are additionally saved in population_by_seed.csv, but are not the plotted fraction.

Seed matching uses all three mechanisms within each task, consistently over the full history. Missing runs are excluded, not imputed. These are descriptive measurements of structural dependencies, not executed-frequency measurements or experimentally proven neutrality.

The earlier champion figures incorrectly joined standalone Mountain Car runs onto fresh transfer runs. Those figures have been regenerated using the correct source histories. Different champions can still be selected after switching tasks even when the full population is transferred intact.

'''+''.join(f'- {task} matched seeds: {seeds}\n' for task,seeds in seedsets.items())+
        f'\nVerified {len(audits)} source/target checkpoint identities and {sum(a["validated_champions"] for a in audits)} logged champion count pairs.\n')
    print(f'Wrote population plots, data and audits to {OUT}',flush=True)


if __name__=='__main__':
    main()
