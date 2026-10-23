"""Standalone runtime instruction composition from available full populations.

No population training trajectory is inferred from champion-only selection logs.
"""
import csv
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from plot_runtime_population_instruction_roles import population,load_parameters,effective_count

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/standalone_runtime_instruction_composition'
ENVIRONMENTS=(('mountaincar_continuous','Mountain Car','mountaincar_continuous_execution_modified_rates'),
              ('pendulum','Pendulum','pendulum_execution_modified_rates'),
              ('acrobot','Acrobot','acrobot_execution_modified_rates'),
              ('reacher','Reacher','reacher_execution_modified_rates'),
              ('ant','Ant','ant_execution_modified_rates_all'))
CATEGORIES=('action_effective','mutation_only','structural_introns')
LABELS=('Action effective (includes shared)','Mutation-only effective','Structural introns')
COLORS=('#98B5C9','#D6A77E','#E3E3E3')


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records,audits,coverage=[],[],[]
    for environment,title,experiment in ENVIRONMENTS:
        directory=ROOT/'experiments'/experiment
        params=load_parameters(ROOT/'configs'/f'{experiment}.yaml')
        for seed in range(1,21):
            paths=sorted((directory/'checkpoints').glob(f'cp.*.{seed}.0.rslt'),key=lambda p:int(p.name.split('.')[1]))
            paths=[p for p in paths if int(p.name.split('.')[1])<=5000]
            if not paths:
                coverage.append(dict(environment=environment,seed=seed,available=False,latest_generation='',has_generation_5000=False))
                continue
            checkpoint=paths[-1];generation=int(checkpoint.name.split('.')[1])
            coverage.append(dict(environment=environment,seed=seed,available=True,latest_generation=generation,has_generation_5000=generation==5000))
            effective_count.cache_clear()
            programs,teams,metrics=population(checkpoint,params)
            with (directory/f'logs/selection/selection.{seed}.0.csv').open() as f:
                champion=next((r for r in csv.DictReader(f) if int(r['generation'])==generation),None)
            if champion is not None:
                pid=teams[int(champion['team_id'])][0]
                total,full,action,_=programs[pid]
                observed=(total,full,full-action)
                expected=tuple(int(champion[k]) for k in ('program_instruction_count','effective_program_instruction_count','self_modification_only_instruction_count'))
                if observed!=expected:raise ValueError(f'Engine role-count mismatch: {checkpoint}: {observed} vs {expected}')
                validation_status='selection_counts_match'
            else:
                total=full=action=''
                validation_status='no_selection_row_at_checkpoint_generation'
            removal=directory/f'logs/removal/removal.{seed}.0.csv'
            row=None
            if removal.exists():
                with removal.open() as f:
                    row=next((r for r in csv.DictReader(f) if int(r['generation'])==generation),None)
            if row is not None:
                elite_count=int(row['num_elite_teams'])
                if not (int(row['num_teams'])==elite_count==len(teams)):
                    raise ValueError(f'Survivor/elite membership mismatch: {checkpoint}')
                membership_basis='removal_log'
            else:
                if len(teams)!=params['n_root'] or params.get('map_elites',0)!=0:
                    raise ValueError(f'Unexpected saved survivor population: {checkpoint}')
                elite_count=len(teams)
                membership_basis='checkpoint_size_and_single_task_selection_code'
            records.append(dict(environment=environment,title=title,experiment=experiment,seed=seed,
                                generation=generation,**metrics))
            audits.append(dict(environment=environment,seed=seed,generation=generation,n_elites=elite_count,
                               elite_membership_basis=membership_basis,champion_total=total,champion_effective=full,champion_mutation_only=full-action if champion is not None else '',
                               engine_counts_match=champion is not None,validation_status=validation_status))
        print(f'{title}: reconstructed {sum(r["environment"]==environment for r in records)} full populations',flush=True)
        write_csv('latest_population_by_seed.csv',records)
        write_csv('validation.csv',audits)
    write_csv('seed_coverage.csv',coverage)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.bottom':False,'axes.edgecolor':'#BBBBBB','pdf.fonttype':42})
    summaries=[]
    for mode in ('generation_5000','latest_available'):
        selected=[r for r in records if mode=='latest_available' or r['generation']==5000]
        fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
        names=[]
        for i,(environment,title,_) in enumerate(ENVIRONMENTS):
            group=[r for r in selected if r['environment']==environment]
            means=np.array([np.mean([r[f'mean_{key}_fraction'] for r in group]) for key in CATEGORIES])
            if not np.isclose(means.sum(),1):raise ValueError('Composition does not sum to one')
            minimum=min(r['generation'] for r in group);maximum=max(r['generation'] for r in group)
            gen=f'g={minimum:,}' if minimum==maximum else f'g={minimum:,}–{maximum:,}'
            names.append(f'{title}  ·  n={len(group)}'+(f'  ·  {gen}' if mode=='latest_available' else ''))
            left=0.
            for j,(value,color,label) in enumerate(zip(means,COLORS,LABELS)):
                ax.barh(i,value,left=left,height=.54,color=color,label=label if i==0 else None)
                left+=value
            summaries.append(dict(cohort=mode,environment=environment,n_seeds=len(group),min_generation=minimum,max_generation=maximum,
                                  **{key+'_fraction':float(value) for key,value in zip(CATEGORIES,means)}))
            if mode=='generation_5000':single(environment,title,group,means)
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Mean elite instruction composition')
        ax.set_title('Standalone runtime-modified · '+('generation 5,000' if mode=='generation_5000' else 'latest saved checkpoint per seed'),fontsize=12,pad=15)
        ax.grid(axis='x',color='white',lw=.7);ax.set_axisbelow(False)
        fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=8.5)
        save(fig,f'all_environments_{mode}')
    write_csv('composition_summary.csv',summaries)
    lines=['# Standalone runtime-modified instruction composition','',
           'The generation_5000 overview compares full saved populations at exactly generation 5000. Each policy receives equal weight within its seed, then seed population means are averaged. The latest_available overview additionally includes seeds whose last local checkpoint is earlier; its generation ranges are explicitly labeled. Do not interpret that second overview as a common-budget comparison.','',
           'Action effective includes dependencies shared with mutation registers. Mutation-only effective = full effective − action-effective. Structural introns = total − full effective. These are structural masks, not causal ablations or execution-frequency measurements. S6 decoy treatment is identical in both masks.','',
           'Where a selection row exists at the checkpoint generation, reconstructed champion total, effective and mutation-only counts are cross-checked against it. Some Pendulum selection logs end before their saved checkpoints; those populations are reconstructed from the full checkpoints and explicitly marked as lacking a matching selection row in validation.csv. The entire saved population is included. Saved checkpoints contain 1000 unique single-program policies, matching configured n_root=1000. Engine-elite interpretation follows the single-task survivor-selection code; standalone removal logs are absent, so this is not a direct check of serialized elite flags. Population means themselves use every saved policy.','',
           'Standalone full-population checkpoints are mostly final-only. A training-time population composition curve cannot be reconstructed from best-policy selection logs. No such curve is fabricated here.','',
           'Ant uses ant_execution_modified_rates_all. Pendulum uses pendulum_execution_modified_rates (the standard run), not the distinct no_limit experiment. Half-Cheetah standalone data are not present locally.','',
           '| Environment | Seeds at g=5000 | Latest available seeds | Latest generation range |','|---|---:|---:|---|']
    for environment,title,_ in ENVIRONMENTS:
        group=[r for r in records if r['environment']==environment]
        lines.append(f'| {title} | {sum(r["generation"]==5000 for r in group)} | {len(group)} | {min(r["generation"] for r in group)}–{max(r["generation"] for r in group)} |')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(OUT/'standalone_composition_plots.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.suffix in ('.png','.pdf','.csv','.md'):z.write(p,p.name)
    print(f'Wrote {len(records)} reconstructed populations and plots to {OUT}',flush=True)


def single(environment,title,group,means):
    fig,ax=plt.subplots(figsize=(7.5,2.5),layout='constrained')
    left=0.
    for value,color,label in zip(means,COLORS,LABELS):
        ax.barh(0,value,left=left,height=.48,color=color,label=label);left+=value
    ax.set_ylim(-.55,.55);ax.set_yticks([]);ax.set_xlim(0,1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f'{title} · runtime-modified · generation 5,000 · n={len(group)}',fontsize=11,pad=12)
    ax.set_xlabel('Mean elite instruction composition')
    fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    save(fig,f'{environment}_composition')


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=220,facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
