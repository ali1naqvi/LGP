#!/usr/bin/env python3
"""Direct and indirect observations in register writes contributing to rate outputs."""
import csv
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from plot_dependency_matrices import ROOT, CASES, OBSERVATIONS


def analyze(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    active = set()
    def walk(key):
        if key in active:
            return
        active.add(key)
        for parent in nodes[key]['parents']:
            walk(parent)
    for key in graph['output_nodes'].values():
        walk(key)
    observations, upstream, incoming = {}, {}, {}
    for n in graph['nodes']:
        key = n['id']
        observations[key] = {n['observation']} if n['kind'] == 'observation' else set()
        upstream[key] = set()
        incoming[key] = {n['register']} if n['kind'] == 'initial' else set()
        for parent in n['parents']:
            observations[key] |= observations[parent]
            incoming[key] |= incoming[parent]
            if observations[parent]:
                upstream[key] |= upstream[parent]
                if nodes[parent]['kind'] == 'instruction':
                    upstream[key].add(nodes[parent]['register'])
    registers = {'S2', 'S3', 'S4', 'S5', 'S6'} | {
        nodes[k]['register'] for k in active if nodes[k]['kind'] == 'instruction'}
    records = []
    for register in sorted(registers, key=lambda r: ('SVM'.index(r[0]), int(r[1:]))):
        writes = [nodes[k] for k in active if nodes[k]['kind'] == 'instruction' and nodes[k]['register'] == register]
        if register in graph['output_nodes']:
            final = nodes[graph['output_nodes'][register]]
            writes = [final] if final['kind'] == 'instruction' else []
        direct, total, via, prior = set(), set(), set(), set()
        for n in writes:
            direct |= {nodes[p]['observation'] for p in n['parents'] if nodes[p]['kind'] == 'observation'}
            total |= observations[n['id']]
            via |= upstream[n['id']]
            prior |= incoming[n['id']]
        if not writes and register in graph['output_nodes']:
            prior |= incoming[graph['output_nodes'][register]]
        via.discard(register)
        records.append(dict(register=register, direct=sorted(direct), indirect_only=sorted(total-direct),
                            total=sorted(total), via=sorted(via, key=lambda r: ('SVM'.index(r[0]), int(r[1:]))),
                            prior=sorted(prior), contributing_instructions=sorted({n['index'] for n in writes})))
    return records


def detailed_main():
    base = ROOT / 'output/structural_dependencies_all_seeds'
    out = base / 'register_observation_links'
    out.mkdir(parents=True, exist_ok=True)
    with (base/'summary_mean_and_best.csv').open() as f:
        summary = {(r['environment'], r['output']): r for r in csv.DictReader(f)}
    roles = {'S2':'swap', 'S3':'delete', 'S4':'add', 'S5':'point', 'S6':'decoy'}
    plt.rcParams.update({'text.color':'#202020', 'font.family':'DejaVu Serif',
                         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    exported = []
    with PdfPages(out/'all_best_register_observation_links.pdf') as pdf:
        for env, name, generation in CASES:
            seed = summary[env,'S2']['best_seed']
            data = json.loads((base/'per_seed'/f'{env}.g{generation}.seed{seed}.json').read_text())
            graph, = data['graphs']
            records = analyze(graph)
            count = graph['observation_dimension']
            width = max(13, 7 + count*.60)
            height = 2.2 + len(records)*.43
            fig, ax = plt.subplots(figsize=(width,height), facecolor='white')
            fig.subplots_adjust(left=.03,right=.99,top=1-.75/height,bottom=1.0/height)
            ax.set_facecolor('white'); ax.axis('off')
            ax.set_xlim(0, width);ax.set_ylim(len(records)+.8,-1.3)
            start=2.7; spacing=(width-8)/max(count-1,1)
            xs=[start+i*spacing for i in range(count)]
            via_x=width-4.8;prior_x=width-2.2
            fig.text(.03,1-.16/height,f'{name}: generation {generation}, seed {seed}',fontsize=13,va='top')
            for i,x in enumerate(xs):
                ax.plot([x,x],[-.6,len(records)-.5],color='#E5E5E5',lw=.6,zorder=0)
                ax.text(x,-.9,f'$o_{{{i}}}$',ha='center',va='center',fontsize=11)
            ax.text(.1,-.9,'Register',fontsize=10)
            ax.text(via_x,-.9,'Mediating registers',fontsize=10)
            ax.text(prior_x,-.9,'Prior state',fontsize=10)
            for y,r in enumerate(records):
                reg=r['register']; connected=bool(r['total'])
                ax.plot([2.2,via_x-.3],[y,y],color='#E5E5E5',lw=.6,zorder=0)
                ax.text(.1,y,reg + (' ('+roles[reg]+')' if reg in roles else ''),ha='left',va='center',fontsize=10,
                        weight='bold' if reg in roles else 'normal')
                for obs in r['direct']:
                    ax.scatter(xs[obs],y,s=75,color='#222222',zorder=3)
                for obs in r['indirect_only']:
                    ax.scatter(xs[obs],y,s=75,facecolors='white',edgecolors='#222222',linewidths=1.3,zorder=3)
                ax.text(via_x,y,'\n'.join(textwrap.wrap(', '.join(r['via']) or 'none',width=24)),va='center',fontsize=9)
                ax.text(prior_x,y,', '.join(r['prior']) or 'none',va='center',fontsize=9)
            key=';  '.join(f'o{i} = {label}' for i,label in enumerate(OBSERVATIONS[env]))
            fig.text(.03,.80/height,'● Direct observation input     ○ Indirect dependency only     Blank: no observation dependency',fontsize=9)
            fig.text(.03,.57/height,'S2–S6: final values. Other registers: contributing writes. Mediators are unordered; same-register earlier writes are omitted from that list.',fontsize=8)
            fig.text(.03,.36/height,'\n'.join(textwrap.wrap(key,width=int(width*14))),fontsize=8,va='top')
            prefix=out/env
            for ext in ('png','svg','pdf'):
                fig.savefig(str(prefix)+'.'+ext,dpi=300,facecolor=fig.get_facecolor())
            pdf.savefig(fig,facecolor=fig.get_facecolor());plt.close(fig)
            exported.append(dict(environment=env,generation=generation,seed=int(seed),registers=records))
    (out/'dependencies.json').write_text(json.dumps(exported,indent=2)+'\n')
    print(f'Created five charts in {out}')


def main():
    """Single-page figure, using the same style as training/testing plots."""
    base = ROOT / 'output/structural_dependencies_all_seeds'
    out = base / 'register_observation_links'
    out.mkdir(parents=True, exist_ok=True)
    with (base/'summary_mean_and_best.csv').open() as handle:
        summary = {(r['environment'], r['output']): r for r in csv.DictReader(handle)}
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
                         'text.color':'#262626', 'pdf.fonttype':42, 'ps.fonttype':42})
    fig = plt.figure(figsize=(14, 7.5), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.065, right=.98, top=.94, bottom=.15,
                           hspace=.60, wspace=.16, height_ratios=[1,1,1])
    specs = [(CASES[0][0], 'Mountain Car Continuous', CASES[0][2]),
             CASES[1], CASES[4], CASES[3], CASES[2]]
    exported = []
    roles = [('S2','Swap'), ('S3','Delete'), ('S4','Add'), ('S5','One-point')]
    role_registers = [register for register, _ in roles]
    for panel, (env, name, generation) in enumerate(specs):
        ax = fig.add_subplot(grid[panel//2, panel%2] if panel < 4 else grid[2,:])
        seed = summary[env,'S2']['best_seed']
        data = json.loads((base/'per_seed'/f'{env}.g{generation}.seed{seed}.json').read_text())
        graph, = data['graphs']
        records = [r for r in analyze(graph) if r['register'] in role_registers]
        count = graph['observation_dimension']
        ax.set_xlim(-1.5, count+.7)
        ax.set_ylim(3.5,-1.15)
        ax.axis('off')
        ax.set_title(f'({chr(97+panel)}) Best agent of {name}',
                     fontsize=10, loc='left', pad=8)
        for obs in range(count):
            ax.text(obs,-.8,f'$o_{{{obs}}}$',ha='center',fontsize=9)
            ax.plot([obs,obs],[-.45,3.4],color='#E3E3E3',lw=.6,zorder=0)
        for y, (row, (register, operation)) in enumerate(zip(records, roles)):
            ax.text(-.15,y,operation,ha='right',va='center',fontsize=9)
            ax.plot([0,count-.6],[y,y],color='#E3E3E3',lw=.6,zorder=0)
            if row['direct']:
                ax.scatter(row['direct'],[y]*len(row['direct']),s=34,c='#262626',zorder=3)
            if row['indirect_only']:
                ax.scatter(row['indirect_only'],[y]*len(row['indirect_only']),s=34,
                           facecolors='white',edgecolors='#262626',lw=1.1,zorder=3)
        exported.append(dict(environment=env,generation=generation,seed=int(seed),registers=records))
    fig.text(.055,.068,'● Direct observation input     ○ Indirect dependency only     Blank cells indicate no observation dependency.',fontsize=9)
    fig.text(.055,.041,'Dependencies of final mutation-rate values in one execution. Generations: Mountain Car 2,000; Pendulum 1,000; Acrobot 5,000; Reacher 3,000; Half-Cheetah 4,000.',fontsize=8)
    prefix = out/'best_rate_observation_dependencies'
    for ext in ('png','pdf','svg'):
        fig.savefig(str(prefix)+'.'+ext,dpi=300,facecolor='white')
    plt.close(fig)
    Path(str(prefix)+'.json').write_text(json.dumps(exported,indent=2)+'\n')
    caption = ('Structural observation dependencies of the best-fitness individuals at the indicated generations '
               '(execution_modified_rates). Filled circles denote observation operands read directly by the final write; '
               'open circles denote dependence through earlier register computations. Both direct and indirect routes may '
               'exist for a filled-circle cell. Blank cells indicate no observation dependency. This is structural dependence '
               'without numerical simplification.\n\n')
    for env,name,_ in specs:
        caption += name+': '+ '; '.join(f'o{i} = {label}' for i,label in enumerate(OBSERVATIONS[env]))+'.\n'
    Path(str(prefix)+'.caption.txt').write_text(caption)
    print(f'Created single-page figure: {prefix}.pdf')


if __name__ == '__main__':
    main()
