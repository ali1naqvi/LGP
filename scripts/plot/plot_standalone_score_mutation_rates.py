"""Full standalone training scores and elite mutation rates, with coverage audit."""
import csv,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/standalone_score_mutation_rates'
RATES=('swap','delete','add','mutate','decoy')
VARIANTS=('execution_modified_rates','fixed_individual','fixed_rates')
LABELS=('Execution-modified rates','Fixed individual rates','Fixed rates')

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 audit=[]
 for env,title,threshold in [('mountaincar_continuous','Mountain Car',90),('pendulum','Pendulum',-200),('acrobot','Acrobot',-100)]:
  fig,axes=plt.subplots(6,1,figsize=(11,15),sharex=True,layout='constrained')
  summary=[]
  for variant,label,color in zip(VARIANTS,LABELS,('tab:green','tab:blue','tab:orange')):
   name=f'{env}_{variant}'+('_one_point' if env=='pendulum' and variant=='fixed_individual' else '')
   static={}
   if variant=="fixed_rates":
    config=(ROOT/"configs"/f"{name}.yaml").read_text()
    static={rate:float(re.search(rf"^\s*p_instructions_{rate}:\s*([0-9.eE+-]+)",config,re.M).group(1)) for rate in RATES[:4]}
   runs=[]
   for path in sorted((ROOT/'experiments'/name/'logs/selection').glob('selection.*.0.csv')):
    with path.open() as f: rows={int(r['generation']):r for r in csv.DictReader(f) if int(r['generation'])<=5000}
    if not rows: continue
    runs.append(rows)
    validation=[g for g,r in rows.items() if float(r['validation_fitness'])!=0]
    audit.append(dict(experiment=name,file=str(path),first=min(rows),last=max(rows),validation_nonzero=len(validation),validation_first=validation[:10]))
   for index,rate in enumerate((None,)+RATES):
    if rate=='decoy' and variant!='execution_modified_rates': continue
    for stage in (('score',) if rate is None else ('start','output') if variant=='execution_modified_rates' else ('start',)):
     values=np.full((len(runs),5001),np.nan)
     for i,run in enumerate(runs):
      for g,r in run.items():
       values[i,g]=(1+(float(r['best_fitness'])-threshold)/abs(threshold) if rate is None else static[rate] if static else float(r[f'elite_avg_{stage}_rate_{rate}']))
     count=np.isfinite(values).sum(axis=0); valid=count>0; x=np.arange(5001)[valid]
     median=np.nanmedian(values[:,valid],axis=0); q1,q3=np.nanquantile(values[:,valid],[.25,.75],axis=0)
     axes[index].plot(x,median,color=color,ls='--' if stage=='start' and variant=='execution_modified_rates' else '-',lw=1.2)
     axes[index].fill_between(x,q1,q3,color=color,alpha=.12)
     summary.extend(dict(variant=variant,metric=rate or 'normalized_score',stage=stage,generation=int(g),n=int(n),median=float(m),q25=float(a),q75=float(b)) for g,n,m,a,b in zip(x,count[valid],median,q1,q3))
   print(name,len(runs),'logs',flush=True)
  axes[0].set_ylabel('Normalized training score');axes[0].axhline(1,color='black',ls=':',lw=1)
  axes[0].legend(handles=[Line2D([],[],color=c,label=l) for c,l in zip(('tab:green','tab:blue','tab:orange'),LABELS)],fontsize=9)
  axes[1].legend(handles=[Line2D([],[],color='black',ls='--',label='Inherited (modified variant)'),Line2D([],[],color='black',label='Execution output / fixed variant')],fontsize=9)
  for ax,rate in zip(axes[1:],RATES): ax.set_ylabel(rate.capitalize()+' rate');ax.set_ylim(0,1)
  for ax in axes: ax.set_xlim(0,5000);ax.grid(alpha=.2)
  axes[-1].set_xlabel('Generation')
  fig.suptitle(title+' — standalone training\nMedian and interquartile range across available runs')
  for ext in ('png','pdf'):fig.savefig(OUT/f'{env}.{ext}',dpi=160)
  plt.close(fig)
  with (OUT/f'{env}_summary.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
