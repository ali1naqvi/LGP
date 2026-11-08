"""Test current training experiments for 20 episodes and plot seed means."""
import argparse
import csv
from pathlib import Path
import replay_test_table as replay

ROOT = replay.ROOT
replay.RESULT_DIR = ROOT / 'output/testing_5000'
GROUPS = (
 ('Mountain Car Continuous', 'mountaincar_continuous', 'mountaincar_continuous_n_eval_test'),
 ('Pendulum', 'pendulum', 'pendulum_n_eval_test'),
 ('Acrobot', 'acrobot', 'acrobot_n_eval_test'),
 ('Reacher', 'reacher', 'mj_n_eval_test'),
 ('Half Cheetah', 'halfcheetah', 'mj_n_eval_test'),
)
METHODS = ('Modified mutation rates', 'Fixed individual mutation rates', 'Fixed mutation rates')
VARIANTS = ('execution_modified_rates', 'fixed_individual', 'fixed_rates')

def prepare():
 rows = []
 for env, task, parameter in GROUPS:
  for method, variant in zip(METHODS, VARIANTS):
   experiment = f'{task}_{variant}'
   if task in ('reacher','halfcheetah'): experiment += '_vector'
   if experiment == 'pendulum_fixed_individual': experiment += '_one_point'
   directory = ROOT / 'experiments' / experiment
   count = 0
   for checkpoint in sorted((directory/'checkpoints').glob('cp.5000.*.0.rslt')):
    seed = int(checkpoint.name.split('.')[2])
    log = directory/'logs/selection'/f'selection.{seed}.0.csv'
    try: team, fitness = replay.selected_agent(log, 5000)
    except (ValueError, FileNotFoundError) as error:
     print(f'Skipping {experiment} seed {seed}: {error}'); continue
    rows.append(dict(environment=env, method=method, experiment=experiment, generation=5000,
      seed=seed, team_id=team, training_fitness=fitness,
      checkpoint=str(checkpoint.relative_to(ROOT)), config=str(replay.config_for(experiment).relative_to(ROOT)),
      test_eval_parameter=parameter))
    count += 1
   if not count: raise ValueError(f'No generation-5000 checkpoint/selection pairs for {experiment}')
   print(f'{experiment}: {count} seeds')
 replay.write_csv(replay.RESULT_DIR/'manifest.csv', replay.MANIFEST_FIELDS, rows)
 return rows

def plot(rows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import statistics
 scores = {}
 missing = []
 for row in rows:
  path = replay.job_path(row,20)
  if not path.exists(): missing.append(path); continue
  import json
  result=json.loads(path.read_text())
  if result['selection'] != row or result['episodes'] != 20: raise ValueError(f'Stale result: {path}')
  raw=result['test_scores']
  if len(raw)!=20: raise ValueError(f'Wrong episode count: {path}')
  values=([raw[0]]+[b-a for a,b in zip(raw,raw[1:])]
          if row['test_eval_parameter']!='mj_n_eval_test' else raw)
  scores.setdefault((row['environment'],row['method']),[]).append(statistics.mean(values))
 if missing: raise ValueError(f'{len(missing)} tests remain; run the run action first')
 plt.style.use('seaborn-v0_8-whitegrid')
 fig,axes=plt.subplots(1,5,figsize=(18,5),layout='constrained')
 colors=('tab:green','tab:blue','tab:orange')
 for i,(ax,(env,_,_)) in enumerate(zip(axes,GROUPS)):
  data=[scores[(env,m)] for m in METHODS]
  boxes=ax.boxplot(data,patch_artist=True,widths=.55,medianprops={'linewidth':2.2})
  for j,color in enumerate(colors):
   boxes['boxes'][j].set(facecolor=color,edgecolor=color,alpha=.26)
   boxes['medians'][j].set(color=color)
  ax.set_title(f'({chr(97+i)}) Testing - {env.replace(" Continuous","")}',fontsize=16)
  ax.set_xticks([1,2,3],['Modified\nrates','Fixed\nindividual','Fixed\nrates'])
  ax.set_xlabel('Mutation variant')
  if i==0: ax.set_ylabel('Mean test fitness (20 episodes)')
  ax.spines[['top','right']].set_visible(False)
  print(env, ', '.join(f'{m}: n={len(v)}' for m,v in zip(METHODS,data)))
 from matplotlib.patches import Patch
 fig.legend(handles=[Patch(color=c,label=m,alpha=.4) for c,m in zip(colors,METHODS)],loc='lower center',bbox_to_anchor=(0.5, -0.08),ncol=3,frameon=False)
 for ext in ('pdf','png'): fig.savefig(replay.RESULT_DIR/f'testing_boxplots.{ext}',dpi=180,bbox_inches='tight')
 print(replay.RESULT_DIR/'testing_boxplots.pdf')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('action',choices=['prepare','run','plot'])
 args=parser.parse_args()
 path=replay.RESULT_DIR/'manifest.csv'
 rows=prepare() if args.action=='prepare' else replay.read_manifest(path)
 if args.action=='run': replay.run_jobs(rows,20)
 if args.action=='plot': plot(rows)
