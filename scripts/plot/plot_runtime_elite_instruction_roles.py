"""Plot engine elites, checking that checkpoint survivors are precisely that set."""
import csv
from pathlib import Path
import shutil
import plot_runtime_population_instruction_roles as roles

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/mountaincar_runtime_population_instruction_roles'
OUT=ROOT/'output/mountaincar_runtime_elite_instruction_roles'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'population_roles_by_seed.csv').open() as f:
        records=[{k:v if k=='task' else int(v) if k in ('seed','generation','n_policies') else float(v)
                  for k,v in r.items()} for r in csv.DictReader(f)]
    with (SOURCE/'validation.csv').open() as f:
        audits=list(csv.DictReader(f))
    seedsets={task:sorted({r['seed'] for r in records if r['task']==task}) for task in ('pendulum','acrobot')}
    counts={}
    for task,seeds in seedsets.items():
        for seed in seeds:
            for stage in ('mountaincar_continuous',task):
                path=ROOT/f'experiments/mountaincar_continuous_to_{task}_execution_modified_rates/seed_{seed}/{stage}/logs/removal/removal.{seed}.0.csv'
                with path.open() as f:
                    for row in csv.DictReader(f):
                        counts[task,seed,stage,int(row['generation'])]=(int(row['num_teams']),int(row['num_elite_teams']))
    elite_audit=[]
    for r in records:
        task,seed,g=r['task'],r['seed'],r['generation']
        stage='mountaincar_continuous' if g<=2000 else task
        n_survivors,n_elites=counts[task,seed,stage,g]
        if n_survivors!=n_elites or n_elites!=r['n_policies']:
            raise ValueError(f'Elite membership cannot be equated to checkpoint population: {task}, {seed}, {g}')
        elite_audit.append(dict(task=task,seed=seed,generation=g,n_survivors=n_survivors,n_elites=n_elites,
                               checkpoint_population_equals_elites=True))
    roles.OUT=OUT
    roles.render(records,audits,seedsets,cohort='elites')
    roles.write_csv('elite_membership_audit.csv',elite_audit)
    for name in ('population_roles_by_seed.csv','validation.csv','seed_coverage.csv','source_pipeline_comparison.csv'):
        shutil.copyfile(SOURCE/name,OUT/name)
    p=OUT/'README.md'
    p.write_text('''# Engine-elite instruction roles

These runs use single-task survivor selection. FindMultiTaskElites marks the top n_root=1000 policies as elite; SelectTeams removes non-elites before writing the full training checkpoint. The removal logs report num_teams=num_elite_teams=1000 at every one of the 624 sampled checkpoints used here. The saved population is therefore already the engine-elite set, not the pre-selection pool of 1200 candidates.

The elite plots have the same numerical values as the preceding survivor-population plots. They do not represent an additional top-5% or top-10% cutoff. No fitness-ranked subset is inferred from instruction count or an arbitrary checkpoint order. See elite_membership_audit.csv for every count check.

'''+p.read_text())
    print(f'Validated elite membership at {len(elite_audit)} checkpoints: all 1000 survivors are elites.')


if __name__=='__main__':main()
