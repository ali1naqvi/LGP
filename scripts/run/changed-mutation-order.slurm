#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=changed-mutation-order
#SBATCH --array=1-240%4
#SBATCH --ntasks=21
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-23:00:00
#SBATCH --output=changed-mutation-order-%A_%a.out
#SBATCH --error=changed-mutation-order-%A_%a.err

set -euo pipefail
# Submit from the repo root after rebuilding TPGExperimentMPI.
repo=$(cd "${TPG:-${SLURM_SUBMIT_DIR:?Submit using sbatch}}" && pwd)
task=${SLURM_ARRAY_TASK_ID:?Submit as an array}
environments=(acrobot pendulum mountaincar_continuous)
rotation_count=3
environment_count=${#environments[@]}
rotation_runs_per_seed=$(( environment_count * rotation_count ))
rotation_runs=$(( 20 * rotation_runs_per_seed ))
if (( $# != 0 || task < 1 || task > rotation_runs + 20 * environment_count )); then
    echo "Expected an array task from 1 through 240 and no script arguments." >&2
    exit 2
fi

if (( task <= rotation_runs )); then
    seed=$(( (task - 1) / rotation_runs_per_seed + 1 ))
    run_index=$(( (task - 1) % rotation_runs_per_seed ))
    environment=${environments[$(( run_index / rotation_count ))]}
    rotation=$(( run_index % rotation_count + 1 ))
    variant="rotation_$rotation"
    setting="self_modifying_register_rotation: $rotation"
else
    run_index=$(( task - rotation_runs - 1 ))
    seed=$(( run_index / environment_count + 1 ))
    environment=${environments[$(( run_index % environment_count ))]}
    variant=randomized_per_individual
    setting="randomize_self_modifying_register_order: 1"
fi
binary="$repo/build/release/experiments/TPGExperimentMPI"
base_config="$repo/configs/${environment}_execution_modified_rates.yaml"
[[ -x $binary && -r $base_config ]] || {
    echo "Expected a rebuilt executable at $binary and a readable config at $base_config." >&2
    exit 1
}

output_dir="$repo/experiments/changed_mutation_order/$environment/$variant"
mkdir -p "$output_dir"
run_dir="$output_dir/seed_$seed"
mkdir "$run_dir" || { echo "Run directory already exists: $run_dir" >&2; exit 1; }
cd "$run_dir"
mkdir -p logs checkpoints
cp "$base_config" parameters.yaml
# Save the assignment with the config so replay uses the same mapping.
printf '\nself_modifying_parameters:\n  %s\n' "$setting" >> parameters.yaml
export OMP_NUM_THREADS=1
srun "$binary" parameters_file="$PWD/parameters.yaml" seed_tpg="$seed" \
    > "logs/tpg.$seed.${SLURM_JOB_ID}.std" 2> "logs/tpg.$seed.${SLURM_JOB_ID}.err"
