#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=classic-control-evolved-constants
#SBATCH --array=1-180%4
#SBATCH --ntasks=21
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-23:00:00
#SBATCH --output=classic-control-evolved-constants-%A_%a.out
#SBATCH --error=classic-control-evolved-constants-%A_%a.err

set -euo pipefail
# Submit from the repo root after rebuilding TPGExperimentMPI.
# Usage: sbatch scripts/run/classic-control-evolved-constants.slurm
repo=$(cd "${TPG:-${SLURM_SUBMIT_DIR:?Submit using sbatch}}" && pwd)
task=${SLURM_ARRAY_TASK_ID:?Submit as an array}

environments=(acrobot pendulum mountaincar_continuous)
conditions=(initial_0.2 initial_0.4 initial_random)
condition_count=${#conditions[@]}
runs_per_seed=$(( ${#environments[@]} * condition_count ))
if (( $# != 0 || task < 1 || task > 20 * runs_per_seed )); then
    echo "Expected an array task from 1 through 180 and no script arguments." >&2
    exit 2
fi

seed=$(( (task - 1) / runs_per_seed + 1 ))
run_index=$(( (task - 1) % runs_per_seed ))
environment=${environments[$(( run_index / condition_count ))]}
condition=${conditions[$(( run_index % condition_count ))]}
case "$condition" in
    initial_0.2|initial_0.4)
        rate=${condition#initial_}
        overrides=(p_instructions_swap="$rate" p_instructions_delete="$rate"
                   p_instructions_add="$rate" p_instructions_mutate="$rate")
        ;;
    initial_random)
        overrides=(randomize_self_modifying_constants=1)
        ;;
esac

binary="$repo/build/release/experiments/TPGExperimentMPI"
base_config="$repo/configs/${environment}_execution_modified_rates.yaml"
[[ -x $binary && -r $base_config ]] || {
    echo "Expected a rebuilt executable at $binary and a readable config at $base_config." >&2
    exit 1
}
output_dir="$repo/experiments/classic_control_evolved_constants/$environment/$condition"
mkdir -p "$output_dir"
run_dir="$output_dir/seed_$seed"
mkdir "$run_dir" || { echo "Run directory already exists: $run_dir" >&2; exit 1; }
cd "$run_dir"
mkdir -p logs checkpoints
cp "$base_config" parameters.yaml
for override in "${overrides[@]}"; do
    key=${override%%=*}
    value=${override#*=}
    if [[ $key == randomize_self_modifying_constants ]]; then
        printf '\nself_modifying_initialization:\n  %s: %s\n' "$key" "$value" >> parameters.yaml
    else
        sed "s/^  $key:.*/  $key: $value/" parameters.yaml > parameters.updated.yaml
        mv parameters.updated.yaml parameters.yaml
    fi
done
command=(srun "$binary" parameters_file="$PWD/parameters.yaml" seed_tpg="$seed"
         "${overrides[@]}")
printf '%q ' "${command[@]}" > command.txt
printf '\n' >> command.txt
export OMP_NUM_THREADS=1
"${command[@]}" > "logs/tpg.$seed.${SLURM_JOB_ID}.std" 2> "logs/tpg.$seed.${SLURM_JOB_ID}.err"
