#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=acrobot-dynamic
#SBATCH --array=1-60%4
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-23:00:00
#SBATCH --output=acrobot-dynamic-%A_%a.out
#SBATCH --error=acrobot-dynamic-%A_%a.err

set -euo pipefail
# Submit from the repository root in the module environment used to build TPG.
# Array 1-20: fixed rates; 21-40: fixed individual; 41-60: execution modified.
# Usage: sbatch scripts/run/acrobot-dynamic.slurm [last_generation] [output_root]
if (( $# > 2 )); then
    echo "Usage: sbatch $0 [last_generation] [output_root]" >&2; exit 2
fi
repo=${TPG:-${SLURM_SUBMIT_DIR:?Submit using sbatch}}
repo=$(cd "$repo" && pwd)
generations=${1:-5000}
output_root=${2:-"$repo/experiments/acrobot_dynamic"}
[[ $generations =~ ^[0-9]+$ ]] || { echo 'Invalid generation count' >&2; exit 2; }
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
: "${SLURM_NTASKS:?Missing MPI task count}"
(( SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 60 && SLURM_NTASKS >= 2 )) || exit 2
variants=(fixed_rates fixed_individual execution_modified_rates)
variant=${variants[$(( (SLURM_ARRAY_TASK_ID - 1) / 20 ))]}
seed=$(( (SLURM_ARRAY_TASK_ID - 1) % 20 + 1 ))
binary="$repo/build/release/experiments/TPGExperimentMPI"
[[ -x $binary ]] || { echo "Rebuild the executable first: $binary" >&2; exit 1; }
mkdir -p "$output_root/$variant"
run_dir="$output_root/$variant/seed_$seed"
# Atomic guard: never overwrite an existing or concurrently running experiment.
mkdir "$run_dir" || { echo "Run directory already exists: $run_dir" >&2; exit 1; }
cd "$run_dir"
mkdir -p logs checkpoints
cp "$repo/configs/acrobot_dynamic_$variant.yaml" parameters.yaml
export OMP_NUM_THREADS=1
printf 'variant=%s\nseed_tpg=%s\nseed_aux=42\nmass_seed=42\nlast_generation=%s\njob_id=%s\n' \
    "$variant" "$seed" "$generations" "$SLURM_JOB_ID" > run_settings.txt
git -C "$repo" rev-parse HEAD >> run_settings.txt
srun --ntasks="$SLURM_NTASKS" "$binary" \
    parameters_file="$PWD/parameters.yaml" seed_tpg="$seed" seed_aux=42 \
    n_generations="$generations" pid="$$" \
    > "logs/tpg.$seed.$$.std" 2> "logs/tpg.$seed.$$.err"
