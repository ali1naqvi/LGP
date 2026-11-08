#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=acrobot-frozen
#SBATCH --array=1-60%4
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-23:00:00
#SBATCH --output=acrobot-frozen-%A_%a.out
#SBATCH --error=acrobot-frozen-%A_%a.err

set -euo pipefail
# Continue all 60 dynamic runs from 5000 through 6000 with mass frozen at 5000.
# Usage: sbatch scripts/run/acrobot-frozen.slurm [source_root] [output_root]
(( $# <= 2 )) || { echo "Expected optional source_root and output_root" >&2; exit 2; }
repo=${TPG:-${SLURM_SUBMIT_DIR:?Submit using sbatch}}
repo=$(cd "$repo" && pwd)
source_root=${1:-"$repo/experiments/acrobot_dynamic"}
output_root=${2:-"$repo/experiments/acrobot_dynamic_frozen"}
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
: "${SLURM_NTASKS:?Missing MPI task count}"
(( SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 60 && SLURM_NTASKS >= 2 )) || exit 2
variants=(fixed_rates fixed_individual execution_modified_rates)
variant=${variants[$(( (SLURM_ARRAY_TASK_ID - 1) / 20 ))]}
seed=$(( (SLURM_ARRAY_TASK_ID - 1) % 20 + 1 ))
binary="$repo/build/release/experiments/TPGExperimentMPI"
source_dir="$source_root/$variant/seed_$seed"
checkpoint="$source_dir/checkpoints/cp.5000.$seed.0.rslt"
[[ -x $binary && -f $checkpoint && -f "$source_dir/parameters.yaml" ]] || {
    echo "Missing executable, source config, or checkpoint: $checkpoint" >&2; exit 1;
}
[[ $(tail -n 1 "$checkpoint") == end ]] || { echo "Incomplete checkpoint: $checkpoint" >&2; exit 1; }
mkdir -p "$output_root/$variant"
run_dir="$output_root/$variant/seed_$seed"
mkdir "$run_dir" || { echo "Run directory already exists: $run_dir" >&2; exit 1; }
mkdir -p "$run_dir/logs" "$run_dir/checkpoints"
cp "$checkpoint" "$run_dir/checkpoints/"
# Preserve the source run's settings, replacing any previous freeze setting.
sed '/^[[:space:]]*acrobot_mass_freeze_generation:/d' "$source_dir/parameters.yaml" > "$run_dir/parameters.yaml"
printf '\nacrobot_frozen_continuation:\n  acrobot_mass_freeze_generation: 5000\n' >> "$run_dir/parameters.yaml"
cd "$run_dir"
export OMP_NUM_THREADS=1
srun --ntasks="$SLURM_NTASKS" "$binary" \
    parameters_file="$PWD/parameters.yaml" seed_tpg="$seed" seed_aux=42 \
    start_from_checkpoint=1 checkpoint_in_t=5000 checkpoint_in_phase=0 \
    n_generations=6000 replay=0 animate=0 pid="$$" \
    > "logs/tpg.$seed.$$.std" 2> "logs/tpg.$seed.$$.err"
