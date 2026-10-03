#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=pendulum-acrobot
#SBATCH --array=1-20%4
#SBATCH --ntasks=21
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-23:00:00
#SBATCH --output=transfer-%A_%a.out
#SBATCH --error=transfer-%A_%a.err

set -euo pipefail

# Submit from the same loaded module environment used to build TPGExperimentMPI.
# Usage: sbatch scripts/run/transfer-pendulum-acrobot.slurm [variant] [last_generation]
# Each array task transfers the matching Pendulum checkpoint for one seed.
: "${TPG:?Export TPG to the repository path before submitting}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch as an array}"
: "${SLURM_NTASKS:?Missing Slurm MPI task count}"
variant=${1:-fixed_rates}
last_generation=${2:-5000}
case "$variant" in
    fixed_rates|fixed_individual|execution_modified_rates) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
    echo "Usage: sbatch $0 [variant] [last_generation]" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
cd "$TPG"

# Launch Python once; it starts the MPI executable with srun.
python3 "$TPG/scripts/run/transfer_pendulum_to_acrobot.py" "$variant" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --launcher srun \
    --processes "$SLURM_NTASKS" \
    --jobs 1 \
    --generations "$last_generation"
