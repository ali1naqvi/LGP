#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=acrobot-test
#SBATCH --array=1-60%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=0-01:00:00
#SBATCH --output=acrobot-test-%A_%a.out
#SBATCH --error=acrobot-test-%A_%a.err

set -euo pipefail

# From the repository root, prepare the manifest before submitting:
#   python3 scripts/replay_test_table.py prepare --environment Acrobot
#   sbatch scripts/run/acrobot-test-replay.slurm
# After all array tasks finish, collect and plot:
#   python3 scripts/replay_test_table.py collect --environment Acrobot --episodes 20
#   python3 scripts/plot/plot_test_boxplots.py --environment Acrobot
: "${TPG:?Export TPG to the repository path before submitting}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array}"

export OMP_NUM_THREADS=1
cd "$TPG"
python3 "$TPG/scripts/replay_test_table.py" run \
    --episodes 20 \
    --environment Acrobot \
    --job-index "$SLURM_ARRAY_TASK_ID"
