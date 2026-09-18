#!/bin/bash 
#SBATCH --account=def-skelly

# cpus anywhere
#SBATCH --ntasks=21               
#SBATCH --mem-per-cpu=6G      
#SBATCH --time=0-12:00  # time (DD-HH:MM)

#SBATCH --mail-user=alinaqvi8014@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL


mkdir -p checkpoints
mkdir -p logs

#defaults
mode=0 #Train:0, Replay:1, Debug:2
seed_tpg=42
parameters_file="parameters_TPG.yaml"

while getopts m:p:s: flag
do
   case "${flag}" in
      m) mode=${OPTARG};;
      p) parameters_file=${OPTARG};;
      s) seed_tpg=${OPTARG};;
   esac
done

# Start from scratch ###########################################################
if [ $mode -eq 0 ]; then
  srun $TPG/build/release/experiments/TPGExperimentMPI \
  parameters_file=${parameters_file} \
  seed_tpg=${seed_tpg} \
  1> logs/tpg.${seed_tpg}.$$.std \
  2> logs/tpg.${seed_tpg}.$$.err
fi

# Pickup from checkpoint #######################################################
if [ $mode -eq 4 ]; then
  checkpoint_in_phase=0
  checkpoint_in_t=$(python3 "$TPG/scripts/run/latest-checkpoint.py" \
    "$seed_tpg" --phase "$checkpoint_in_phase") || exit 1
  # Keep each resume attempt's logs; do not append to an arbitrary older job.
  pid=${SLURM_JOB_ID:-$$}
  echo "Starting run ${seed_tpg} t ${checkpoint_in_t} pid $pid"
  srun $TPG/build/release/experiments/TPGExperimentMPI \
    parameters_file=${parameters_file} \
    seed_tpg=${seed_tpg} \
    start_from_checkpoint=1 \
    checkpoint_in_phase=${checkpoint_in_phase} \
    checkpoint_in_t=${checkpoint_in_t} \
    1>> logs/tpg.${seed_tpg}.${pid}.std \
    2>> logs/tpg.${seed_tpg}.${pid}.err
fi
