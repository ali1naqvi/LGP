#!/bin/bash

# replay single best agent from each repeat
for seed in $(ls *.std | grep -v replay | cut -d '.' -f 2); do
   tpg-run-mpi.sh -s $seed -m 1 -r 0
done

# wait for replays to finish
sleep 10

# get and print test MSE from each replayed agent
python3 $TPG/scripts/plot/get_test_mse_all_repeats.py `pwd`
