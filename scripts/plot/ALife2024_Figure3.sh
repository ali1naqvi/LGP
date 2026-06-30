#!/bin/bash

# find single best test fitness score
best_test_fitness=$(grep setEl *std | grep "phs 0" | \
  awk -F "p2t0a0" '{print $2}' | awk '{print $1}' | sort -n | uniq | tail -n 1)
echo "best_test_fitness: $best_test_fitness"

# find repeat (seed) which produced signle best test fitness
seed_of_best_repeat=$(grep -iRl "p2t0a0 $best_test_fitness" *.std | \
  cut -d '.' -f 2)
echo "Seed_of_best_repeat: $seed_of_best_repeat"

# replay single best agent
tpg-run-mpi.sh -m 1 -s $seed_of_best_repeat -r 0

# wait for replay to finish
sleep 5

# plot test results for single best agent
#python3 $TPG/scripts/plot/tpg-plot-horizon-alife2024.py tpg_${seed_of_best_repeat}_test_t200.csv

# view plot
#evince tpg_${seed_of_best_repeat}_test_t200.pdf
