
#include "EvalData.h"

void EvalData::AccumulateStepData() {
    if (n_prediction == 0) {
        fill(stats_double.begin(), stats_double.end(), 0);
        fingerprint.clear();
    }
    n_prediction++;
    stats_double[VISITED_TEAMS_IDX] += team_path.size();
    stats_double[INSTRUCTIONS_IDX] += instruction_count;

    timestep++;
}