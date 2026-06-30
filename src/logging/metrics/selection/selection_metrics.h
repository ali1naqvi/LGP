#ifndef SELECTION_METRICS_H
#define SELECTION_METRICS_H

#include <vector>
#include <optional>
#include <string>

class SelectionMetricsBuilder;

struct SelectionMetrics {
    const long team_id;
    const long generation;
    const double best_fitness;
    const double validation_fitness;
    const int team_size;
    const long age;
    const double fitness_value_for_selection;
    const int program_instruction_count;
    const int effective_program_instruction_count;
    const double best_agent_register_size;
    const double best_agent_effective_register_size;
    const double best_agent_flops;
    // Equal-weighted mean across all training elite root teams. Each team's
    // value is itself the mean of its direct programs. Start rates are the
    // probability readout from inherited constants; output rates are the
    // probability readout from working S2-S6. S6 is the neutral decoy.
    const double elite_avg_start_rate_swap;
    const double elite_avg_start_rate_delete;
    const double elite_avg_start_rate_add;
    const double elite_avg_start_rate_mutate;
    const double elite_avg_start_rate_s5;
    const double elite_avg_output_rate_swap;
    const double elite_avg_output_rate_delete;
    const double elite_avg_output_rate_add;
    const double elite_avg_output_rate_mutate;
    const double elite_avg_output_rate_s5;
    const std::string operations_use;
    
    // NSGA-II Pareto front metrics
    const int pareto_front_0_size;           // Number of non-dominated teams
    const int pareto_front_1_size;           // Number of teams in second front
    const int best_agent_pareto_rank;        // Best agent's Pareto front rank
    const double best_agent_crowding_dist;   // Best agent's crowding distance
    const double avg_complexity_front_0;     // Average complexity of non-dominated front

    // Register structural similarity (clone divergence tracking)
    const double best_agent_mean_register_similarity;
    const double best_agent_max_register_similarity;

    // Validation-only, within-generation statistics across root teams:
    // validation fitness vs (effective_registers / total_registers)
    const double validation_register_efficiency_pearson;
    const double validation_register_efficiency_spearman;
    const int validation_register_efficiency_n;

    SelectionMetrics(const SelectionMetricsBuilder& builder);
};

#include "selection_metrics_builder.h"

#endif
