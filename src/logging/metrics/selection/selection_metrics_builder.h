#ifndef SELECTION_METRICS_BUILDER_H
#define SELECTION_METRICS_BUILDER_H

#include "selection_metrics.h"
#include <string>

class SelectionMetricsBuilder {
public:
    SelectionMetricsBuilder& with_generation(long generation);
    SelectionMetricsBuilder& with_best_fitness(double best_fitness);
    SelectionMetricsBuilder& with_validation_fitness(double validation_fitness);
    SelectionMetricsBuilder& with_team_id(long team_id);    
    SelectionMetricsBuilder& with_team_size(int team_size);
    SelectionMetricsBuilder& with_age(int age);
    SelectionMetricsBuilder& with_fitness_value_for_selection(double fitness_value_for_selection);
    SelectionMetricsBuilder& with_total_program_instructions(int total_program_instructions);
    SelectionMetricsBuilder& with_total_effective_program_instructions(int total_effective_program_instructions);
    SelectionMetricsBuilder& with_best_agent_register_size(double best_agent_register_size);
    SelectionMetricsBuilder& with_best_agent_effective_register_size(double size);
    SelectionMetricsBuilder& with_best_agent_flops(double flop_size);
    SelectionMetricsBuilder& with_elite_avg_start_rates(double swap_rate,
                                                        double delete_rate,
                                                        double add_rate,
                                                        double mutate_rate,
                                                        double decoy_rate);
    SelectionMetricsBuilder& with_elite_avg_output_rates(double swap_rate,
                                                         double delete_rate,
                                                         double add_rate,
                                                         double mutate_rate,
                                                         double decoy_rate);
    SelectionMetricsBuilder& with_operations_use(std::vector<int> operations);
    
    // NSGA-II Pareto front metrics
    SelectionMetricsBuilder& with_pareto_front_0_size(int size);
    SelectionMetricsBuilder& with_pareto_front_1_size(int size);
    SelectionMetricsBuilder& with_best_agent_pareto_rank(int rank);
    SelectionMetricsBuilder& with_best_agent_crowding_dist(double dist);
    SelectionMetricsBuilder& with_avg_complexity_front_0(double avg);

    // Register similarity
    SelectionMetricsBuilder& with_best_agent_mean_register_similarity(double sim);
    SelectionMetricsBuilder& with_best_agent_max_register_similarity(double sim);

    // Validation-fitness vs register efficiency statistics (validation generations only)
    SelectionMetricsBuilder& with_validation_register_efficiency_pearson(double r);
    SelectionMetricsBuilder& with_validation_register_efficiency_spearman(double r);
    SelectionMetricsBuilder& with_validation_register_efficiency_n(int n);

    long get_generation() const;
    double get_best_fitness() const;
    double get_validation_fitness() const;
    long get_team_id() const;
    int get_team_size() const;
    long get_age() const;
    double get_fitness_value_for_selection() const;
    int get_total_program_instructions() const;
    int get_total_effective_program_instructions() const;
    double get_best_agent_register_size() const;
    double get_best_agent_effective_register_size() const;
    double get_best_agent_flops() const;
    double get_elite_avg_start_rate_swap() const;
    double get_elite_avg_start_rate_delete() const;
    double get_elite_avg_start_rate_add() const;
    double get_elite_avg_start_rate_mutate() const;
    double get_elite_avg_start_rate_s5() const;
    double get_elite_avg_output_rate_swap() const;
    double get_elite_avg_output_rate_delete() const;
    double get_elite_avg_output_rate_add() const;
    double get_elite_avg_output_rate_mutate() const;
    double get_elite_avg_output_rate_s5() const;
    std::string get_operations_use() const;
    
    // NSGA-II getters
    int get_pareto_front_0_size() const;
    int get_pareto_front_1_size() const;
    int get_best_agent_pareto_rank() const;
    double get_best_agent_crowding_dist() const;
    double get_avg_complexity_front_0() const;

    // Register similarity getters
    double get_best_agent_mean_register_similarity() const;
    double get_best_agent_max_register_similarity() const;

    double get_validation_register_efficiency_pearson() const;
    double get_validation_register_efficiency_spearman() const;
    int get_validation_register_efficiency_n() const;

    SelectionMetrics build() const;

private:
    long team_id = 0;
    long generation = 0;
    double best_fitness = 0.0;
    double validation_fitness = 0.0;
    int team_size = 0;
    long age = 0;
    double fitness_value_for_selection = 0.0;
    int program_instruction_count = 0;
    int effective_program_instruction_count = 0;
    double best_agent_register_size = 0.0;
    double best_agent_effective_register_size = 0.0;
    double best_agent_flops = 0.0;
    double elite_avg_start_rate_swap = 0.0;
    double elite_avg_start_rate_delete = 0.0;
    double elite_avg_start_rate_add = 0.0;
    double elite_avg_start_rate_mutate = 0.0;
    double elite_avg_start_rate_s5 = 0.0;
    double elite_avg_output_rate_swap = 0.0;
    double elite_avg_output_rate_delete = 0.0;
    double elite_avg_output_rate_add = 0.0;
    double elite_avg_output_rate_mutate = 0.0;
    double elite_avg_output_rate_s5 = 0.0;
    std::string operations_use = "";
    
    // NSGA-II Pareto front metrics
    int pareto_front_0_size = 0;
    int pareto_front_1_size = 0;
    int best_agent_pareto_rank = 0;
    double best_agent_crowding_dist = 0.0;
    double avg_complexity_front_0 = 0.0;

    // Register similarity
    double best_agent_mean_register_similarity = 0.0;
    double best_agent_max_register_similarity = 0.0;

    double validation_register_efficiency_pearson = 0.0;   // NaN on non-validation generations
    double validation_register_efficiency_spearman = 0.0;  // NaN on non-validation generations
    int validation_register_efficiency_n = 0;
};


#endif
