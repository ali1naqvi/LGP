#include "selection_metrics_builder.h"
#include "selection_metrics.h"
#include <string>
#include <sstream>

SelectionMetricsBuilder& SelectionMetricsBuilder::with_generation(long generation) {
    this->generation = generation;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_fitness(double best_fitness) {
    this->best_fitness = best_fitness;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_validation_fitness(double validation_fitness) {
    this->validation_fitness = validation_fitness;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_team_id(long team_id) {
    this->team_id = team_id;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_team_size(int team_size) {
    this->team_size = team_size;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_age(int age) {
    this->age = age;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_fitness_value_for_selection(double fitness_value_for_selection) {
    this->fitness_value_for_selection = fitness_value_for_selection;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_total_program_instructions(int total_program_instructions) {
    this->program_instruction_count = total_program_instructions;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_total_effective_program_instructions(int total_effective_program_instructions) {
    this->effective_program_instruction_count = total_effective_program_instructions;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_register_size(double best_agent_register_size) {
    this->best_agent_register_size = best_agent_register_size;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_effective_register_size(double size) {
    this->best_agent_effective_register_size = size;
    return *this;
}


SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_flops(double flop_size) {
    this->best_agent_flops = flop_size;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_elite_avg_start_rates(
    double swap_rate, double delete_rate, double add_rate, double mutate_rate,
    double decoy_rate) {
    elite_avg_start_rate_swap = swap_rate;
    elite_avg_start_rate_delete = delete_rate;
    elite_avg_start_rate_add = add_rate;
    elite_avg_start_rate_mutate = mutate_rate;
    elite_avg_start_rate_s5 = decoy_rate;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_elite_avg_output_rates(
    double swap_rate, double delete_rate, double add_rate, double mutate_rate,
    double decoy_rate) {
    elite_avg_output_rate_swap = swap_rate;
    elite_avg_output_rate_delete = delete_rate;
    elite_avg_output_rate_add = add_rate;
    elite_avg_output_rate_mutate = mutate_rate;
    elite_avg_output_rate_s5 = decoy_rate;
    return *this;
}


SelectionMetricsBuilder& SelectionMetricsBuilder::with_operations_use(std::vector<int> operations_use) {
    std::stringstream ss;

    bool first = true;
    for (int op : operations_use) {
        if (!first) {
            ss << ",";
        }
        ss << op;
        first = false;
    }

    this->operations_use = ss.str();
    return *this;
}

SelectionMetrics SelectionMetricsBuilder::build() const {
    return { *this };
}

long SelectionMetricsBuilder::get_generation() const {
    return generation;
}

double SelectionMetricsBuilder::get_best_fitness() const {
    return best_fitness;
}

double SelectionMetricsBuilder::get_validation_fitness() const {
    return validation_fitness;
}

long SelectionMetricsBuilder::get_team_id() const {
    return team_id;
}

int SelectionMetricsBuilder::get_team_size() const {
    return team_size;
}

long SelectionMetricsBuilder::get_age() const {
    return age;
}

double SelectionMetricsBuilder::get_fitness_value_for_selection() const {
    return fitness_value_for_selection;
}

int SelectionMetricsBuilder::get_total_program_instructions() const {
    return program_instruction_count;
}

int SelectionMetricsBuilder::get_total_effective_program_instructions() const {
    return effective_program_instruction_count;
}

double SelectionMetricsBuilder::get_best_agent_register_size() const {
    return best_agent_register_size;
}

double SelectionMetricsBuilder::get_best_agent_effective_register_size() const {
    return best_agent_effective_register_size;
}


double SelectionMetricsBuilder::get_best_agent_flops() const {
    return best_agent_flops;
}

double SelectionMetricsBuilder::get_elite_avg_start_rate_swap() const {
    return elite_avg_start_rate_swap;
}

double SelectionMetricsBuilder::get_elite_avg_start_rate_delete() const {
    return elite_avg_start_rate_delete;
}

double SelectionMetricsBuilder::get_elite_avg_start_rate_add() const {
    return elite_avg_start_rate_add;
}

double SelectionMetricsBuilder::get_elite_avg_start_rate_mutate() const {
    return elite_avg_start_rate_mutate;
}

double SelectionMetricsBuilder::get_elite_avg_start_rate_s5() const {
    return elite_avg_start_rate_s5;
}

double SelectionMetricsBuilder::get_elite_avg_output_rate_swap() const {
    return elite_avg_output_rate_swap;
}

double SelectionMetricsBuilder::get_elite_avg_output_rate_delete() const {
    return elite_avg_output_rate_delete;
}

double SelectionMetricsBuilder::get_elite_avg_output_rate_add() const {
    return elite_avg_output_rate_add;
}

double SelectionMetricsBuilder::get_elite_avg_output_rate_mutate() const {
    return elite_avg_output_rate_mutate;
}

double SelectionMetricsBuilder::get_elite_avg_output_rate_s5() const {
    return elite_avg_output_rate_s5;
}


std::string SelectionMetricsBuilder::get_operations_use() const {
    return operations_use;
}

// NSGA-II Pareto front metrics - setters
SelectionMetricsBuilder& SelectionMetricsBuilder::with_pareto_front_0_size(int size) {
    this->pareto_front_0_size = size;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_pareto_front_1_size(int size) {
    this->pareto_front_1_size = size;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_pareto_rank(int rank) {
    this->best_agent_pareto_rank = rank;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_crowding_dist(double dist) {
    this->best_agent_crowding_dist = dist;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_avg_complexity_front_0(double avg) {
    this->avg_complexity_front_0 = avg;
    return *this;
}

// NSGA-II Pareto front metrics - getters
int SelectionMetricsBuilder::get_pareto_front_0_size() const {
    return pareto_front_0_size;
}

int SelectionMetricsBuilder::get_pareto_front_1_size() const {
    return pareto_front_1_size;
}

int SelectionMetricsBuilder::get_best_agent_pareto_rank() const {
    return best_agent_pareto_rank;
}

double SelectionMetricsBuilder::get_best_agent_crowding_dist() const {
    return best_agent_crowding_dist;
}

double SelectionMetricsBuilder::get_avg_complexity_front_0() const {
    return avg_complexity_front_0;
}

// Register similarity
SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_mean_register_similarity(double sim) {
    this->best_agent_mean_register_similarity = sim;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_best_agent_max_register_similarity(double sim) {
    this->best_agent_max_register_similarity = sim;
    return *this;
}

double SelectionMetricsBuilder::get_best_agent_mean_register_similarity() const {
    return best_agent_mean_register_similarity;
}

double SelectionMetricsBuilder::get_best_agent_max_register_similarity() const {
    return best_agent_max_register_similarity;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_validation_register_efficiency_pearson(double r) {
    this->validation_register_efficiency_pearson = r;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_validation_register_efficiency_spearman(double r) {
    this->validation_register_efficiency_spearman = r;
    return *this;
}

SelectionMetricsBuilder& SelectionMetricsBuilder::with_validation_register_efficiency_n(int n) {
    this->validation_register_efficiency_n = n;
    return *this;
}

double SelectionMetricsBuilder::get_validation_register_efficiency_pearson() const {
    return validation_register_efficiency_pearson;
}

double SelectionMetricsBuilder::get_validation_register_efficiency_spearman() const {
    return validation_register_efficiency_spearman;
}

int SelectionMetricsBuilder::get_validation_register_efficiency_n() const {
    return validation_register_efficiency_n;
}
