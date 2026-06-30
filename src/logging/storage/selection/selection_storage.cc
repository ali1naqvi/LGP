#include "selection_storage.h"
#include "instruction.h"
#include <iomanip>

void SelectionStorage::init(const int& seed_tpg, const int& pid) {
    std::string filename = generate_filename("selection", seed_tpg, pid);

    file_.open(filename);
    file_ << "generation,best_fitness,validation_fitness,team_id,team_size,age,fitness_value_for_selection,program_instruction_count,effective_program_instruction_count,best_agent_register_size,best_agent_effective_register_size,best_agent_flops,elite_avg_start_rate_swap,elite_avg_start_rate_delete,elite_avg_start_rate_add,elite_avg_start_rate_mutate,elite_avg_start_rate_s5,elite_avg_output_rate_swap,elite_avg_output_rate_delete,elite_avg_output_rate_add,elite_avg_output_rate_mutate,elite_avg_output_rate_s5,pareto_front_0_size,pareto_front_1_size,best_agent_pareto_rank,best_agent_crowding_dist,avg_complexity_front_0,best_agent_mean_register_similarity,best_agent_max_register_similarity,validation_register_efficiency_pearson,validation_register_efficiency_spearman,validation_register_efficiency_n";
    
    appendOperationHeaders();

    file_ << "\n";
    file_.flush();
}

void SelectionStorage::append(const SelectionMetrics& metrics) {
    file_ << metrics.generation << ","
          << std::fixed << std::setprecision(6) << metrics.best_fitness << ","
          << std::fixed << std::setprecision(6) << metrics.validation_fitness << ","
          << metrics.team_id << ","
          << metrics.team_size << ","
          << metrics.age << ","
          << metrics.fitness_value_for_selection << ","
          << metrics.program_instruction_count << ","
          << metrics.effective_program_instruction_count << ","
          << metrics.best_agent_register_size << ","
          << metrics.best_agent_effective_register_size << ","
          << metrics.best_agent_flops << ","
          << metrics.elite_avg_start_rate_swap << ","
          << metrics.elite_avg_start_rate_delete << ","
          << metrics.elite_avg_start_rate_add << ","
          << metrics.elite_avg_start_rate_mutate << ","
          << metrics.elite_avg_start_rate_s5 << ","
          << metrics.elite_avg_output_rate_swap << ","
          << metrics.elite_avg_output_rate_delete << ","
          << metrics.elite_avg_output_rate_add << ","
          << metrics.elite_avg_output_rate_mutate << ","
          << metrics.elite_avg_output_rate_s5 << ","
          << metrics.pareto_front_0_size << ","
          << metrics.pareto_front_1_size << ","
          << metrics.best_agent_pareto_rank << ","
          << metrics.best_agent_crowding_dist << ","
          << metrics.avg_complexity_front_0 << ","
          << metrics.best_agent_mean_register_similarity << ","
          << metrics.best_agent_max_register_similarity << ","
          << metrics.validation_register_efficiency_pearson << ","
          << metrics.validation_register_efficiency_spearman << ","
          << metrics.validation_register_efficiency_n << ","
          << metrics.operations_use << "\n";
    file_.flush();
}

void SelectionStorage::appendOperationHeaders() {
    for (std::string op_name : instruction::op_names_) {
        std::transform(op_name.begin(), op_name.end(), op_name.begin(), ::tolower);
        file_ << "," << op_name;
    }
}
