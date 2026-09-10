#pragma once

#include "RegisterMachine.h"
#include <array>
#include <iomanip>
#include <sstream>

using XPredPreyRates = std::array<double, 6>;

inline XPredPreyRates ReadXPredPreyRates(const RegisterMachine& program,
                                      bool inherited = false) {
  XPredPreyRates values;
  values.fill(std::numeric_limits<double>::quiet_NaN());
  const auto* memory = program.private_memory_[MemoryEigen::kScalarType_];
  const size_t size = inherited ? memory->const_memory_.size() : memory->working_memory_.size();
  for (size_t i = 0; i < values.size(); ++i) {
    const size_t reg = kSelfModifyingFirstRegister + i;
    if (reg < size) values[i] = inherited ? memory->const_memory_[reg](0, 0)
                                        : memory->working_memory_[reg](0, 0);
  }
  return values;
}

inline std::string XPredPreyRateColumns(const std::string& prefix) {
  std::string columns;
  for (int reg = 2; reg <= 7; ++reg) columns += "," + prefix + "_s" + std::to_string(reg);
  return columns;
}

inline void WriteXPredPreyRates(std::ostream& out, const XPredPreyRates& rates,
                              bool probabilities = false) {
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (double rate : rates) {
    out << ',';
    if (probabilities)
      out << SelfModifyingRawTendencyToProbability(rate);
    else out << rate;
  }
}

inline const char* XPredPreyRoleName(int role) {
  return role == 0 ? "predator" : "prey";
}

inline std::string XPredPreyEncounterHeader() {
  return "seed,generation,phase,encounter_id,encounter_seed,episode,team_id,role,"
         "opponent_id,appearance,encounter_count,program_id,self_modifying,"
         "steps,role_reward,result,program_length,effective_length" +
      XPredPreyRateColumns("before_raw") + XPredPreyRateColumns("after_raw") +
      XPredPreyRateColumns("inherited_raw") + XPredPreyRateColumns("before_probability") +
      XPredPreyRateColumns("after_probability");
}

inline std::string XPredPreyReproductionHeader() {
  return "seed,generation,team_id,role,parent_program_id,child_program_id,"
         "self_modifying,reset_before_mutation,mutation_pass" +
      XPredPreyRateColumns("encounter_output_raw") +
      XPredPreyRateColumns("inherited_before_raw") +
      ",used_swap,used_delete,used_add,used_mutate,used_redundancy" +
      XPredPreyRateColumns("inherited_after_raw");
}
