#include "RegisterMachine.h"
#include "XPredPreyLogging.h"
#include <iostream>
#include <stdexcept>

void check(bool condition, const char* message) {
   if (!condition) throw std::runtime_error(message);
}

int main() {
   instruction::SetupOps();
   std::mt19937 rng(42);
   std::unordered_map<std::string, std::any> params{
       {"self_modifying", 1}, {"matrix_modulation", 0}, {"dynamic_lgp", 0},
       {"use_all_scalar_registers", 0}, {"continuous_output", 1},
       {"p_instructions_swap", 0.0}, {"p_instructions_delete", 0.0},
       {"p_instructions_add", 0.0}, {"p_instructions_mutate", 0.0},
       {"max_prog_size", 100}, {"p_memory_mu_const", 0.0},
       {"mut_learning_rate", 0.0}, {"p_memory_size", 0.0},
       {"p_memory_slots", 0.0}, {"p_hidden_stateful_mutation", 0.0},
       {"p_observation_index", 0.0}};
   std::vector<std::string> fields{
       "RegisterMachine", "1", "0", "-1", "0", "0", "1", "0", "0", "0", "8", "SM1"};
   std::vector<std::map<long, MemoryEigen*>> memories(3);
   bool rejected = false;
   try { RegisterMachine legacy(fields, memories, params, rng); }
   catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("Legacy self-modifying") != std::string::npos;
   }
   check(rejected, "Legacy layout must be rejected before memory is loaded");
   fields.push_back("SL4");
   for (int type = 0; type < 3; ++type) memories[type][1] = new MemoryEigen(type, 8, 1);
   RegisterMachine program(fields, memories, params, rng);
   auto* scalar = program.private_memory_[0];
   for (size_t reg = 2; reg <= 5; ++reg) scalar->working_memory_[reg](0, 0) = double(reg);
   auto rates = program.MutationProbabilities(params);
   check(rates.size() == 4 && kSelfModifyingDecoyRegister == 6, "Four rates plus S6 decoy required");
   scalar->working_memory_[6](0, 0) = 1000;
   check(rates == program.MutationProbabilities(params), "Decoy must not affect mutation rates");
   for (int destination : {1, 2, 6, 7}) {
      auto* op = new instruction(params, rng, 8);
      op->op_ = instruction::SCALAR_SUM_OP_;
      op->outIdx_ = destination;
      op->in1Src_ = op->in2Src_ = 2;
      op->in0Idx_ = op->in1Idx_ = op->in2Idx_ = op->in3Idx_ = 0;
      program.instructions_.push_back(op);
   }
   program.MarkIntrons(params, 8);
   check(program.instructions_effective_.size() == 2, "Only action and active-rate writes should survive");
   check(program.n_self_modification_only_instructions_ == 1, "Only active-rate instruction is self-modification-only");
   check(program.ToString(false).find(":SM1:SL4:") != std::string::npos, "Checkpoint must identify the new layout");
   check(XPredPreyReproductionHeader().find("redundancy") == std::string::npos, "No obsolete logging column");
   // A legacy config key cannot re-enable duplication, even at probability one.
   params["self_modifying"] = 0;
   params["p_instructions_redundancy"] = 1.0;
   std::unordered_map<std::string, int> state;
   std::vector<bool> legal_ops(instruction::NUM_OP, true);
   for (int pass = 0; pass < 100; ++pass) program.Mutate(params, state, rng, legal_ops);
   check(program.instructions_.size() == 4, "Removed duplication operator must never run");
   std::cout << "Four-rate layout, decoy isolation, metrics, checkpoint guard, and mutation checks passed\n";
}
