#include "RegisterMachine.h"
#include "TPG.h"
#include "XPredPreyLogging.h"
#include "ActionWrappers.h"
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
   auto init_params = params;
   init_params.insert({
       {"stateful", 1}, {"static_learning_rate", 0.0},
       {"observation_buff_size", 1}, {"min_initial_prog_size", 1},
       {"max_initial_prog_size", 1}, {"min_initial_mem_slots", 10},
       {"max_initial_mem_slots", 10}, {"memory_size", 1}});
   for (const auto* key : {"p_instructions_swap", "p_instructions_delete",
                          "p_instructions_add", "p_instructions_mutate"}) {
      init_params[key] = 0.2;
   }
   init_params["p_memory_mu_const"] = 0.25;
   std::unordered_map<std::string, int> init_state;
   std::vector<bool> init_ops(instruction::NUM_OP, false);
   init_ops[instruction::SCALAR_SUM_OP_] = true;
   auto fixed_rng = rng;
   auto random_rng = rng;
   RegisterMachine fixed(-1, 0, init_params, init_state, fixed_rng, init_ops);
   init_params["randomize_self_modifying_constants"] = 1;
   RegisterMachine random(-1, 0, init_params, init_state, random_rng, init_ops);
   check(fixed_rng == random_rng, "Random initialization should reuse seeded memory draws");
   RegisterMachine child(random, init_params, init_state, random_rng);
   const auto& random_scalar = *random.private_memory_[MemoryEigen::kScalarType_];
   const auto& fixed_scalar = *fixed.private_memory_[MemoryEigen::kScalarType_];
   const auto& child_scalar = *child.private_memory_[MemoryEigen::kScalarType_];
   for (size_t reg = kSelfModifyingFirstRegister; reg <= kSelfModifyingDecoyRegister; ++reg) {
      const double raw = random_scalar.const_memory_[reg](0, 0);
      check(raw >= -1.0 && raw <= 1.0, "Random constants must use the ordinary raw range");
      check(random_scalar.working_memory_[reg](0, 0) == raw,
            "Random constants must seed working memory");
      check(child_scalar.const_memory_[reg](0, 0) == raw,
            "Clones must inherit random constants");
      check(std::abs(SelfModifyingRawTendencyToProbability(
                         fixed_scalar.const_memory_[reg](0, 0)) - 0.2) < 1e-12,
            "Fixed constants must start at the configured probability");
   }
   init_params["randomize_self_modifying_constants"] = 0;
   init_params["randomize_self_modifying_register_order"] = 1;
   init_params["seed_tpg"] = 42;
   for (const auto* key : {"p_instructions_swap", "p_instructions_delete",
                          "p_instructions_add", "p_instructions_mutate"}) {
      init_params[key] = 0.1;
   }
   std::set<std::array<size_t, kSelfModifyingRegisterCount>> founder_orders;
   for (int founder = 0; founder < 64; ++founder) {
      RegisterMachine individual(-1, 0, init_params, init_state, random_rng, init_ops);
      const auto order = individual.RateRegisters(init_params);
      founder_orders.insert(order);
      check(std::is_permutation(order.begin(), order.end(), std::array{2, 3, 4, 5}.begin()),
            "Each founder must get a permutation of S2-S5");
      for (double rate : individual.MutationProbabilities(init_params)) {
         check(std::abs(rate - 0.1) < 1e-12, "Random order must start at 0.1");
      }
      for (size_t reg : order) {
         individual.private_memory_[0]->working_memory_[reg](0, 0) = double(reg);
      }
      const auto probabilities = individual.MutationProbabilities(init_params);
      for (size_t op = 0; op < order.size(); ++op) {
         check(probabilities[op] == SelfModifyingRawTendencyToProbability(double(order[op])),
               "Mutation must use the founder's operator assignments");
      }
      RegisterMachine offspring(individual, init_params, init_state, random_rng);
      check(offspring.RateRegisters(init_params) == order &&
            offspring.MutationProbabilities(init_params) == probabilities,
            "Offspring must inherit the mapping and phenotype");
      RegisterMachine crossover(-1, individual.instructions_, init_params, init_state,
                                random_rng, init_ops, individual.n_memories_, &individual);
      check(crossover.RateRegisters(init_params) == order,
            "Crossover must inherit its primary parent's mapping");
      std::vector<std::string> saved_fields;
      const auto record = individual.ToString(false);
      SplitString(record.substr(0, record.find('\n')), ':', saved_fields);
      std::vector<std::map<long, MemoryEigen*>> saved_memory(3);
      for (size_t type = 0; type < saved_memory.size(); ++type) {
         saved_memory[type][individual.id_] = new MemoryEigen(*individual.private_memory_[type]);
      }
      RegisterMachine restored(saved_fields, saved_memory, init_params, random_rng);
      check(restored.RateRegisters(init_params) == order, "Checkpoint must preserve the mapping");
   }
   check(founder_orders.size() > 4, "Founders must sample beyond the four cyclic orders");
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
   const std::array<std::array<size_t, 4>, 4> expected_registers{{
       {2, 3, 4, 5}, {5, 2, 3, 4}, {4, 5, 2, 3}, {3, 4, 5, 2}}};
   for (int rotation = 0; rotation < 4; ++rotation) {
      auto rotated = params;
      rotated["self_modifying_register_rotation"] = rotation;
      const auto probabilities = program.MutationProbabilities(rotated);
      for (size_t op = 0; op < probabilities.size(); ++op) {
         check(probabilities[op] == rates[expected_registers[rotation][op] - 2],
               "Mutation must read each operator's assigned register");
      }
      rotated["self_modifying_mutation_rates"] = std::string("p_instructions_delete");
      const auto delete_only = program.MutationProbabilities(rotated);
      check(delete_only == std::array<double, 4>{0.0, probabilities[1], 0.0, 0.0},
            "The selected operators must use their rotated register assignments");
   }
   auto invalid_order = params;
   invalid_order["self_modifying_register_rotation"] = 4;
   rejected = false;
   try { SelfModifyingRateRegisters(invalid_order); }
   catch (const std::invalid_argument&) { rejected = true; }
   check(rejected, "A rotation outside 0 through 3 must be rejected");
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
   EvalData eval;
   eval.program_out = &program;
   for (size_t reg = 0; reg < 8; ++reg) scalar->working_memory_[reg](0, 0) = double(reg) / 10;
   auto actions = WrapScalarActionsMujoco(eval);
   check(actions.size() == 3, "Self-modifying actions must skip S2-S6");
   check(actions[2] == std::tanh(0.7), "Third action must read S7");
   params["use_all_scalar_registers"] = 1;
   params["environment_action_space"] = 3;
   program.MarkIntrons(params, 8);
   check(program.instructions_effective_.size() == 3, "Scalar action mapping must preserve S7 writes");
   check(program.n_self_modification_only_instructions_ == 1, "Rate outputs must remain separate from actions");
   program.self_modifying_ = false;
   actions = WrapScalarActionsMujoco(eval);
   check(actions.size() == 8 && actions[2] == std::tanh(0.2), "Ordinary programs must use consecutive registers");
   program.self_modifying_ = true;
   params["use_all_scalar_registers"] = 0;
   // Held-out MPI results must not change the rates used by reproduction.
   TPG tpg;
   tpg.program_pop_[program.id_] = &program;
   const auto training_rates = program.MutationProbabilities(params);
   const double training_decoy = scalar->working_memory_[6](0, 0);
   std::string held_out = "R:" + std::to_string(program.id_) + ":-1:-2:-3:-4:-5\n";
   std::vector<team*> no_teams;
   for (int phase : {_VALIDATION_PHASE, _TEST_PHASE}) {
      tpg.state_["phase"] = phase;
      tpg.DecodeEvalResultString(held_out);
      check(program.MutationProbabilities(params) == training_rates,
            "Held-out results changed training mutation probabilities");
      check(scalar->working_memory_[6](0, 0) == training_decoy,
            "Held-out results changed training decoy");
      std::string result = "fitness\n";
      tpg.AppendSelfModifyingRates(result, no_teams);
      check(result == "fitness\n", "Held-out sender changed result payload");
   }
   tpg.state_["phase"] = _TRAIN_PHASE;
   tpg.DecodeEvalResultString(held_out);
   check(scalar->working_memory_[4](0, 0) == -3,
         "Training results must still update mutation registers");
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
