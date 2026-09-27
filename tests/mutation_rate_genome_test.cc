#include "TPG.h"
#include <cassert>
#include <iostream>
#include <memory>

template <class F> void Rejects(F operation) {
   bool rejected = false;
   try { operation(); } catch (const std::exception&) { rejected = true; }
   assert(rejected);
}

int main() {
   // New keys are available as CLI overrides even with an unchanged legacy YAML.
   TPG cli;
   char arg0[] = "test";
   char arg1[] = "parameters_file=configs/pendulum_fixed_rates.yaml";
   char arg2[] = "genome_mutation_rates=p_instructions_swap";
   char arg3[] = "genome_rate_sigma=0";
   char* args[] = {arg0, arg1, arg2, arg3};
   cli.SetParams(4, args);
   assert(MutationRateGenome::FromParams(cli.params_).genes.size() == 1);

   TPG tpg;
   tpg.ReadParameters("configs/pendulum_fixed_individual.yaml", tpg.params_);
   auto& p = tpg.params_;
   p["genome_mutation_rates"] = std::string("p_instructions_add,p_instructions_mutate");
   p["min_initial_mem_slots"] = 2; // No reserved S2-S6 needed.
   p["max_initial_mem_slots"] = 2;
   p["min_initial_prog_size"] = 4;
   p["max_initial_prog_size"] = 4;
   p["memory_size"] = 2;
   p["p_memory_mu_const"] = 0.0;
   p["p_scalar_stateful_mutation"] = 0.0;
   p["p_observation_index"] = 0.0;
   p["mut_learning_rate"] = 0.0;
   p["p_instructions_swap"] = 0.0;
   p["p_instructions_delete"] = 0.0;
   p["n_mutation_passes"] = 3;
   tpg.Seed(TPG_SEED, 42);
   std::mt19937 rng(99);
   RegisterMachine parent(-1, 0, p, tpg.state_, rng, tpg._ops);
   assert(parent.n_memories_ == 2);
   assert(!parent.self_modifying_);
   assert(parent.mutation_rate_genome_.genes.size() == 2);
   parent.mutation_rate_genome_.genes["p_instructions_add"] = 0.12345678901234567;
   const auto inherited = parent.mutation_rate_genome_.Encode();
   RegisterMachine copy(parent);
   RegisterMachine child(parent, p, tpg.state_, rng);
   assert(copy.mutation_rate_genome_.Encode() == inherited);
   assert(child.mutation_rate_genome_.Encode() == inherited);

   // The reproduction entry point adapts once, independently of pass count.
   auto expected = child.mutation_rate_genome_;
   auto expected_rng = tpg.rngs_[TPG_SEED];
   expected.Mutate(p, expected_rng);
   tpg.ProgramMutator_Instructions(&child);
   assert(child.mutation_rate_genome_.genes == expected.genes);
   assert(child.mutation_rate_genome_.Encode() != inherited);
   assert(parent.mutation_rate_genome_.Encode() == inherited);

   // Program checkpoint round-trip, including exact double precision.
   std::vector<std::string> fields;
   SplitString(child.ToString(false).substr(0, child.ToString(false).find('\n')), ':', fields);
   std::vector<std::map<long, MemoryEigen*>> memory(3);
   for (size_t type = 0; type < 3; ++type)
      memory[type][child.id_] = new MemoryEigen(*child.private_memory_[type]);
   RegisterMachine restored(fields, memory, p, rng);
   assert(restored.mutation_rate_genome_.genes == child.mutation_rate_genome_.genes);
   assert(restored.instructions_.size() == child.instructions_.size());

   // Legacy checkpoints seed genes from config; disabled mode preserves old format/RNG.
   fields.erase(std::remove_if(fields.begin(), fields.end(), [](const auto& f) {
      return f.rfind("MRG", 0) == 0;
   }), fields.end());
   for (size_t type = 0; type < 3; ++type)
      memory[type][child.id_] = new MemoryEigen(*child.private_memory_[type]);
   RegisterMachine legacy(fields, memory, p, rng);
   assert(legacy.mutation_rate_genome_.genes == MutationRateGenome::FromParams(p).genes);
   auto disabled = p;
   disabled["genome_mutation_rates"] = std::string("");
   auto empty = MutationRateGenome::FromParams(disabled);
   const auto saved_rng = rng;
   empty.Mutate(disabled, rng);
   assert(saved_rng == rng && empty.Encode().empty());
   for (size_t type = 0; type < 3; ++type)
      memory[type][child.id_] = new MemoryEigen(*child.private_memory_[type]);
   RegisterMachine old_mode(fields, memory, disabled, rng);
   assert(old_mode.ToString(false).find("MRG") == std::string::npos);

   // Genes override fixed rates and survive changes/resets to executable memory.
   p["genome_rate_sigma"] = 0.0;
   p["p_instructions_add"] = 0.0;
   p["p_instructions_mutate"] = 0.0;
   child.mutation_rate_genome_.genes["p_instructions_add"] = 0.999999;
   child.mutation_rate_genome_.genes["p_instructions_mutate"] = 0.000001;
   auto frozen = child.mutation_rate_genome_.genes;
   for (int trial = 0; trial < 20; ++trial) {
      for (auto* bank : child.private_memory_)
         for (auto& cell : bank->working_memory_) cell.setConstant(trial - 10.0);
      child.ClearWorkingMemory();
      child.ConfigureSelfModifyingRegisters(p);
      const auto length = child.instructions_.size();
      tpg.ProgramMutator_Instructions(&child);
      assert(child.instructions_.size() == length + 3);
      assert(child.mutation_rate_genome_.genes == frozen);
   }

   // Crossover and its small-parent fallback inherit the corresponding strategy block.
   RegisterMachine *c1 = nullptr, *c2 = nullptr;
   tpg.RegisterMachineCrossover(&parent, &child, &c1, &c2);
   assert(c1->mutation_rate_genome_.genes == parent.mutation_rate_genome_.genes);
   assert(c2->mutation_rate_genome_.genes == child.mutation_rate_genome_.genes);
   delete c1; delete c2;
   for (auto* instr : parent.instructions_) instr->outIdx_ = 0;
   tpg.LinearCrossover(&parent, &child, &c1, &c2);
   assert(c1->mutation_rate_genome_.genes == parent.mutation_rate_genome_.genes);
   assert(c2->mutation_rate_genome_.genes == child.mutation_rate_genome_.genes);
   delete c1; delete c2;
   for (auto* program : {&parent, &child}) {
      for (size_t i = 0; i < program->instructions_.size(); ++i) {
         program->instructions_[i]->op_ = instruction::SCALAR_SUM_OP_;
         program->instructions_[i]->outIdx_ = i % 2;
      }
   }
   tpg.LinearCrossover(&parent, &child, &c1, &c2);
   assert(c1->mutation_rate_genome_.genes == parent.mutation_rate_genome_.genes);
   assert(c2->mutation_rate_genome_.genes == child.mutation_rate_genome_.genes);
   delete c1; delete c2;

   // All 16 subsets are selectable; omitted rates use their fixed values.
   const std::vector<std::string> names = {"p_instructions_add", "p_instructions_delete",
                                         "p_instructions_swap", "p_instructions_mutate"};
   for (unsigned mask = 0; mask < 16; ++mask) {
      auto subset_params = p;
      std::string selection;
      for (size_t i = 0; i < names.size(); ++i) {
         subset_params[names[i]] = 0.25;
         if (mask & (1u << i)) {
            if (!selection.empty()) selection += ',';
            selection += names[i];
         }
      }
      subset_params["genome_mutation_rates"] = selection;
      const auto subset = MutationRateGenome::FromParams(subset_params);
      for (size_t i = 0; i < names.size(); ++i)
         assert(subset.Rate(names[i], 0.75) == ((mask & (1u << i)) ? 0.25 : 0.75));
   }

   // Each gene controls its own instruction operator, not just stored metadata.
   for (const auto& selected : names) {
      auto operator_params = p;
      operator_params["genome_mutation_rates"] = selected;
      for (const auto& name : names) operator_params[name] = 0.0;
      RegisterMachine specimen(parent, operator_params, tpg.state_, rng);
      specimen.mutation_rate_genome_ = MutationRateGenome::FromParams(operator_params);
      specimen.mutation_rate_genome_.genes[selected] = 0.999999;
      // Distinct instruction encodings make swaps observable.
      for (size_t i = 0; i < specimen.instructions_.size(); ++i)
         specimen.instructions_[i]->in1Idx_ = static_cast<int>(i);
      std::vector<std::string> before, after;
      for (auto* inst : specimen.instructions_) before.push_back(inst->ToString());
      specimen.Mutate(operator_params, tpg.state_, rng, tpg._ops);
      for (auto* inst : specimen.instructions_) after.push_back(inst->ToString());
      if (selected == "p_instructions_add") assert(after.size() == before.size() + 1);
      if (selected == "p_instructions_delete") assert(after.size() + 1 == before.size());
      if (selected == "p_instructions_swap") {
         assert(before != after);
         std::sort(before.begin(), before.end());
         std::sort(after.begin(), after.end());
         assert(before == after);
      }
      if (selected == "p_instructions_mutate") assert(before.size() == after.size() && before != after);
   }

   // Logistic-normal mutation has zero mean log-odds increment away from bounds.
   auto genome = MutationRateGenome::FromParams(p);
   for (double sigma : {0.1, 0.2, 0.3, 0.4, 0.5}) {
      p["genome_rate_sigma"] = sigma;
      double sum = 0.0, squares = 0.0;
      for (int i = 0; i < 20000; ++i) {
         genome.genes["p_instructions_add"] = 0.5;
         genome.Mutate(p, rng);
         const double value = genome.genes.at("p_instructions_add");
         const double delta = std::log(value / (1 - value));
         sum += delta; squares += delta * delta;
      }
      assert(std::abs(sum / 20000) < 0.03 * sigma);
      assert(std::abs(squares / 20000 - sigma * sigma) < 0.06 * sigma * sigma);
   }
   p["genome_rate_sigma"] = 1000.0;
   for (int i = 0; i < 1000; ++i) {
      genome.Mutate(p, rng);
      for (auto [name, value] : genome.genes)
         assert(std::isfinite(value) && value >= 0.000001 && value <= 0.999999);
   }
   for (const auto& selection : {"unknown", "pma", "p_instructions_add,", "p_instructions_add,p_instructions_add"}) {
      auto bad = p; bad["genome_mutation_rates"] = std::string(selection);
      Rejects([&] { MutationRateGenome::FromParams(bad); });
   }
   auto bad = p; bad["self_modifying"] = 1;
   Rejects([&] { MutationRateGenome::FromParams(bad); });
   bad = p; bad["genome_rate_min"] = 0.0;
   Rejects([&] { MutationRateGenome::FromParams(bad); });
   bad = p; bad["genome_rate_sigma"] = -1.0;
   Rejects([&] { MutationRateGenome::FromParams(bad); });
   Rejects([&] { genome.Decode(inherited, disabled); });
   Rejects([&] { genome.Decode("MRG1|p_instructions_add=nan", p); });
   std::cout << "Mutation-rate genome integration tests passed\n";
}
