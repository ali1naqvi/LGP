#include "TPG.h"
#include "Acrobot.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
   Acrobot default_environment;
   assert(default_environment.max_step_ == 500);
   Acrobot environment(7, 3, 2, 1);
   assert(environment.max_step_ == 7);
   assert(environment.GetNumEval(_TRAIN_PHASE) == 3);
   assert(environment.GetNumEval(_VALIDATION_PHASE) == 2);
   assert(environment.GetNumEval(_TEST_PHASE) == 1);
   assert(!environment.DiscreteActions());
   std::mt19937 rng(42);
   environment.Reset(rng);
   assert(environment.GetObsVec(false).size() == 4);
   for (int step = 0; step < 7; ++step) {
      environment.Update(0, 0.0, rng);
      for (double observation : environment.GetObsVec(false))
         assert(std::isfinite(observation));
   }
   assert(environment.Terminal());
   assert(argc == 4); // Acrobot config, source checkpoint, and expected variant.
   const auto config = std::filesystem::absolute(argv[1]);
   const auto checkpoint = std::filesystem::absolute(argv[2]);
   TPG tpg;
   std::vector<std::string> arguments = {
       "test", "parameters_file=" + config.string(), "seed_tpg=1",
       "start_from_checkpoint=1", "checkpoint_in_t=1000",
       "checkpoint_in_phase=0"};
   std::vector<char*> pointers;
   for (auto& value : arguments) pointers.push_back(value.data());
   tpg.SetParams(pointers.size(), pointers.data());
   assert(tpg.GetParam<std::string>("active_tasks") == "Acrobot");
   assert(tpg.GetParam<std::string>("n_input") == "4");
   assert(tpg.GetParam<int>("continuous_output") == 1);
   assert(tpg.GetParam<int>("n_discrete_action") == 0);
   assert(tpg.GetParam<int>("partially_observable") == 0);
   assert(tpg.GetParam<int>("acrobot_max_timesteps") == 500);
   assert(tpg.GetParam<int>("acrobot_n_eval_train") == 20);
   assert(tpg.GetParam<int>("acrobot_n_eval_validation") == 5);
   assert(tpg.GetParam<int>("validation_mod") == 100);
   assert(tpg.GetParam<int>("validation_episode_seed_offset") == 10000);
   assert(tpg.GetParam<int>("test_episode_seed_offset") == 20000);
   assert(tpg.GetParam<int>("test_mod") == 0);
   assert(tpg.GetParam<int>("memory_size") == 1);
   assert(tpg.GetParam<double>("p_memory_slots") == 0.0);
   int enabled_ops = 0;
   for (size_t op = 0; op < tpg._ops.size(); ++op) {
      if (!tpg._ops[op]) continue;
      ++enabled_ops;
      for (auto type : instruction::op_signatures_[op])
         assert(type == MemoryEigen::kScalarType_);
   }
   assert(enabled_ops > 0);
   const std::string variant = argv[3];
   const auto genes = MutationRateGenome::FromParams(tpg.params_).genes;
   assert(genes.size() == (variant == "fixed_individual" ? 4 : 0));
   assert(tpg.GetParam<int>("self_modifying") ==
          (variant == "execution_modified_rates" ? 1 : 0));
   tpg.state_["n_task"] = 1;
   tpg.n_input_ = {4};
   assert(tpg.GetState("t_start") == 1001);

   std::ifstream input(checkpoint);
   const std::string original((std::istreambuf_iterator<char>(input)), {});
   tpg.ReadCheckpoint(1000, 0, true, original);
   assert(tpg.GetState("t_current") == 1000);
   std::map<long, std::string> programs, memories;
   for (auto& [id, program] : tpg.program_pop_) {
      programs[id] = program->ToString(false);
      memories[id] = program->ToStringMemory();
   }
   const auto teams = tpg.team_pop_.size();
   assert(teams > 0 && !programs.empty());

   std::filesystem::create_directories("checkpoints");
   std::filesystem::copy_file(checkpoint, "checkpoints/cp.1000.1.0.rslt",
       std::filesystem::copy_options::overwrite_existing);
   tpg.ResumeTrainingFromCheckpoint();
   assert(tpg.GetState("t_current") == 1001);
   assert(tpg.GetState("phase") == _TRAIN_PHASE);
   assert(tpg.GetParam<std::string>("active_tasks") == "Acrobot");
   assert(tpg.n_input_ == std::vector<int>{4});
   assert(tpg.team_pop_.size() == teams);
   assert(tpg.program_pop_.size() == programs.size());
   for (auto& [id, program] : tpg.program_pop_) {
      assert(program->ToString(false) == programs.at(id));
      assert(program->ToStringMemory() == memories.at(id));
   }
   for (auto* tm : tpg.GetRootTeamsInVec()) {
      // Source scores cannot be used for selection on Acrobot.
      assert(tm->numOutcomes(_TRAIN_PHASE, 0) == 0);
      assert(tm->numOutcomes(_VALIDATION_PHASE, 0) == 0);
   }
   tpg.finalize();
   std::cout << variant << " " << config.filename()
             << ": scalar-only config, budgets/dynamics pass; population/rates preserved; starts at 1001\n";
}
