#ifndef evaluators_gradient_h
#define evaluators_gradient_h

#include <iostream>
#include "fastsim.hpp"
#include "FastSimEnvironment.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <thread>
#include "EvalData.h"
#include "TPG.h"

inline void EvalGradient(TPG& tpg, EvalData& eval_data) {
   FastSimEnv* task = dynamic_cast<FastSimEnv*>(eval_data.task);
   bool evolving_observations = false;
   auto evolving_it = tpg.params_.find("gradient_evolving_observations");
   if (evolving_it != tpg.params_.end()) {
      evolving_observations = std::any_cast<int>(evolving_it->second) != 0;
   }
   if (evolving_observations && eval_data.tm != nullptr) {
      auto members = eval_data.tm->CopyMembers();
      if (members.size() == 1) {
         RegisterMachine* program = *members.begin();
         int observation_count = static_cast<int>(
             program->private_memory_[MemoryEigen::kVectorType_]->memory_size_);
         task->SetResourceObservationCount(observation_count);
      }
   }

   task->reset(tpg.rngs_[AUX_SEED], eval_data.episode);
   eval_data.running_mean = 0.0;
   eval_data.n_prediction = 0;
   eval_data.obs = new state(task->GetObsSize());
   eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable));
   while (!task->terminal()) {
      tpg.GetAction(
          eval_data, tpg.rngs_[AUX_SEED],
          tpg.params_, tpg.prev_prog_history_);
      if (std::any_cast<int>(tpg.params_["gradient_discrete"])) {
         int n_actions = std::any_cast<int>(tpg.params_["n_discrete_action"]);
         int multi_output = 1;
         if (tpg.params_.find("gradient_multi_output") != tpg.params_.end()) {
            multi_output = std::any_cast<int>(tpg.params_["gradient_multi_output"]);
         }
         // When enabled, use every entry in vector register 1. Its length is
         // the winning program's current vector-memory size, so programs that
         // evolve that size also evolve their number of sequential outputs.
         bool dynamic_multi_output = false;
         if (tpg.params_.find("gradient_multi_output_dynamic") !=
             tpg.params_.end()) {
            dynamic_multi_output =
                std::any_cast<int>(tpg.params_["gradient_multi_output_dynamic"]) != 0;
         }

         if (multi_output > 1 || dynamic_multi_output) {
            auto actions = WrapDiscreteActionsFromVector(eval_data, n_actions);
            for (size_t i = 0; i < actions.size() && !task->terminal(); ++i) {
               std::vector<double> ctrl{ static_cast<double>(actions[i]) };
               TaskEnv::Results r = task->sim_step(ctrl);
               eval_data.stats_double[REWARD1_IDX] += r.r1;
               eval_data.AccumulateStepData();
               tpg.LogReplaySelfModifyingRates(eval_data);
               if (std::any_cast<std::string>(tpg.params_["homeostatic_TD"]) == "step") {
                  tpg.ComputeIndReward(eval_data, tpg.params_, r.r1);
               }
            }
            eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable));
         } else {
            auto ctrl = WrapDiscreteActionFromRegister(eval_data, n_actions);
            TaskEnv::Results r = task->sim_step(ctrl);
            eval_data.stats_double[REWARD1_IDX] += r.r1;
            eval_data.AccumulateStepData();
            tpg.LogReplaySelfModifyingRates(eval_data);
            if (std::any_cast<std::string>(tpg.params_["homeostatic_TD"]) == "step") {
               tpg.ComputeIndReward(eval_data, tpg.params_, r.r1);
            }
            eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable));
         }
      } else {
         auto ctrl = WrapVectorActionMuJoco(eval_data); // continuous turn rate + forward speed
         ctrl.resize(2, 0.0);
         TaskEnv::Results r = task->sim_step(ctrl);
         eval_data.stats_double[REWARD1_IDX] += r.r1;
         eval_data.AccumulateStepData();
         tpg.LogReplaySelfModifyingRates(eval_data);
         if (std::any_cast<std::string>(tpg.params_["homeostatic_TD"]) == "step") {
            tpg.ComputeIndReward(eval_data, tpg.params_, r.r1);
         }
         eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable));
      }
   }
   delete eval_data.obs;
}

#endif
