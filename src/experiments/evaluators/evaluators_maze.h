#ifndef evaluators_maze_h
#define evaluators_maze_h

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

inline void EvalMaze(TPG& tpg, EvalData& eval_data) {
   FastSimEnv* task = dynamic_cast<FastSimEnv*>(eval_data.task);   
   task->reset(tpg.rngs_[AUX_SEED], eval_data.episode);
   eval_data.running_mean = 0.0;
   eval_data.n_prediction = 0;
   eval_data.obs = new state(task->GetObsSize());   
   eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable)); //not partially observable
   while (!task->terminal()) {
      tpg.GetAction(
          eval_data, tpg.rngs_[AUX_SEED],
          tpg.params_, tpg.prev_prog_history_); 
      if(std::any_cast<int>(tpg.params_["maze_discrete"])){
         int n_actions = std::any_cast<int>(tpg.params_["n_discrete_action"]);
         auto ctrl = WrapDiscreteActionFromRegister(eval_data, n_actions);
            TaskEnv::Results r = task->sim_step(ctrl);
            eval_data.stats_double[REWARD1_IDX] += r.r1;
            eval_data.AccumulateStepData();   
            if (std::any_cast<std::string>(tpg.params_["homeostatic_TD"]) == "step"){  
            tpg.ComputeIndReward(eval_data, tpg.params_, r.r1);}    
            eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable));  
      }
      else{
         auto ctrl = WrapVectorActionMuJoco(eval_data); // uses same as mujoco [-1,1]
            TaskEnv::Results r = task->sim_step(ctrl);
            eval_data.stats_double[REWARD1_IDX] += r.r1;  
            eval_data.AccumulateStepData();   
            tpg.ComputeIndReward(eval_data, tpg.params_, r.r1);    
            eval_data.obs->Set(task->GetObsVec(eval_data.partially_observable));  
      }
      //TODO(ali): cleanup (this is terrible)
  
   } 
   delete eval_data.obs;
}

#endif