#ifndef EVALUATORS_XPREDPREY_H
#define EVALUATORS_XPREDPREY_H

#include "ActionWrappers.h"
#include "EvalData.h"
#include "TPG.h"
#include "XPredPreyTask.h"
#include "XPredPreySchedule.h"
#include "XPredPreyLogging.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

inline std::vector<double> XPredPreyAction(
    TPG& tpg, EvalData& data,
    std::tuple<long, double, double>& previous_program) {
  data.instruction_count = 0;
  data.team_path.clear();
  data.tm->GetAction(data, tpg.rngs_[AUX_SEED], tpg.params_, previous_program);
  auto action = WrapVectorActionMuJoco(data);
  action.resize(2, 0.0);
  return action;
}

inline void EvalXPredPreyEncounter(TPG& tpg, EvalData& candidate, team* opponent,
                                  const XPredPreyEncounter* encounter) {
  auto* task = dynamic_cast<XPredPreyTask*>(candidate.task);
  if (task == nullptr) {
    throw std::runtime_error("XPredPrey evaluator received the wrong task type");
  }

  const int candidate_role = candidate.tm->populationRole();
  const int opponent_role = 1 - candidate_role;
  if (!opponent || opponent->populationRole() != opponent_role)
    throw std::runtime_error("XPredPrey encounter requires opposite roles");

  if (encounter) tpg.rngs_[AUX_SEED].seed(encounter->seed);

  task->reset(tpg.rngs_[AUX_SEED], candidate.episode);
  candidate.n_prediction = 0;
  candidate.timestep = 0;
  candidate.sample = 0;
  candidate.pred_error = 0;
  candidate.running_mean = 0;
  candidate.obs = new state(task->observationSize());
  candidate.obs->Set(task->observation(
      static_cast<XPredPreyTask::Role>(candidate_role)));

  EvalData opponent_data = tpg.InitEvalData();
  opponent_data.tm = opponent;
  opponent_data.task = task;
  opponent_data.team_map = candidate.team_map;
  opponent_data.teams = candidate.teams;
  opponent_data.episode = candidate.episode;
  opponent_data.obs = new state(task->observationSize());
  opponent_data.obs->Set(task->observation(
      static_cast<XPredPreyTask::Role>(opponent_role)));

  candidate.tm->InitMemory(tpg.team_map_, tpg.params_);
  opponent->InitMemory(tpg.team_map_, tpg.params_);
  using Snapshot = std::map<RegisterMachine*, XPredPreyRates, RegisterMachineIdComp>;
  const auto snapshot = [&](team* tm) {
    std::set<team*, teamIdComp> visited;
    std::set<RegisterMachine*, RegisterMachineIdComp> programs;
    tm->GetAllNodes(tpg.team_map_, visited, programs);
    Snapshot values;
    for (auto* program : programs) values.emplace(program, ReadXPredPreyRates(*program));
    return values;
  };
  const auto candidate_before = snapshot(candidate.tm);
  const auto opponent_before = snapshot(opponent);
  std::tuple<long, double, double> candidate_history{-1, 1.0, 1.0};
  std::tuple<long, double, double> opponent_history{-1, 1.0, 1.0};

  while (!task->terminal()) {
    const auto candidate_action =
        XPredPreyAction(tpg, candidate, candidate_history);
    const auto opponent_action =
        XPredPreyAction(tpg, opponent_data, opponent_history);

    const auto result = candidate_role == POPULATION_ROLE_PREDATOR
        ? task->sim_step(candidate_action, opponent_action)
        : task->sim_step(opponent_action, candidate_action);
    const double role_reward = candidate_role == POPULATION_ROLE_PREDATOR
                                   ? result.r1
                                   : result.r2;
    candidate.AccumulateStepData();
    candidate.stats_double[REWARD1_IDX] += role_reward;
    tpg.ComputeIndReward(candidate, tpg.params_, role_reward);
    // The opponent has the same execution lifetime as a candidate, including
    // its timestep and learning bookkeeping. Only candidate fitness is scored.
    const double opponent_reward = candidate_role == POPULATION_ROLE_PREDATOR
        ? result.r2 : result.r1;
    opponent_data.AccumulateStepData();
    opponent_data.stats_double[REWARD1_IDX] += opponent_reward;
    tpg.ComputeIndReward(opponent_data, tpg.params_, opponent_reward);

    candidate.obs->Set(task->observation(
        static_cast<XPredPreyTask::Role>(candidate_role)));
    opponent_data.obs->Set(task->observation(
        static_cast<XPredPreyTask::Role>(opponent_role)));
  }

  candidate.tm->HebbianMap.calculatePlasticity(
      0.0, candidate.tm->members_run_.size());
  opponent->HebbianMap.calculatePlasticity(0.0, opponent->members_run_.size());
  if (encounter) {
    const auto log = [&](const EvalData& data, const Snapshot& before,
                         long opponent_id, const char* appearance, int count) {
      for (const auto& [program, rates] : before) {
        std::ostringstream row;
        row << tpg.seeds_[TPG_SEED] << ',' << tpg.GetState("t_current") << ','
            << tpg.GetState("phase") << ',' << encounter->id << ',' << encounter->seed
            << ',' << encounter->episode << ',' << data.tm->id_ << ','
            << XPredPreyRoleName(data.tm->populationRole()) << ',' << opponent_id
            << ',' << appearance << ',' << count << ',' << program->id_ << ','
            << program->self_modifying_ << ',' << data.timestep << ','
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << data.stats_double[REWARD1_IDX] << ','
            << (task->captured() ? "capture" : "timeout") << ','
            << program->instructions_.size() << ',' << program->instructions_effective_.size();
        const auto after = ReadXPredPreyRates(*program);
        WriteXPredPreyRates(row, rates);
        WriteXPredPreyRates(row, after);
        WriteXPredPreyRates(row, ReadXPredPreyRates(*program, true));
        WriteXPredPreyRates(row, rates, true);
        WriteXPredPreyRates(row, after, true);
        candidate.eval_result += "X:" + row.str() + "\n";
      }
    };
    log(candidate, candidate_before, opponent->id_, "candidate", encounter->candidate_count);
    log(opponent_data, opponent_before, candidate.tm->id_, "opponent", encounter->opponent_count);
  }
  delete opponent_data.obs;
  delete candidate.obs;
  candidate.obs = nullptr;
}

// Replay one selected candidate without changing the training scheduler.
inline void EvalXPredPrey(TPG& tpg, EvalData& candidate) {
  std::vector<team*> opponents;
  for (auto* tm : candidate.teams) {
    if (tm->populationRole() == 1 - candidate.tm->populationRole()) opponents.push_back(tm);
  }
  if (opponents.empty()) throw std::runtime_error("XPredPrey requires an opposing population");
  std::sort(opponents.begin(), opponents.end(), teamIdComp());
  EvalXPredPreyEncounter(tpg, candidate,
      opponents[(candidate.episode + candidate.tm->id_) % opponents.size()], nullptr);
}

inline void EvalXPredPreySchedule(TPG& tpg, EvalData& data) {
  const auto schedule = XPredPreyEncounter::decode(data.checkpointString);
  if (schedule.empty()) throw std::runtime_error("Missing central XPredPrey schedule");
  for (const auto& encounter : schedule) {
    data.tm = tpg.team_map_.at(encounter.candidate);
    data.episode = encounter.episode;
    EvalXPredPreyEncounter(tpg, data, tpg.team_map_.at(encounter.opponent), &encounter);
    tpg.FinalizeStepData(data);
  }
  // Validation and test run on this worker's disposable copy. Their final
  // outputs must never overwrite the master's training phenotype.
  if (tpg.GetState("phase") == _TRAIN_PHASE)
    tpg.AppendSelfModifyingRates(data.eval_result, data.teams);
}

#endif
