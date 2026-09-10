#pragma once

#include "evaluators_xpredprey.h"
#include <boost/mpi.hpp>

inline std::vector<team*> GetXPredPreyTeamsToEval(TPG& tpg, TaskEnv* task) {
  if (tpg.GetParam<int>("keep_old_outcomes"))
    throw std::runtime_error("Balanced XPredPrey scheduling requires keep_old_outcomes: 0");
  auto roots = tpg.GetRootTeamsInVec();
  if (tpg.GetState("phase") == _TEST_PHASE) {
    roots.clear();
    for (const std::string role : {"predator", "prey"}) {
      if (!tpg.haveEliteTeam(role, tpg.GetParam<int>("fit_mode"), _VALIDATION_PHASE))
        throw std::runtime_error("XPredPrey testing requires both validation champions");
      auto* champion = tpg._eliteTeamPS[role][tpg.GetParam<int>("fit_mode")][_VALIDATION_PHASE];
      if (!champion) throw std::runtime_error("XPredPrey validation champion is null");
      roots.push_back(champion);
    }
  }
  std::vector<team*> result;
  for (auto* tm : roots) {
    tm->resetOutcomes(tpg.GetState("phase"));
    tm->_n_eval = task->GetNumEval(tpg.GetState("phase"));
    if (tm->_n_eval > 0) result.push_back(tm);
  }
  return result;
}

inline std::string XPredPreyEvaluationPacket(TPG& tpg, std::vector<team*>& teams) {
  if (teams.empty()) return "x";
  std::vector<long> populations[2];
  const int episodes = teams.front()->_n_eval;
  for (auto* tm : teams) {
    const int role = tm->populationRole();
    if (role != POPULATION_ROLE_PREDATOR && role != POPULATION_ROLE_PREY)
      throw std::runtime_error("XPredPrey candidate has no population role");
    populations[role].push_back(tm->id_);
    if (tm->_n_eval != episodes)
      throw std::runtime_error("XPredPrey requires equal candidate evaluation budgets");
  }
  const auto schedule = MakeXPredPreySchedule(populations[0], populations[1],
      episodes, tpg.seeds_[AUX_SEED], tpg.GetState("t_current"), tpg.GetState("phase"));
  std::string packet = tpg.WriteMPICheckpoint(teams);
  for (const auto& encounter : schedule) packet += encounter.encode();
  return packet;
}

inline void EvaluateXPredPreyMPI(TPG& tpg, boost::mpi::communicator& world, TaskEnv* task) {
  if (world.size() < 2)
    throw std::runtime_error("XPredPrey needs at least one MPI evaluator (mpirun -np 2)");
  auto teams = GetXPredPreyTeamsToEval(tpg, task);
  const auto packet = XPredPreyEvaluationPacket(tpg, teams);
  // Policy memory and team learning state are part of an encounter lifetime.
  // Keep the full ordered phase on one worker until all runtime state can be
  // synchronized safely between parallel matches. Worker count cannot change
  // opponents, encounter order, seeds, or mutation-rate exposure.
  world.send(1, 0, packet);
  for (int rank = 2; rank < world.size(); ++rank) world.send(rank, 0, std::string("x"));
  std::vector<std::string> results;
  boost::mpi::gather(world, std::string(), results, 0);
  for (size_t rank = 1; rank < results.size(); ++rank)
    if (!results[rank].empty()) tpg.DecodeEvalResultString(results[rank]);
  if (tpg.xpredprey_encounter_log_.is_open()) tpg.xpredprey_encounter_log_.flush();
  if (tpg.xpredprey_reproduction_log_.is_open()) tpg.xpredprey_reproduction_log_.flush();
  tpg.oss << "xpredprey_schedule generation " << tpg.GetState("t_current")
          << " phase " << tpg.GetState("phase") << " evaluator_workers "
          << (teams.empty() ? 0 : 1) << " available_workers " << world.size() - 1 << '\n';
}

inline void XPredPreyEvaluatorMPI(TPG& tpg, boost::mpi::communicator& world,
                                 const std::vector<TaskEnv*>& tasks) {
  for (;;) {
    std::string packet;
    world.recv(0, 0, packet);
    if (packet == "done") return;
    std::string result;
    if (packet != "x") {
      tpg.ReadCheckpoint(-1, _TRAIN_PHASE, true, packet);
      auto data = tpg.InitEvalData();
      data.world_rank = world.rank();
      data.world_size = world.size();
      data.checkpointString = packet;
      data.teams = tpg.GetRootTeamsInVec();
      data.team_map = tpg.team_map_;
      data.task = tasks.at(tpg.GetState("active_task"));
      EvalXPredPreySchedule(tpg, data);
      result = std::move(data.eval_result);
    }
    // Even idle ranks join every collective and remain available next phase.
    boost::mpi::gather(world, result, 0);
  }
}
