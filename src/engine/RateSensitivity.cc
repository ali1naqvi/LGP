#include "RateSensitivity.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

#include "EvalData.h"
#include "RegisterMachine.h"
#include "TPG.h"
#include "instruction.h"
#include "state.h"
#include "team.h"

namespace {

bool IsRandomOp(int op) {
   return op >= instruction::SCALAR_UNIFORM_SET_OP_ &&
          op <= instruction::MATRIX_GAUSSIAN_SET_OP_;
}

std::vector<double> ParseFractions(const std::string &text) {
   std::vector<double> fractions;
   std::stringstream ss(text);
   std::string item;
   while (std::getline(ss, item, ',')) {
      if (item.empty()) continue;
      const double value = std::stod(item);
      if (!(value > 0.0)) {
         throw std::invalid_argument("rate_sensitivity_fractions must be positive");
      }
      fractions.push_back(value);
   }
   if (fractions.empty()) {
      throw std::invalid_argument("rate_sensitivity_fractions is empty");
   }
   return fractions;
}

const char *kRateNames[kSelfModifyingRegisterCount] = {"swap", "delete", "add",
                                                       "mutate"};

}  // namespace

bool RateSensitivity::Enabled(TPG &tpg, EvalData &eval_data) {
   if (!tpg.HaveParam("rate_sensitivity") ||
       tpg.GetParam<int>("rate_sensitivity") == 0 ||
       tpg.GetParam<int>("replay") == 0 || eval_data.tm == nullptr) {
      return false;
   }
   if (team_id_ == eval_data.tm->id_) return active_;
   team_id_ = eval_data.tm->id_;
   active_ = false;
   steps_.clear();
   programs_.clear();
   if (tpg.GetParam<int>("dynamic_lgp") != 0) {
      std::cerr << "rate_sensitivity: dynamic_lgp programs are unsupported\n";
      return false;
   }
   if (eval_data.obs == nullptr || eval_data.obs->scalarStateType != 0) {
      std::cerr << "rate_sensitivity: requires double-valued observations\n";
      return false;
   }
   // Every root-team member executes at every step. Programs in child teams
   // execute only when visited, so their per-step state is not reconstructed.
   for (auto *prog : eval_data.tm->members_) {
      if (prog->self_modifying_) programs_.push_back(prog);
   }
   if (programs_.empty()) {
      std::cerr << "rate_sensitivity: team " << team_id_
                << " has no self-modifying root programs\n";
      return false;
   }
   active_ = true;
   return true;
}

RateSensitivity::Snapshot RateSensitivity::Capture(const RegisterMachine &prog) {
   Snapshot snapshot;
   for (auto *memory : prog.private_memory_) {
      snapshot.private_working.push_back(memory->working_memory_);
   }
   for (auto *memory : prog.observation_memory_buff_) {
      snapshot.observation_working.push_back(memory->working_memory_);
   }
   for (size_t i = 0; i < prog.instructions_effective_.size(); ++i) {
      const auto *istr = prog.instructions_effective_[i];
      if (IsRandomOp(istr->op_)) snapshot.rngs.emplace_back(i, istr->rng_);
   }
   return snapshot;
}

void RateSensitivity::Restore(RegisterMachine &prog, const Snapshot &snapshot) {
   for (size_t m = 0; m < snapshot.private_working.size(); ++m) {
      prog.private_memory_[m]->working_memory_ = snapshot.private_working[m];
   }
   for (size_t m = 0; m < snapshot.observation_working.size(); ++m) {
      prog.observation_memory_buff_[m]->working_memory_ =
          snapshot.observation_working[m];
   }
   for (const auto &[index, rng] : snapshot.rngs) {
      prog.instructions_effective_[index]->rng_ = rng;
   }
}

bool RateSensitivity::SameState(const RegisterMachine &prog,
                                const Snapshot &snapshot) {
   auto same = [](const std::deque<MatrixDynamic> &a,
                  const std::deque<MatrixDynamic> &b) {
      if (a.size() != b.size()) return false;
      for (size_t i = 0; i < a.size(); ++i) {
         if (a[i].rows() != b[i].rows() || a[i].cols() != b[i].cols() ||
             a[i] != b[i]) {
            return false;
         }
      }
      return true;
   };
   for (size_t m = 0; m < snapshot.private_working.size(); ++m) {
      if (!same(prog.private_memory_[m]->working_memory_,
                snapshot.private_working[m])) {
         return false;
      }
   }
   for (size_t m = 0; m < snapshot.observation_working.size(); ++m) {
      if (!same(prog.observation_memory_buff_[m]->working_memory_,
                snapshot.observation_working[m])) {
         return false;
      }
   }
   return true;
}

std::array<double, kSelfModifyingRegisterCount> RateSensitivity::RawRates(
    const RegisterMachine &prog) {
   std::array<double, kSelfModifyingRegisterCount> raw{};
   const auto *scalar = prog.private_memory_[MemoryEigen::kScalarType_];
   for (size_t r = 0; r < kSelfModifyingRegisterCount; ++r) {
      raw[r] = scalar->working_memory_[kSelfModifyingFirstRegister + r](0, 0);
   }
   return raw;
}

void RateSensitivity::BeforeStep(TPG &tpg, EvalData &eval_data) {
   if (!Enabled(tpg, eval_data)) return;
   const int task = tpg.GetState("active_task");
   if (task != last_task_ || eval_data.episode != last_episode_) {
      ++segment_;
      last_task_ = task;
      last_episode_ = eval_data.episode;
   }
   Step step;
   step.segment = segment_;
   step.episode = eval_data.episode;
   step.timestep = eval_data.timestep;
   const auto &obs = *eval_data.obs->getStatePointerDouble();
   step.obs.assign(obs.begin(), obs.begin() + eval_data.obs->dim_);
   for (auto *prog : programs_) step.before.push_back(Capture(*prog));
   steps_.push_back(std::move(step));
}

void RateSensitivity::AfterStep(TPG &tpg, EvalData &eval_data) {
   if (!active_ || steps_.empty() || eval_data.tm == nullptr ||
       eval_data.tm->id_ != team_id_) {
      return;
   }
   auto &step = steps_.back();
   if (!step.recorded_raw.empty()) return;
   for (auto *prog : programs_) step.recorded_raw.push_back(RawRates(*prog));
}

void RateSensitivity::Analyze(TPG &tpg, EvalData &eval_data) {
   if (!active_ || steps_.empty()) return;
   auto &params = tpg.params_;
   const auto fractions =
       ParseFractions(tpg.GetParam<std::string>("rate_sensitivity_fractions"));
   const int horizon = tpg.GetParam<int>("rate_sensitivity_carry_horizon");
   const size_t n_steps = steps_.size();
   const size_t dim = steps_.front().obs.size();

   std::vector<double> mean(dim, 0.0), sd(dim, 0.0);
   for (const auto &step : steps_)
      for (size_t i = 0; i < dim; ++i) mean[i] += step.obs[i];
   for (auto &m : mean) m /= n_steps;
   for (const auto &step : steps_)
      for (size_t i = 0; i < dim; ++i)
         sd[i] += (step.obs[i] - mean[i]) * (step.obs[i] - mean[i]);
   for (auto &s : sd) s = std::sqrt(s / n_steps);

   std::filesystem::create_directories("logs/misc");
   std::ostringstream prefix;
   prefix << "logs/misc/rate_sensitivity." << tpg.GetParam<int>("seed_tpg") << '.'
          << team_id_;
   std::ofstream observations(prefix.str() + ".observations.csv");
   std::ofstream one_step(prefix.str() + ".one_step.csv");
   std::ofstream carried(prefix.str() + ".carried.csv");
   if (!observations || !one_step || !carried) {
      throw std::runtime_error("rate_sensitivity: cannot open output files in logs/misc");
   }
   for (auto *out : {&observations, &one_step, &carried}) {
      *out << std::setprecision(std::numeric_limits<double>::max_digits10);
   }
   observations << "program,observation,mean,std,n_steps,n_episodes,"
                   "baseline_mismatches,boundary_mismatches\n";
   one_step << "program,step,episode,timestep,observation,fraction,delta,"
               "observation_value";
   carried << "program,step,episode,timestep,observation,fraction,delta,status,"
              "steps_followed,last_rate_change_lag";
   for (const char *name : kRateNames) {
      one_step << ",base_" << name << ",dp_" << name;
   }
   for (const char *name : kRateNames) carried << ",final_dp_" << name;
   for (const char *name : kRateNames) carried << ",max_abs_dp_after_" << name;
   one_step << '\n';
   carried << '\n';

   std::vector<double> obs_buffer;
   state perturbed_state(dim);
   perturbed_state.Set(obs_buffer);
   state *saved_obs = eval_data.obs;
   eval_data.obs = &perturbed_state;
   auto run = [&](RegisterMachine &prog, const std::vector<double> &obs,
                  int timestep) {
      obs_buffer = obs;
      size_t depth = 1;
      bool verbose = false;
      prog.Run(eval_data, timestep, depth, verbose);
   };
   auto probabilities = [&](RegisterMachine &prog) {
      return prog.MutationProbabilities(params);
   };

   for (size_t k = 0; k < programs_.size(); ++k) {
      auto &prog = *programs_[k];
      std::vector<std::array<double, kSelfModifyingRegisterCount>> base_raw(n_steps);
      std::vector<std::array<double, kSelfModifyingRegisterCount>> base_p(n_steps);
      size_t baseline_mismatches = 0, boundary_mismatches = 0;
      for (size_t t = 0; t < n_steps; ++t) {
         Restore(prog, steps_[t].before[k]);
         run(prog, steps_[t].obs, steps_[t].timestep);
         base_raw[t] = RawRates(prog);
         base_p[t] = probabilities(prog);
         if (!steps_[t].recorded_raw.empty() && base_raw[t] != steps_[t].recorded_raw[k]) {
            ++baseline_mismatches;
         }
         if (t + 1 < n_steps) {
            if (steps_[t + 1].segment != steps_[t].segment) {
               team::InitProgramMemory(&prog, params);
            }
            if (!SameState(prog, steps_[t + 1].before[k])) ++boundary_mismatches;
         }
      }
      std::set<int> segments;
      for (const auto &step : steps_) segments.insert(step.segment);
      for (size_t i = 0; i < dim; ++i) {
         observations << prog.id_ << ',' << i << ',' << mean[i] << ',' << sd[i] << ','
                      << n_steps << ',' << segments.size() << ','
                      << baseline_mismatches << ',' << boundary_mismatches << '\n';
      }
      if (baseline_mismatches || boundary_mismatches) {
         std::cerr << "rate_sensitivity: program " << prog.id_
                   << " baseline does not reproduce replay (" << baseline_mismatches
                   << " rate mismatches, " << boundary_mismatches
                   << " state mismatches)\n";
      }

      for (size_t t = 0; t < n_steps; ++t) {
         const auto &step = steps_[t];
         for (size_t i = 0; i < dim; ++i) {
            if (sd[i] == 0.0) continue;
            for (double fraction : fractions) {
               for (int sign : {-1, 1}) {
                  const double delta = sign * fraction * sd[i];
                  auto obs = step.obs;
                  obs[i] += delta;
                  Restore(prog, step.before[k]);
                  run(prog, obs, step.timestep);
                  const auto p = probabilities(prog);
                  one_step << prog.id_ << ',' << t << ',' << step.episode << ','
                           << step.timestep << ',' << i << ',' << fraction << ','
                           << delta << ',' << step.obs[i];
                  for (size_t r = 0; r < kSelfModifyingRegisterCount; ++r) {
                     one_step << ',' << base_p[t][r] << ',' << p[r] - base_p[t][r];
                  }
                  one_step << '\n';

                  int last_change = RawRates(prog) != base_raw[t] ? 0 : -1;
                  std::array<double, kSelfModifyingRegisterCount> max_after{};
                  std::array<double, kSelfModifyingRegisterCount> final_dp{};
                  const char *status = "converged";
                  size_t s = t + 1;
                  for (;; ++s) {
                     if (s == n_steps || (horizon > 0 && s - t > size_t(horizon))) {
                        status = s == n_steps ? "end" : "horizon";
                        const auto final_p = probabilities(prog);
                        for (size_t r = 0; r < kSelfModifyingRegisterCount; ++r) {
                           final_dp[r] = final_p[r] - base_p[s - 1][r];
                        }
                        break;
                     }
                     if (steps_[s].segment != steps_[s - 1].segment) {
                        team::InitProgramMemory(&prog, params);
                     }
                     if (SameState(prog, steps_[s].before[k])) break;
                     for (const auto &[index, rng] : steps_[s].before[k].rngs) {
                        prog.instructions_effective_[index]->rng_ = rng;
                     }
                     run(prog, steps_[s].obs, steps_[s].timestep);
                     if (RawRates(prog) != base_raw[s]) {
                        last_change = int(s - t);
                        const auto carried_p = probabilities(prog);
                        for (size_t r = 0; r < kSelfModifyingRegisterCount; ++r) {
                           max_after[r] = std::max(
                               max_after[r], std::abs(carried_p[r] - base_p[s][r]));
                        }
                     }
                  }
                  carried << prog.id_ << ',' << t << ',' << step.episode << ','
                          << step.timestep << ',' << i << ',' << fraction << ','
                          << delta << ',' << status << ',' << s - t - 1 << ','
                          << last_change;
                  for (double v : final_dp) carried << ',' << v;
                  for (double v : max_after) carried << ',' << v;
                  carried << '\n';
               }
            }
         }
         if ((t + 1) % 500 == 0 || t + 1 == n_steps) {
            std::cout << "rate_sensitivity: program " << prog.id_ << " analyzed "
                      << t + 1 << '/' << n_steps << " steps" << std::endl;
         }
      }
      // Leave the program in its post-replay state.
      Restore(prog, steps_.back().before[k]);
      run(prog, steps_.back().obs, steps_.back().timestep);
   }
   eval_data.obs = saved_obs;
   std::cout << "rate_sensitivity: wrote " << prefix.str()
             << ".{observations,one_step,carried}.csv" << std::endl;
}
