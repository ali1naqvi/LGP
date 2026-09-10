#include "XPredPreyTask.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

#ifndef XPREDPREY_DEFAULT_ASSET_DIR
#define XPREDPREY_DEFAULT_ASSET_DIR "."
#endif

namespace {

template <typename T>
T parameterOr(const std::unordered_map<std::string, std::any>& params,
              const std::string& name, T fallback) {
  const auto found = params.find(name);
  if (found == params.end()) return fallback;
  return std::any_cast<T>(found->second);
}

float wheel(const std::vector<double>& action, std::size_t index) {
  if (index >= action.size() || !std::isfinite(action[index])) return 0.0F;
  return static_cast<float>(std::clamp(action[index], -1.0, 1.0));
}

}  // namespace

XPredPreyTask::XPredPreyTask(
    std::unordered_map<std::string, std::any>& params) {
  eval_type_ = "XPredPrey";
  n_eval_train_ = parameterOr<int>(params, "xpredprey_n_eval_train", 10);
  n_eval_validation_ =
      parameterOr<int>(params, "xpredprey_n_eval_validation", 5);
  n_eval_test_ = parameterOr<int>(params, "xpredprey_n_eval_test", 5);
  max_steps_ = parameterOr<int>(params, "xpredprey_max_steps", 2000);
  if (max_steps_ < 1) {
    throw std::invalid_argument("xpredprey_max_steps must be positive");
  }

  const std::string asset_directory = parameterOr<std::string>(
      params, "xpredprey_asset_path", XPREDPREY_DEFAULT_ASSET_DIR);
  simulator_ = std::make_unique<XPredPrey>(max_steps_, asset_directory);
  raw_observations_.resize(static_cast<std::size_t>(simulator_->ninputs) * 2U);
  actions_.resize(static_cast<std::size_t>(simulator_->noutputs) * 2U, 0.0F);
  for (auto& observations : role_observations_) {
    observations.resize(static_cast<std::size_t>(simulator_->ninputs));
  }
  state_.resize(static_cast<std::size_t>(simulator_->ninputs));
  state_po_ = state_;

  simulator_->copyObs(raw_observations_.data());
  simulator_->copyAct(actions_.data());
  simulator_->copyDone(&done_);
}

void XPredPreyTask::reset(std::mt19937& rng, int episode) {
  const std::uint32_t seed = rng() ^ static_cast<std::uint32_t>(episode);
  simulator_->seed(static_cast<int>(seed));
  done_ = 0;
  step_ = 0;
  reward = 0.0;
  simulator_->reset();
  refreshObservations();
}

TaskEnv::Results XPredPreyTask::sim_step(
    const std::vector<double>& predator_action,
    const std::vector<double>& prey_action) {
  actions_[0] = wheel(predator_action, 0);
  actions_[1] = wheel(predator_action, 1);
  actions_[2] = wheel(prey_action, 0);
  actions_[3] = wheel(prey_action, 1);

  const double predator_reward = simulator_->step();
  ++step_;
  refreshObservations();

  if (!terminal()) return {0.0, 0.0};
  const double prey_reward = 1.0 - predator_reward;
  return {predator_reward, prey_reward};
}

bool XPredPreyTask::terminal() const { return done_ != 0 || step_ >= max_steps_; }

std::vector<double>& XPredPreyTask::observation(Role role) {
  return role_observations_[static_cast<int>(role)];
}

int XPredPreyTask::observationSize() const { return simulator_->ninputs; }

void XPredPreyTask::refreshObservations() {
  const std::size_t width = static_cast<std::size_t>(simulator_->ninputs);
  for (std::size_t role = 0; role < 2; ++role) {
    std::copy_n(raw_observations_.begin() + role * width, width,
                role_observations_[role].begin());
  }
  state_ = role_observations_[Predator];
  state_po_ = state_;
}
