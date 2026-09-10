#ifndef LGP_XPREDPREY_TASK_H
#define LGP_XPREDPREY_TASK_H

#include "TaskEnv.h"
#include "XPredPrey.h"

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class XPredPreyTask : public TaskEnv {
 public:
  enum Role : int { Predator = 0, Prey = 1 };

  explicit XPredPreyTask(std::unordered_map<std::string, std::any>& params);

  void reset(std::mt19937& rng, int episode);
  Results sim_step(const std::vector<double>& predator_action,
                   const std::vector<double>& prey_action);
  bool terminal() const;
  std::vector<double>& observation(Role role);
  int observationSize() const;
  bool captured() const { return simulator_->isCaptured(); }

 private:
  void refreshObservations();

  int max_steps_ = 2000;
  int done_ = 0;
  std::unique_ptr<XPredPrey> simulator_;
  std::vector<float> raw_observations_;
  std::vector<float> actions_;
  std::vector<double> role_observations_[2];
};

#endif
