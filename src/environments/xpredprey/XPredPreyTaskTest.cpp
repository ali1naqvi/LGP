#include "XPredPreyTask.h"

#include <any>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::unordered_map<std::string, std::any> parameters(int max_steps) {
  return {
      {"xpredprey_max_steps", max_steps},
      {"xpredprey_n_eval_train", 1},
      {"xpredprey_n_eval_validation", 1},
      {"xpredprey_n_eval_test", 1},
      {"xpredprey_asset_path", std::string(XPREDPREY_DEFAULT_ASSET_DIR)},
  };
}

}  // namespace

int main() {
  std::mt19937 rng(42);
  const std::vector<double> forward{1.0, 1.0};
  const std::vector<double> stopped{0.0, 0.0};

  {
    auto params = parameters(100);
    XPredPreyTask task(params);
    task.reset(rng, 0);
    TaskEnv::Results result{0.0, 0.0};
    while (!task.terminal()) result = task.sim_step(forward, stopped);
    if (!std::isfinite(result.r1) || !std::isfinite(result.r2) ||
        !task.captured() || result.r1 <= 0.0 || result.r2 >= 1.0 ||
        std::abs(result.r1 + result.r2 - 1.0) >= 1e-12) {
      std::cerr << "capture rewards are not complementary: predator="
                << result.r1 << " prey=" << result.r2 << '\n';
      return 1;
    }
  }

  {
    auto params = parameters(5);
    XPredPreyTask task(params);
    task.reset(rng, 0);
    TaskEnv::Results result{0.0, 0.0};
    while (!task.terminal()) result = task.sim_step(stopped, stopped);
    if (task.captured() || result.r1 != 0.0 || result.r2 != 1.0) {
      std::cerr << "timeout rewards are incorrect: predator=" << result.r1
                << " prey=" << result.r2 << '\n';
      return 1;
    }
  }
}
