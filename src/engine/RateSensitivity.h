#ifndef RateSensitivity_h
#define RateSensitivity_h

#include <array>
#include <deque>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "MemoryEigen.h"
#include "misc.h"
#include "state.h"

class EvalData;
class RegisterMachine;
class TPG;

/*******************************************************************************
 Numerical sensitivity of mutation rates (S2-S5) to each observation, measured
 during replay (rate_sensitivity=1).

 With rate_sensitivity_gaussian=1, each channel is tested separately with fresh
 N(0, (alpha * std)^2) noise at every timestep; other channels stay clean. Each alpha starts from the
 clean initial memory and carries its noisy state through the full recorded
 trajectory, applying normal episode resets. The standard deviations come
 from clean replay. Noise draws are paired across alphas and programs using
 rate_sensitivity_noise_seed. Outputs report per-step mutation-probability
 differences, not reward changes; environment dynamics are not re-simulated.

 In the legacy impulse mode (rate_sensitivity_gaussian=0):
 Replay records each root-team program's working memory and observation before
 every step. Afterwards, each step is re-executed from its recorded state with
 one observation shifted by +/- fraction * std (std over the replayed steps):

 - One-step: change in mutation probability from that single execution.
 - Carried-over (open loop): the perturbed state continues on the recorded
   observations until it equals the baseline state again, the evaluation ends,
   or rate_sensitivity_carry_horizon steps elapse (0 = no limit). Episode
   boundaries apply the same memory reset as team::InitMemory.
*******************************************************************************/
class RateSensitivity {
  public:
   void BeforeStep(TPG &tpg, EvalData &eval_data);
   void AfterStep(TPG &tpg, EvalData &eval_data);
   void Analyze(TPG &tpg, EvalData &eval_data);

  private:
   struct Snapshot {
      std::vector<std::deque<MatrixDynamic>> private_working;
      std::vector<std::deque<MatrixDynamic>> observation_working;
      std::vector<std::pair<size_t, std::mt19937>> rngs;
   };
   struct Step {
      int segment;  // Increments at every episode (InitMemory) boundary.
      int episode;
      int timestep;
      std::vector<double> obs;
      std::vector<Snapshot> before;  // One per analyzed program.
      std::vector<std::array<double, kSelfModifyingRegisterCount>> recorded_raw;
   };

   // Replay-only sensor noise uses its own RNG and observation buffer.
   state noisy_observation_{0};
   state *original_observation_ = nullptr;
   std::vector<double> noisy_buffer_;
   std::mt19937 noise_rng_;
   std::normal_distribution<double> standard_normal_{0.0, 1.0};
   int noise_episode_ = -1;
   size_t calibration_count_ = 0;
   std::vector<double> calibration_mean_, calibration_m2_;
   size_t rate_sample_count_ = 0;
   std::array<double, kSelfModifyingRegisterCount> rate_sum_{};
   std::array<double, kSelfModifyingRegisterCount> final_rates_{};
   bool active_ = false;
   long team_id_ = -1;
   int last_task_ = -1;
   int last_episode_ = -1;
   int segment_ = -1;
   std::vector<RegisterMachine *> programs_;
   std::vector<Step> steps_;

   bool Enabled(TPG &tpg, EvalData &eval_data);
   static Snapshot Capture(const RegisterMachine &prog);
   static void Restore(RegisterMachine &prog, const Snapshot &snapshot);
   static bool SameState(const RegisterMachine &prog, const Snapshot &snapshot);
   static std::array<double, kSelfModifyingRegisterCount> RawRates(
       const RegisterMachine &prog);
};

#endif
