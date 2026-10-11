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

class EvalData;
class RegisterMachine;
class TPG;

/*******************************************************************************
 Numerical sensitivity of mutation rates (S2-S5) to each observation, measured
 during replay (rate_sensitivity=1).

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
