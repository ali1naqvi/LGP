#include "RegisterMachine.h"
#include "EvalData.h"
#include "misc.h"

#include <cmath>
#include <optional>
#include <stacktrace>

namespace {

bool SelfModifyingEnabled(
    const std::unordered_map<std::string, std::any> &params) {
   auto it = params.find("self_modifying");
   return it != params.end() && std::any_cast<int>(it->second) != 0;
}

bool LamarckismEvolvedConstantsEnabled(
    const std::unordered_map<std::string, std::any> &params) {
   auto it = params.find("lamarkism_evolved_constants");
   return it != params.end() && std::any_cast<int>(it->second) != 0;
}

bool SelfModifyingConstantsEvolvable(
    const std::unordered_map<std::string, std::any> &params) {
   const auto it = params.find("evolve_self_modifying_constants");
   // Preserve the historical behaviour for old configs and checkpoints.
   return it == params.end() || std::any_cast<int>(it->second) != 0;
}

bool OperationsAdditiveEnabled(
    const std::unordered_map<std::string, std::any> &params) {
   auto it = params.find("operations_additive");
   return it != params.end() && std::any_cast<int>(it->second) != 0;
}

bool IsSelfModifyingDecoyAccess(size_t memory_type, int index,
                                int n_memories) {
   return memory_type == MemoryEigen::kScalarType_ &&
          n_memories > static_cast<int>(kSelfModifyingDecoyRegister) &&
          index % n_memories ==
              static_cast<int>(kSelfModifyingDecoyRegister);
}

bool IsSelfModifyingReservedScalarRegister(int idx) {
   return IsSelfModifyingRateRegister(idx) ||
          idx == static_cast<int>(kSelfModifyingDecoyRegister);
}

void AvoidInitialSelfModifyingOutput(instruction& instr, int n_memories,
                                     std::mt19937& rng) {
   if (instr.GetOutType() != MemoryEigen::kScalarType_) return;
   if (n_memories <= 0) return;

   const int out_idx = instr.outIdx_ % n_memories;
   if (!IsSelfModifyingReservedScalarRegister(out_idx)) return;

   std::vector<int> allowed_outputs;
   allowed_outputs.reserve(static_cast<size_t>(n_memories));
   for (int i = 0; i < n_memories; ++i) {
      if (!IsSelfModifyingReservedScalarRegister(i)) {
         allowed_outputs.push_back(i);
      }
   }

   if (allowed_outputs.empty()) {
      instr.outIdx_ = 0;
      return;
   }

   std::uniform_int_distribution<int> pick(
       0, static_cast<int>(allowed_outputs.size()) - 1);
   instr.outIdx_ = allowed_outputs[static_cast<size_t>(pick(rng))];
}

void ReplaceSelfModifyingDecoyIndex(int& index, int n_memories,
                                    std::mt19937& rng) {
   if (n_memories <= static_cast<int>(kSelfModifyingDecoyRegister) ||
       index % n_memories != static_cast<int>(kSelfModifyingDecoyRegister)) {
      return;
   }

   std::uniform_int_distribution<int> pick(0, n_memories - 2);
   int replacement = pick(rng);
   if (replacement >= static_cast<int>(kSelfModifyingDecoyRegister)) {
      replacement++;
   }
   index = replacement;
}

void RemoveSelfModifyingDecoyAccess(instruction& instr, int n_memories,
                                    std::mt19937& rng) {
   if (instr.GetOutType() == MemoryEigen::kScalarType_) {
      ReplaceSelfModifyingDecoyIndex(instr.outIdx_, n_memories, rng);
   }
   for (int in = 0; in < 2; ++in) {
      if (instr.GetInType(in) == MemoryEigen::kScalarType_ &&
          !instr.IsObs(in)) {
         int index = instr.GetInIdx(in);
         ReplaceSelfModifyingDecoyIndex(index, n_memories, rng);
         instr.SetInIdx(in, index);
      }
   }
}

// Apply elementwise sigmoid to map values to (0,1)
inline void SigmoidInPlace(MatrixDynamic &M) {
    M = M.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

// Multiply activation with the destination matrix. Prefer standard matrix product
// when dimensions align; otherwise fall back to elementwise multiplication when shapes match.
inline void MultiplyActivationWith(MatrixDynamic const &activation, MatrixDynamic &target) {
    if (activation.cols() == target.rows()) {
        target = activation * target; // standard matrix product
    } else if (activation.rows() == target.rows() && activation.cols() == target.cols()) {
        target = activation.cwiseProduct(target); // elementwise as a safe fallback
    } else {
        // Shapes incompatible; leave target unchanged.
        // (Optional) You may log if a logging facility is available.
    }
}

// Apply numerically stabilized, clamped row-wise softmax to row r of A.
inline void RowSoftmaxClampInPlace(MatrixDynamic &A, size_t r, double eps = 1e-12) {
    const int C = static_cast<int>(A.cols());
    if (C <= 0) return;
    // Numerically stable softmax: subtract row max before exponentiation
    double rmax = A.row(static_cast<int>(r)).maxCoeff();
    double sum = 0.0;
    // Temporary vector to hold exponentials
    std::vector<double> exps(static_cast<size_t>(C));
    for (int c = 0; c < C; ++c) {
        double v = std::exp(A(static_cast<int>(r), c) - rmax);
        exps[static_cast<size_t>(c)] = v;
        sum += v;
    }
    if (!(sum > 0.0)) {
        // Fallback to uniform distribution if underflow or degenerate
        double val = 1.0 / static_cast<double>(C);
        for (int c = 0; c < C; ++c) A(static_cast<int>(r), c) = val;
        return;
    }
    // First normalize and floor to eps to guarantee strictly positive entries
    for (int c = 0; c < C; ++c) {
        double v = exps[static_cast<size_t>(c)] / sum;
        if (v < eps) v = eps;
        A(static_cast<int>(r), c) = v;
    }
    // Renormalize so the row sums to 1 exactly after flooring
    double row_sum = A.row(static_cast<int>(r)).sum();
    if (row_sum > 0.0) {
        for (int c = 0; c < C; ++c) A(static_cast<int>(r), c) /= row_sum;
    }
}
}

/******************************************************************************/
std::string RegisterMachine::ToString(bool effective_only) {
   std::ostringstream oss;
   oss << "RegisterMachine:" << id_ << ":" << gtime_ << ":" << action_ << ":"
       << stateful_ << ":" << nrefs_ << ":" << observation_buff_size_ << ":"
       << learning_rate_ << ":" << initial_heb_noise_ << ":" << obs_index_ << ":" << n_memories_;

   oss << ":SF";
   if (register_stateful_.empty()) {
      // Fallback: encode using global stateful_ flag
      for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
         oss << "|";
         size_t n = private_memory_[mem_t]->n_memories_;
         for (size_t i = 0; i < n; ++i) {
            oss << (stateful_ ? '1' : '0');
         }
      }
   } else {
      for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
         oss << "|";
         auto* mem = private_memory_[mem_t];
         size_t n = mem->n_memories_;
         const auto& flags = register_stateful_[mem_t];
         for (size_t i = 0; i < n; ++i) {
            bool v = (i < flags.size()) ? flags[i] : (stateful_ != 0);
            oss << (v ? '1' : '0');
         }
      }
   }
   // for (auto &i : private_memory_ids_) oss << ":" << i;
   auto prog = effective_only ? instructions_effective_ : instructions_;
   for (auto istr : prog) oss << ":" << istr->ToString();
   oss << endl;
   for (auto* m : private_memory_)
      oss << m->ToString(id_);
   return oss.str();
}

/******************************************************************************/
std::string RegisterMachine::ToStringMemory() {
   std::ostringstream oss;
   for (auto* m : private_memory_)
      oss << m->ToString(id_);
   return oss.str();
}

void RegisterMachine::PrintInstructions(bool effective_only) const {
    const bool used_effective = effective_only && !instructions_effective_.empty();
    const auto& instrs = used_effective ? instructions_effective_ : instructions_;
   cout << "coordination on?: " << coordination_allowed << endl;
    auto TypeLabel = [](int memType) -> const char* {
        static const char* lbl[] = {"S","V","M"};
        if (memType < 0 || memType > 2) return "?";
        return lbl[memType];
    };

    auto RegLabel = [&](int memType, size_t idx) -> std::string {
        std::ostringstream os; os << TypeLabel(memType) << idx; return os.str();
    };

    auto PrintMatrixCompact = [&](const MatrixDynamic& mat) {
        size_t R = mat.rows();
        size_t C = mat.cols();
        std::cout.setf(std::ios::fixed); std::cout.precision(4);
        if (R == 1 && C == 1) {
            std::cout << mat(0,0);
            return;
        }
        if (C == 1) {
            std::cout << "[";
            size_t show = std::min<size_t>(8, R);
            for (size_t r = 0; r < show; ++r) {
                std::cout << mat(r,0);
                if (r + 1 < show) std::cout << ", ";
            }
            if (R > show) std::cout << ", …";
            std::cout << "]";
            return;
        }
        size_t showR = std::min<size_t>(3, R);
        size_t showC = std::min<size_t>(3, C);
        std::cout << "[";
        for (size_t r = 0; r < showR; ++r) {
            if (r) std::cout << "; ";
            std::cout << "[";
            for (size_t c = 0; c < showC; ++c) {
                std::cout << mat(r,c);
                if (c + 1 < showC) std::cout << ", ";
            }
            if (C > showC) std::cout << ", …";
            std::cout << "]";
        }
        if (R > showR) std::cout << "; …";
        std::cout << "]";
    };

    auto OperandLabel = [this](const instruction* inst, int in) -> std::string {
        if (inst->IsMemoryRef(in)) {
            int memType = inst->GetInType(in);
            int idx = inst->GetInIdx(in) % private_memory_[memType]->n_memories_;
            static const char* lbl[] = {"S","V","M"};
            std::ostringstream os;
            if (inst->IsConstRef(in)) os << "C";
            os << lbl[memType] << idx;
            return os.str();
        } else if (inst->IsObs(in)) {
            int t = inst->GetInType(in);
            size_t total = 0;
            if (t == MemoryEigen::kScalarType_) {
                total = static_cast<size_t>(observation_buff_size_);
            } else {
                total = observation_memory_buff_[t]->n_memories_;
            }
            int raw = inst->GetInIdx(in);
            int idx = (total > 0) ? (raw % static_cast<int>(total)) : 0;
            static const char* lbl[] = {"S","V","M"};
            std::ostringstream os;
            if (total > 0) {
                os << "OBS" << raw << "->obs_" << lbl[t] << "[" << idx << "/" << total << "]";
            } else {
                os << "OBS" << raw << "->obs_" << lbl[t] << "[" << idx << "]";
            }
            return os.str();
        } else {
            return std::string("IMM");
        }
    };

    std::cout << "prog: " << id_ << endl;
    std::cout << "discrete action: "<<  action_ << endl; 
    std::cout << "  instr_count: " << instrs.size();
    std::cout << "  effective: " << (used_effective ? 1 : 0);
    if (effective_only && !used_effective) std::cout << " (fallback)";
    std::cout << std::endl;

    {
        std::cout.setf(std::ios::fixed);
        std::cout.precision(4);
        std::cout << "evolved: lr=" << learning_rate_
                  << "  init_heb_noise=" << initial_heb_noise_
                  << "  use_evolved_const=" << (use_evolved_const_ ? 1 : 0)
                  << "  obs_index=" << obs_index_
                  << "  obs_buff_size=" << observation_buff_size_
                  << std::endl;
    }

    if (use_evolved_const_) {
        std::cout << "[Evolved Constants]" << std::endl;
        for (int t = 0; t < MemoryEigen::kNumMemoryType_; ++t) {
            const auto* bank = private_memory_[t];
            for (size_t idx = 0; idx < bank->n_memories_; ++idx) {
                std::cout << RegLabel(t, idx) << ": ";
                PrintMatrixCompact(bank->const_memory_[idx]);
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
    auto RenderImmSuffix = [&](const instruction* ins) -> std::string {
        std::ostringstream os;
        bool has = false;
        for (int k = 0; k < 2; ++k) {
            if (ins->GetInType(k) != -1 && !ins->IsMemoryRef(k) && !ins->IsObs(k)) {
                if (!has) { os << "  {"; has = true; }
                os << (k==0?"imm0":"imm1") << "=" << ins->GetInIdx(k);
                os << "; ";
            }
        }
        if (has) {
            std::string s = os.str();
            if (!s.empty() && s.back() == ' ') { s.pop_back(); }
            if (!s.empty() && s.back() == ';') { s.pop_back(); }
            return s + "}";
        }
        return std::string();
    };

    auto RenderInstr = [&](const instruction* ins) -> std::string {
        std::ostringstream os;
        std::string op_name = instruction::op_names_[ins->op_];

        std::string in0 = (ins->GetInType(0) != -1) ? OperandLabel(ins, 0) : std::string("-");
        std::string in1 = (ins->GetInType(1) != -1) ? OperandLabel(ins, 1) : std::string("-");

        os << op_name << "(" << in0;
        if (ins->GetInType(1) != -1) os << ", " << in1;
        os << ")";

        std::string imm = RenderImmSuffix(ins);
        if (!imm.empty()) os << imm;

        if (op_name == "SCALAR_VECTOR_ASSIGN_OP") {
            os << "  {i=" << ins->in1Idx_ << "}";
        }

        return os.str();
    };

    auto key64 = [](int t, size_t i){ return (uint64_t(t) << 32) | uint64_t(i); };
    std::unordered_map<uint64_t, std::string> resident_instr; 

    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto* ins = instrs[i];
        int tOut = ins->GetOutType();
        size_t oIdx = ins->outIdx_ % private_memory_[tOut]->n_memories_;
        resident_instr[key64(tOut, oIdx)] = RenderInstr(ins);
    }

    std::cout << "[Register Instructions]" << std::endl;
    for (size_t t = 0; t < MemoryEigen::kNumMemoryType_; ++t) {
        const auto* bank = private_memory_[t];
        for (size_t idx = 0; idx < bank->n_memories_; ++idx) {
            auto it = resident_instr.find(key64((int)t, idx));

            bool is_stateful = true; 
            if (t < register_stateful_.size()) {
                const auto& flags = register_stateful_[t];
                if (idx < flags.size()) {
                    is_stateful = flags[idx];
                } else {
                    is_stateful = (stateful_ != 0);
                }
            } else {
                is_stateful = (stateful_ != 0);
            }
            const char* sf_label = is_stateful ? "stateful" : "stateless";

            std::cout << "[(U)" << RegLabel((int)t, idx) << "]"
                      << " [" << sf_label << "]"
                      << (operations_additive_ ? " += " : " = ")
                      << (it == resident_instr.end() ? std::string("-") : it->second)
                      << std::endl;
        }
    }

    std::cout << "\n[Instructions]" << std::endl;
    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto* instr = instrs[i];
        int outType = instr->GetOutType();
        size_t outIdx = instr->outIdx_ % private_memory_[outType]->n_memories_;

        std::string in0 = (instr->GetInType(0) != -1) ? OperandLabel(instr, 0) : std::string("-");
        std::string in1 = (instr->GetInType(1) != -1) ? OperandLabel(instr, 1) : std::string("-");

        std::cout << "[" << i << "] "
                  << RegLabel(outType, outIdx)
                  << (operations_additive_ ? " += " : " = ")
                  << instruction::op_names_[instr->op_] << "(" << in0;
        if (instr->GetInType(1) != -1) std::cout << ", " << in1;
        std::cout << ")";
        std::string imm = RenderImmSuffix(instr);
        if (!imm.empty()) std::cout << imm;
        std::cout << std::endl;
    }
    auto* Scal = private_memory_[MemoryEigen::kScalarType_];
    if (Scal->n_memories_ >= 1) {
        std::cout << "\n[Return] bid(s0): ";
        PrintMatrixCompact(Scal->working_memory_[0]);
        std::cout << std::endl;
    }
    if (Scal->n_memories_ >= 2) {
      //   std::cout << "[Return] action(s1): ";
        PrintMatrixCompact(Scal->working_memory_[1]);
        std::cout << std::endl;
    }
}

/******************************************************************************/
// Create arbitrary RegisterMachine
RegisterMachine::RegisterMachine(
    long action, int team_obs_index, std::unordered_map<std::string, std::any>& params,
    std::unordered_map<std::string, int>& state, mt19937& rng,
    std::vector<bool>& legal_ops) {
   action_ = action;
   self_modifying_ = SelfModifyingEnabled(params);
   stateful_ = std::any_cast<int>(params["stateful"]);
   operations_additive_ = OperationsAdditiveEnabled(params);
   gtime_ = state["t_current"];
   id_ = state["program_count"]++;
   nrefs_ = 0;
   matrix_modulation = (std::any_cast<int>(params["matrix_modulation"]) == 1);
   coordination_allowed = (std::any_cast<int>(params["dynamic_lgp"]) == 1);

   //before lr = 0.0 - 0.25
   //before lr = 0.0 - 0.50
   std::normal_distribution<double> lr_dist(0.5, 0.15);
   std::lognormal_distribution<double> noise_dist(0.0, 0.5); 

   learning_rate_ = std::any_cast<double>(params["static_learning_rate"]);
   if (learning_rate_ == 0.0) {
            learning_rate_    = std::clamp(lr_dist(rng), 0.0, 1.0);}
   initial_heb_noise_ = std::clamp(noise_dist(rng), 0.0, 1.0);
      

   observation_buff_size_ = std::any_cast<int>(params["observation_buff_size"]);
   obs_index_ = team_obs_index;
   uniform_int_distribution<int> disP(
       std::any_cast<int>(params["min_initial_prog_size"]), std::any_cast<int>(params["max_initial_prog_size"]));
   int prog_size = disP(rng);

   //  uniform_int_distribution<int> disMem(
   //     std::any_cast<int>(params["min_initial_mem_slots"]), std::any_cast<int>(params["max_initial_mem_slots"]));
   // n_memories_ =  disMem(rng);

   int minMem = std::any_cast<int>(params["min_initial_mem_slots"]);
   int maxMem = std::any_cast<int>(params["max_initial_mem_slots"]);
   std::geometric_distribution<int> geo(0.15);

   // Rejection sampling with fallback: try up to 100 times, then use uniform
   int v;
   int attempts = 0;
   do {
      v = minMem + geo(rng);
      attempts++;
   } while (v > maxMem && attempts < 100);
   if (v > maxMem) {
      std::uniform_int_distribution<int> fallback(minMem, maxMem);
      v = fallback(rng);
   }
   n_memories_ = v;
   if (SelfModifyingEnabled(params)) {
      n_memories_ = std::max(
          n_memories_, static_cast<int>(kSelfModifyingMinScalarRegisters));
   }

   for (int i = 0; i < prog_size; i++) {
      auto in = new instruction(params, rng, n_memories_);
      in->Mutate(true, legal_ops, observation_buff_size_, rng);
      if (SelfModifyingEnabled(params)) {
         AvoidInitialSelfModifyingOutput(*in, n_memories_, rng);
         RemoveSelfModifyingDecoyAccess(*in, n_memories_, rng);
      }
      instructions_.push_back(in);
   }
   op_counts_.resize(instruction::NUM_OP);
   if (!isEqual(std::any_cast<double>(params["p_memory_mu_const"]),
                0.0)) {
      use_evolved_const_ = true;
   }
   SetupMemory(state, n_memories_,
               std::any_cast<int>(params["memory_size"]), rng);
   ConfigureSelfModifyingRegisters(params);
}

// Create RegisterMachine and copy instructions
// If inherited_n_memories > 0, use that value instead of sampling from config
RegisterMachine::RegisterMachine(
    long action, std::vector<instruction*> &instructions,
    std::unordered_map<std::string, std::any> &params,
    std::unordered_map<std::string, int> &state, mt19937 &rng,
    std::vector<bool> &legal_ops, int inherited_n_memories) {
   action_ = action;
   self_modifying_ = SelfModifyingEnabled(params);
   stateful_ = std::any_cast<int>(params["stateful"]);
   operations_additive_ = OperationsAdditiveEnabled(params);
   gtime_ = state["t_current"];
   id_ = state["program_count"]++;
   nrefs_ = 0;
   matrix_modulation = (std::any_cast<int>(params["matrix_modulation"]) == 1);
   coordination_allowed = (std::any_cast<int>(params["dynamic_lgp"]) == 1);

   
   std::lognormal_distribution<double> lr_dist(0.0, 0.25);
   std::lognormal_distribution<double> noise_dist(0.0, 0.5); 
   learning_rate_    = std::clamp(lr_dist(rng),    0.0, 1.0);
   initial_heb_noise_ = std::clamp(noise_dist(rng), 0.0, 1.0);

   observation_buff_size_ = std::any_cast<int>(params["observation_buff_size"]);
   obs_index_ = 0;

   // Inherit memory slots from parent if provided, otherwise sample from config
   if (inherited_n_memories > 0) {
      n_memories_ = inherited_n_memories;
   } else {
      // uniform_int_distribution<int> disMem(
      //    std::any_cast<int>(params["min_initial_mem_slots"]), 
      //    std::any_cast<int>(params["max_initial_mem_slots"]));
      // n_memories_ = disMem(rng);
      int minMem = std::any_cast<int>(params["min_initial_mem_slots"]);
      int maxMem = std::any_cast<int>(params["max_initial_mem_slots"]);
      std::geometric_distribution<int> geo(0.15);

      // Rejection sampling with fallback: try up to 100 times, then use uniform
      int v;
      int attempts = 0;
      do {
         v = minMem + geo(rng);
         attempts++;
      } while (v > maxMem && attempts < 100);
      if (v > maxMem) {
         std::uniform_int_distribution<int> fallback(minMem, maxMem);
         v = fallback(rng);
      }
      n_memories_ = v;
   }
   if (SelfModifyingEnabled(params)) {
      n_memories_ = std::max(
          n_memories_, static_cast<int>(kSelfModifyingMinScalarRegisters));
   }

   for (auto i : instructions) {
      auto* copied = new instruction(*i);
      if (self_modifying_) {
         RemoveSelfModifyingDecoyAccess(*copied, n_memories_, rng);
      }
      instructions_.push_back(copied);
   }
   op_counts_.resize(instruction::NUM_OP);
   if (!isEqual(std::any_cast<double>(params["p_memory_mu_const"]),
                0.0)) {
      use_evolved_const_ = true;
   }
   SetupMemory(state, n_memories_,
               std::any_cast<int>(params["memory_size"]), rng);
   ConfigureSelfModifyingRegisters(params);
}

// Copy Contructor
RegisterMachine::RegisterMachine(RegisterMachine& rm) {
   action_ = rm.action_;
   gtime_ = rm.gtime_;
   id_ = rm.id_;
   obs_index_ = rm.obs_index_;
   n_memories_ = rm.n_memories_;
   bid_val_ = -(numeric_limits<double>::max());
   nrefs_ = 0;
   stateful_ = rm.stateful_;
   operations_additive_ = rm.operations_additive_;
   self_modifying_ = rm.self_modifying_;
   learning_rate_ = rm.learning_rate_;
   initial_heb_noise_ = rm.initial_heb_noise_;
   observation_buff_size_ = rm.observation_buff_size_;
   use_evolved_const_ = rm.use_evolved_const_;
   op_counts_.resize(instruction::NUM_OP);
   for (size_t i = 0; i < MemoryEigen::kNumMemoryType_; i++) {
      private_memory_.push_back(new MemoryEigen(*(rm.private_memory_[i])));
      observation_memory_buff_.push_back(
          new MemoryEigen(*(rm.observation_memory_buff_[i])));
   }
   private_memory_ids_ = rm.private_memory_ids_;
   for (auto i : rm.instructions_) {
      instructions_.push_back(new instruction(*i));
   }
   register_stateful_ = rm.register_stateful_;
}

// Clone RegisterMachine with new id
RegisterMachine::RegisterMachine(
    RegisterMachine& rm, std::unordered_map<std::string, std::any>& params,
    std::unordered_map<std::string, int>& state, mt19937& rng) {
   action_ = rm.action_;
   self_modifying_ = SelfModifyingEnabled(params);
   gtime_ = state["t_current"];
   id_ = state["program_count"]++;
   obs_index_ = rm.obs_index_;
   bid_val_ = -(numeric_limits<double>::max());
   matrix_modulation = (std::any_cast<int>(params["matrix_modulation"]) == 1);
   coordination_allowed = (std::any_cast<int>(params["dynamic_lgp"]) == 1);
   nrefs_ = 0;
   stateful_ = rm.stateful_;
   operations_additive_ = OperationsAdditiveEnabled(params);
   observation_buff_size_ = rm.observation_buff_size_;
   n_memories_ = rm.n_memories_;
   learning_rate_ = rm.learning_rate_;
   initial_heb_noise_ = rm.initial_heb_noise_;
   use_evolved_const_ = rm.use_evolved_const_;
   n_memories_ = rm.n_memories_;
   op_counts_.resize(instruction::NUM_OP);
   for (auto instr : rm.instructions_) {
      auto* copied = new instruction(*instr);
      if (self_modifying_) {
         RemoveSelfModifyingDecoyAccess(*copied, n_memories_, rng);
      }
      instructions_.push_back(copied);
   }
   SetupMemory(state, n_memories_,
               rm.private_memory_[MemoryEigen::kScalarType_]->memory_size_, rng);
   CopyEvolvedConstants(rm);
   ConfigureSelfModifyingRegisters(params, false);
   CopySelfModifyingOutputs(rm, params);
   register_stateful_ = rm.register_stateful_;
}

// Create RegisterMachine from string
RegisterMachine::RegisterMachine(
    std::vector<std::string> outcomeFields,
    std::vector<std::map<long, MemoryEigen*>>& memory_maps,
    std::unordered_map<std::string, std::any>& params, mt19937& rng) {
   int f = 1;
   id_ = atoi(outcomeFields[f++].c_str());
   gtime_ = atoi(outcomeFields[f++].c_str());
   action_ = atoi(outcomeFields[f++].c_str());
   stateful_ = atoi(outcomeFields[f++].c_str());
   nrefs_ = atoi(outcomeFields[f++].c_str());
   observation_buff_size_ = atoi(outcomeFields[f++].c_str());
   learning_rate_ = stod(outcomeFields[f++].c_str());
   initial_heb_noise_ = stod(outcomeFields[f++].c_str());
   obs_index_ = atoi(outcomeFields[f++].c_str());
   n_memories_ = atoi(outcomeFields[f++].c_str());
   self_modifying_ = SelfModifyingEnabled(params);
   operations_additive_ = OperationsAdditiveEnabled(params);
   matrix_modulation = (std::any_cast<int>(params["matrix_modulation"]) == 1);
   coordination_allowed = (std::any_cast<int>(params["dynamic_lgp"]) == 1);

   bool has_state_flags = false;
   std::string sf_token;
   if (f < static_cast<int>(outcomeFields.size()) &&
       outcomeFields[f].rfind("SF", 0) == 0) { // starts with "SF"
      has_state_flags = true;
      sf_token = outcomeFields[f++];
   }

   // SetupMemry() but from existing memory pointers
   for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
      private_memory_.push_back(memory_maps[mem_t][id_]);
      auto memory_size = memory_maps[mem_t][id_]->memory_size_;
      observation_memory_buff_.push_back(
          new MemoryEigen(mem_t, observation_buff_size_, memory_size));
   }
   if (coordination_allowed || matrix_modulation) {
      const size_t nmem = private_memory_[MemoryEigen::kScalarType_]->n_memories_;
      register_activations.resize(static_cast<int>(nmem), static_cast<int>(nmem));
      register_activations.setOnes();
   } else {
      // Keep empty unless we actually need it (major memory savings for large populations).
      register_activations.resize(0, 0);
   }

   register_stateful_.clear();
   register_stateful_.resize(MemoryEigen::kNumMemoryType_);

   if (has_state_flags) {
      // sf_token format: "SF|<scalar_bits>|<vector_bits>|<matrix_bits>"
      // Split on '|' manually to avoid depending on other utilities
      std::vector<std::string> parts;
      {
         size_t start = 0;
         while (start <= sf_token.size()) {
            size_t pos = sf_token.find('|', start);
            if (pos == std::string::npos) {
               parts.push_back(sf_token.substr(start));
               break;
            } else {
               parts.push_back(sf_token.substr(start, pos - start));
               start = pos + 1;
            }
         }
      }
      // parts[0] should be "SF"
      for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
         auto &flags = register_stateful_[mem_t];
         size_t n = private_memory_[mem_t]->n_memories_;
         std::string bits;
         if (mem_t + 1 < parts.size()) {
            bits = parts[mem_t + 1];  // scalar bits, vector bits, matrix bits...
         }
         for (char c : bits) {
            if (c == '0' || c == '1') {
               flags.push_back(c == '1');
            }
         }
         // Ensure length matches n_memories_ for this type
         if (flags.size() < n) {
            flags.resize(n, stateful_ != 0);
         } else if (flags.size() > n) {
            flags.resize(n);
         }
      }
   } else {
      // Backward compatibility: no SF token, use global stateful_ for all
      for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
         size_t n = private_memory_[mem_t]->n_memories_;
         register_stateful_[mem_t].assign(n, stateful_ != 0);
      }
   }

   for (size_t ii = f; ii < outcomeFields.size(); ii++) {
      vector<string> instructionString;
      SplitString(outcomeFields[ii], '_', instructionString);
      instruction* in = new instruction(params, rng, n_memories_);
      in->in1Src_ = stringToInt(instructionString[0]);
      in->in2Src_ = stringToInt(instructionString[1]);
      in->outIdx_ = stringToInt(instructionString[2]);
      in->op_ = stringToInt(instructionString[3]);
      in->in0Idx_ = stringToInt(instructionString[4]);
      in->in1Idx_ = stringToInt(instructionString[5]);
      in->in2Idx_ = stringToInt(instructionString[6]);
      in->in3Idx_ = stringToInt(instructionString[7]);
      if (self_modifying_) {
         RemoveSelfModifyingDecoyAccess(*in, n_memories_, rng);
      }
      instructions_.push_back(in);
   }
   instructions_effective_ = instructions_;
   op_counts_.resize(instruction::NUM_OP);
   from_string_ = true;
   // Checkpoints already contain evolved S2-S5 constants.
   ConfigureSelfModifyingRegisters(params, false);
   SeedSelfModifyingWorkingFromConstants(params);
}

RegisterMachine::~RegisterMachine() {
   for (auto* i : instructions_)
      delete i;
   for (auto* m : private_memory_)
      delete m;
   for (auto* m : observation_memory_buff_)
      delete m;
}

// TODO(skelly): WARNING: confirm this function works as expected.
void RegisterMachine::MarkFeatures(instruction* istr, int in) {
   // Setting input memory pointer is required for MarkIntrons.
   istr->SetInMem(in, observation_memory_buff_[istr->GetInType(in)]);

   features_.clear();
   if (istr->GetInType(in) == MemoryEigen::kScalarType_) {
      features_.insert(istr->GetInIdx(in));
   } else if (istr->GetInType(in) == MemoryEigen::kVectorType_) {
      for (size_t f = istr->GetInIdx(in), row = 0;
           row < istr->GetInMem(in)->memory_size_; row++) {
         // features_.insert(f++ % num_input_);  // toroidal
         features_.insert(f++);
      }
   } else if (istr->GetInType(in) == MemoryEigen::kMatrixType_) {
      for (size_t f = istr->GetInIdx(in), row = 0;
           row < istr->GetInMem(in)->memory_size_; row++) {
         for (size_t col = 0; col < istr->GetInMem(in)->memory_size_; col++) {
            // features_.insert(f++ % num_input_);  // toroidal
            features_.insert(f++);
         }
      }
   }
}

void RegisterMachine::MarkIntrons(
    std::unordered_map<std::string, std::any>& params, int n_memories) {
   map<int, vector<bool>> memories_effective;
   memories_effective[MemoryEigen::kScalarType_] =
          vector<bool>(n_memories, false);
   memories_effective[MemoryEigen::kVectorType_] =
       vector<bool>(n_memories, false);
   memories_effective[MemoryEigen::kMatrixType_] =
       vector<bool>(n_memories, false);

   if (std::any_cast<int>(params["use_all_scalar_registers"]) == 1) {
      int action_space;
      if (params.find("environment_action_space") != params.end()) {
         action_space = std::any_cast<int>(params["environment_action_space"]);
      } else {
         action_space = std::any_cast<int>(params["min_memory_slots"]);
      }
      if (action_space < 1) action_space = 1;
      if (action_space > n_memories) action_space = n_memories;

      for (int i = 0; i < action_space; ++i) {
         memories_effective[MemoryEigen::kScalarType_][i] = true;
      }
      if (std::any_cast<int>(params["matrix_modulation"]) == 1) {
         memories_effective[MemoryEigen::kMatrixType_][0] = true;
   }
   }else{
      memories_effective[MemoryEigen::kScalarType_][kBidRegister] = true;
      if (std::any_cast<int>(params["continuous_output"]) == 0)
         memories_effective[MemoryEigen::kScalarType_][kActionRegister] = true;
      else if (std::any_cast<int>(params["continuous_output"]) == 1)
         memories_effective[MemoryEigen::kScalarType_][kActionRegister] = true;
      else if (std::any_cast<int>(params["continuous_output"]) == 2)
         memories_effective[MemoryEigen::kVectorType_][1] = true;
      else if (std::any_cast<int>(params["continuous_output"]) == 3)
         memories_effective[MemoryEigen::kMatrixType_][1] = true;
   }

   // Mutation-rate outputs are part of the program phenotype. Without this,
   // intron removal drops instructions that only write S2-S5 and the inherited
   // self-modifying rates remain fixed at their initial values.
   if (SelfModifyingEnabled(params)) {
      for (size_t i = 0; i < kSelfModifyingRegisterCount; ++i) {
         const size_t reg = kSelfModifyingFirstRegister + i;
         if (reg < static_cast<size_t>(n_memories)) {
            memories_effective[MemoryEigen::kScalarType_][reg] = true;
         }
      }
      if (n_memories > static_cast<int>(kSelfModifyingDecoyRegister)) {
         memories_effective[MemoryEigen::kScalarType_]
                           [kSelfModifyingDecoyRegister] = false;
      }
   }

   // Backward pass to find effective instructions when stateless
   std::vector<instruction*> instructions_effective_stateless;
   for (auto riter = instructions_.rbegin(); riter != instructions_.rend();
        riter++) {
      auto istr = *riter;
      const bool writes_decoy = SelfModifyingEnabled(params) &&
          IsSelfModifyingDecoyAccess(istr->GetOutType(), istr->outIdx_,
                                     n_memories);
      if (!writes_decoy &&
          memories_effective[istr->GetOutType()][istr->outIdx_ % n_memories]) {
         instructions_effective_stateless.push_back(istr);
         for (int in = 0; in < 2; in++) {
            if (istr->IsMemoryRef(in) && !istr->IsConstRef(in)) {
               int t   = istr->GetInType(in);
               int idx = istr->GetInIdx(in) % n_memories;
               if (SelfModifyingEnabled(params) &&
                   IsSelfModifyingDecoyAccess(t, istr->GetInIdx(in),
                                              n_memories)) {
                  continue;
               }
               // Only allow Matrix inputs to propagate if they are M0.
               if (t != MemoryEigen::kMatrixType_ || idx == 0) {
                  memories_effective[t][idx] = true;
               }
            }
         }
      }
   }

   // TODO(skelly): Is this the most efficient method? Currently O(n^2)
   for (size_t t = 0; t < instructions_.size(); t++) {
      instructions_effective_.clear();
      // Count occurance of each op.
      std::fill(op_counts_.begin(), op_counts_.end(), 0);
      for (auto istr : instructions_) {
         const bool writes_decoy = SelfModifyingEnabled(params) &&
             IsSelfModifyingDecoyAccess(istr->GetOutType(), istr->outIdx_,
                                        n_memories);
         if ((!writes_decoy && memories_effective[istr->GetOutType()]
                                [istr->outIdx_ % n_memories]) ||
             std::find(instructions_effective_stateless.begin(),
                       instructions_effective_stateless.end(),
                       istr) != instructions_effective_stateless.end()) {
            instructions_effective_.push_back(istr);
            op_counts_[istr->op_]++;
            for (int in = 0; in < 2; in++) {
               if (istr->IsMemoryRef(in) && !istr->IsConstRef(in)) {
                  if (SelfModifyingEnabled(params) &&
                      IsSelfModifyingDecoyAccess(istr->GetInType(in),
                                                 istr->GetInIdx(in),
                                                 n_memories)) {
                     continue;
                  }
                  memories_effective[istr->GetInType(in)]
                                    [istr->GetInIdx(in) % n_memories] = true;
               } else if (istr->IsObs(in)) {
                  MarkFeatures(istr, in);
               }
            }
         }
      }
   }
   map<int, vector<bool>> regs_used;
   regs_used[MemoryEigen::kScalarType_] = vector<bool>(n_memories, false);
   regs_used[MemoryEigen::kVectorType_] = vector<bool>(n_memories, false);
   regs_used[MemoryEigen::kMatrixType_] = vector<bool>(n_memories, false);
   for (auto* istr : instructions_effective_) {
      if (!(SelfModifyingEnabled(params) &&
            IsSelfModifyingDecoyAccess(istr->GetOutType(), istr->outIdx_,
                                       n_memories))) {
         regs_used[istr->GetOutType()][istr->outIdx_ % n_memories] = true;
      }
      for (int in = 0; in < 2; in++) {
         if (istr->IsMemoryRef(in)) {
            if (SelfModifyingEnabled(params) &&
                IsSelfModifyingDecoyAccess(istr->GetInType(in),
                                           istr->GetInIdx(in), n_memories)) {
               continue;
            }
            regs_used[istr->GetInType(in)][istr->GetInIdx(in) % n_memories] = true;
         }
      }
   }
   n_effective_registers_ = 0;
   for (int mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
      for (int i = 0; i < n_memories; ++i) {
         if (regs_used[mem_t][i]) {
            n_effective_registers_++;
         }
      }
   }
}

double RegisterMachine::ComputeMeanPairwiseRegisterSimilarity() const {
   const int n = n_memories_;
   if (n < 2) return 0.0;

   // Build per-register operation profile from effective instructions.
   // profile[reg] = vector of operation counts (length NUM_OP)
   std::vector<std::vector<double>> profiles(n, std::vector<double>(instruction::NUM_OP, 0.0));
   for (auto* ins : instructions_effective_) {
      if (ins->GetOutType() == MemoryEigen::kScalarType_) {
         int reg = ins->outIdx_ % n;
         if (ins->op_ >= 0 && ins->op_ < instruction::NUM_OP)
            profiles[reg][ins->op_] += 1.0;
      }
   }

   auto cosine = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
      double dot = 0.0, na = 0.0, nb = 0.0;
      for (size_t k = 0; k < a.size(); k++) {
         dot += a[k] * b[k];
         na  += a[k] * a[k];
         nb  += b[k] * b[k];
      }
      if (na < 1e-12 || nb < 1e-12) return 0.0;
      return dot / (std::sqrt(na) * std::sqrt(nb));
   };

   double total = 0.0;
   int pairs = 0;
   for (int i = 0; i < n; i++) {
      double ni = 0.0;
      for (auto v : profiles[i]) ni += v * v;
      if (ni < 1e-12) continue;
      for (int j = i + 1; j < n; j++) {
         double nj = 0.0;
         for (auto v : profiles[j]) nj += v * v;
         if (nj < 1e-12) continue;
         total += cosine(profiles[i], profiles[j]);
         pairs++;
      }
   }
   return pairs > 0 ? total / static_cast<double>(pairs) : 0.0;
}

double RegisterMachine::ComputeMaxPairwiseRegisterSimilarity() const {
   const int n = n_memories_;
   if (n < 2) return 0.0;

   std::vector<std::vector<double>> profiles(n, std::vector<double>(instruction::NUM_OP, 0.0));
   for (auto* ins : instructions_effective_) {
      if (ins->GetOutType() == MemoryEigen::kScalarType_) {
         int reg = ins->outIdx_ % n;
         if (ins->op_ >= 0 && ins->op_ < instruction::NUM_OP)
            profiles[reg][ins->op_] += 1.0;
      }
   }

   auto cosine = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
      double dot = 0.0, na = 0.0, nb = 0.0;
      for (size_t k = 0; k < a.size(); k++) {
         dot += a[k] * b[k];
         na  += a[k] * a[k];
         nb  += b[k] * b[k];
      }
      if (na < 1e-12 || nb < 1e-12) return 0.0;
      return dot / (std::sqrt(na) * std::sqrt(nb));
   };

   double max_sim = 0.0;
   for (int i = 0; i < n; i++) {
      double ni = 0.0;
      for (auto v : profiles[i]) ni += v * v;
      if (ni < 1e-12) continue;
      for (int j = i + 1; j < n; j++) {
         double nj = 0.0;
         for (auto v : profiles[j]) nj += v * v;
         if (nj < 1e-12) continue;
         double sim = cosine(profiles[i], profiles[j]);
         if (sim > max_sim) max_sim = sim;
      }
   }
   return max_sim;
}

void RegisterMachine::MutateRegisterStatefulFlags(mt19937 &rng) {
   // Every register is eligible, including scalar action/output registers.
   constexpr double per_flag_p = 0.1;
   std::bernoulli_distribution flip(per_flag_p);

   for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
      auto &flags = register_stateful_[mem_t];

      const size_t n = private_memory_[mem_t]->n_memories_;
      if (flags.size() < n) {
         flags.resize(n, stateful_ != 0);
      }

      for (size_t i = 0; i < n; ++i) {
         if (flip(rng)) {
            flags[i] = !flags[i];
         }
      }
   }
}

void RegisterMachine::Mutate(std::unordered_map<std::string, std::any> &params,
                             std::unordered_map<std::string, int> &state,
                             mt19937 &rng, vector<bool> &legal_ops) {
   uniform_real_distribution<> dis_real(0, 1.0);

      double p_swap = std::any_cast<double>(params["p_instructions_swap"]);
      double p_delete = std::any_cast<double>(params["p_instructions_delete"]);
      double p_add = std::any_cast<double>(params["p_instructions_add"]);
      // Older non-self-modifying configurations did not declare this key.
      // Preserve their previous one-point-mutation-per-pass behavior there.
      const auto point_mutation_rate = params.find("p_instructions_mutate");
      double p_mutate = point_mutation_rate == params.end()
          ? 1.0
          : std::any_cast<double>(point_mutation_rate->second);

      if (SelfModifyingEnabled(params)) {
         auto* scalar_memory = private_memory_[MemoryEigen::kScalarType_];
         if (scalar_memory->n_memories_ < kSelfModifyingMinScalarRegisters) {
            die(__FILE__, __FUNCTION__, __LINE__,
                "self_modifying requires scalar registers S0-S6.");
         }
         const double rate_values[kSelfModifyingRegisterCount] = {
             scalar_memory->working_memory_[kSelfModifyingFirstRegister](0, 0),
             scalar_memory->working_memory_[kSelfModifyingFirstRegister + 1](0, 0),
             scalar_memory->working_memory_[kSelfModifyingFirstRegister + 2](0, 0),
             scalar_memory->working_memory_[kSelfModifyingFirstRegister + 3](0, 0)};
         p_swap = SelfModifyingRawTendencyToProbability(rate_values[0]);
         p_delete = SelfModifyingRawTendencyToProbability(rate_values[1]);
         p_add = SelfModifyingRawTendencyToProbability(rate_values[2]);
         p_mutate = SelfModifyingRawTendencyToProbability(rate_values[3]);
      }

      // Remove random instruction
      if (instructions_.size() > 1 &&
          dis_real(rng) < p_delete) {
         uniform_int_distribution<int> disBid(0, instructions_.size() - 1);
         int i = disBid(rng);
         delete *(instructions_.begin() + i);
         instructions_.erase(instructions_.begin() + i);
      }

      // Insert a new random instruction
      if ((int)instructions_.size() <
              std::any_cast<int>(params["max_prog_size"]) &&
          dis_real(rng) < p_add) {
         instruction *instr = new instruction(params, rng, n_memories_);
         instr->Mutate(true, legal_ops, observation_buff_size_, rng);
         uniform_int_distribution<int> disBid(0, instructions_.size());
         int i = disBid(rng);
         instructions_.insert(instructions_.begin() + i, instr);
      }

      // Mutate a randomly selected instruction
      if (!instructions_.empty() && dis_real(rng) < p_mutate) {
         uniform_int_distribution<int> disBid(0, instructions_.size() - 1);
         auto i = disBid(rng);
         instructions_[i]->Mutate(false, legal_ops,
                                            observation_buff_size_, rng);
      }

      // Mutate constants. S2-S5/S6 use probability-space mutation; other slots
      // use the AutoML-Zero multiplicative rule.
      if (use_evolved_const_ && dis_real(rng) <
          std::any_cast<double>(params["p_memory_mu_const"])) {
         const bool self_modifying = SelfModifyingEnabled(params);
         for (size_t mem_type = 0; mem_type < private_memory_.size(); ++mem_type) {
            auto* memory = private_memory_[mem_type];
            if (self_modifying && mem_type == MemoryEigen::kScalarType_) {
               memory->MutateConstants(
                   rng, {kSelfModifyingFirstRegister,
                         kSelfModifyingFirstRegister + 1,
                         kSelfModifyingFirstRegister + 2,
                         kSelfModifyingFirstRegister + 3,
                         kSelfModifyingDecoyRegister});
            } else {
               memory->MutateConstants(rng);
            }
         }
         if (self_modifying && SelfModifyingConstantsEvolvable(params)) {
            auto* scalar_memory = private_memory_[MemoryEigen::kScalarType_];
            for (size_t i = 0; i < kSelfModifyingRegisterCount; ++i) {
               const size_t reg = kSelfModifyingFirstRegister + i;
               const double prior = scalar_memory->const_memory_[reg](0, 0);
               scalar_memory->const_memory_[reg](0, 0) =
                   MutateSelfModifyingRateConstant(prior, rng);
            }
            if (scalar_memory->n_memories_ > kSelfModifyingDecoyRegister) {
               const double prior = scalar_memory->const_memory_
                   [kSelfModifyingDecoyRegister](0, 0);
               scalar_memory->const_memory_[kSelfModifyingDecoyRegister](0, 0) =
                   MutateSelfModifyingRateConstant(prior, rng);
            }
         }
      }

      
      std::uniform_real_distribution<double> coin(0.0, 1.0);
      double mut_lr    = std::any_cast<double>(params["mut_learning_rate"]);

      static constexpr double sd_meta_lr    = 0.1;
      // static constexpr double sd_meta_noise = 0.2;

      if (coin(rng) < mut_lr) {
         std::normal_distribution<double> gauss_lr(0.0, sd_meta_lr);
         double factor = std::exp(gauss_lr(rng)); 
         learning_rate_ = std::clamp(learning_rate_ * factor, 0.0, 1.0);
      }


      // Swap positions of two instructions
      if (instructions_.size() > 1 &&
          dis_real(rng) < p_swap) {
         uniform_int_distribution<int> disBid(0, instructions_.size() - 1);
         int i = disBid(rng);
         int j;
         do {
            j = disBid(rng);
         } while (i == j);
         std::swap(instructions_[i], instructions_[j]);
      }

      // Change observation buff size
      // if (dis_real(rng) <
      //     std::any_cast<double>(params["p_observation_buff_size"])) {
      //   MutateObsBuffSize(std::any_cast<int>(params["max_observation_buff_size"]),
      //                     rng);
      // }

      // Change memory size
      if (dis_real(rng) < std::any_cast<double>(params["p_memory_size"])) {
         MutateMemorySize(params, state, rng);
      }

      if (dis_real(rng) < std::any_cast<double>(params["p_memory_slots"])) {
         MutateMemorySlots(params, state, rng);
      }

      if (dis_real(rng) < std::any_cast<double>(params["p_hidden_stateful_mutation"])) {
         MutateRegisterStatefulFlags(rng);
      }
      //Change observation index
      if (dis_real(rng) <
          std::any_cast<double>(params["p_observation_index"])) {
         const int max_index = 1000000;  // TODO(skelly): fix magic #
         uniform_int_distribution<int> dis(0, max_index);
         auto prev = obs_index_;
         do {
            obs_index_ = dis(rng);
         } while (obs_index_ == prev);
      }

      // S6 is a genetic-drift control, not executable memory. Repair any
      // instruction or memory-slot mutation that would make it readable or
      // writable by the program.
      self_modifying_ = SelfModifyingEnabled(params);
      if (self_modifying_) {
         for (auto* instr : instructions_) {
            RemoveSelfModifyingDecoyAccess(*instr, n_memories_, rng);
         }
      }
}

void RegisterMachine::ConfigureSelfModifyingRegisters(
    std::unordered_map<std::string, std::any> &params,
    bool initialize_working_memory) {
   self_modifying_ = SelfModifyingEnabled(params);
   if (!self_modifying_) return;

   auto* scalar_memory = private_memory_[MemoryEigen::kScalarType_];
   if (scalar_memory->n_memories_ < kSelfModifyingMinScalarRegisters) {
      die(__FILE__, __FUNCTION__, __LINE__,
          "self_modifying requires scalar registers S0-S6.");
   }

   const double values[kSelfModifyingRegisterCount] = {
       std::any_cast<double>(params["p_instructions_swap"]),
       std::any_cast<double>(params["p_instructions_delete"]),
       std::any_cast<double>(params["p_instructions_add"]),
       std::any_cast<double>(params["p_instructions_mutate"])};

   for (size_t i = 0; i < kSelfModifyingRegisterCount; ++i) {
      const size_t reg = kSelfModifyingFirstRegister + i;
      // New programs begin at the configured rates. Existing programs retain
      // inherited values, which p_memory_mu_const may evolve.
      if (initialize_working_memory) {
         scalar_memory->const_memory_[reg](0, 0) =
             SelfModifyingProbabilityToRawTendency(values[i]);
         scalar_memory->working_memory_[reg](0, 0) =
             scalar_memory->const_memory_[reg](0, 0);
      } else {
         scalar_memory->const_memory_[reg](0, 0) =
             SanitizeSelfModifyingRawTendency(scalar_memory->const_memory_[reg](0, 0));
      }
   }

   // Neutral decoy register: seed to the mean initial rate so its drift is
   // directly comparable to the rate registers, then leave it untouched by any
   // rate-consuming code path.
   if (scalar_memory->n_memories_ > kSelfModifyingDecoyRegister) {
      if (initialize_working_memory) {
         double mean_initial_rate = 0.0;
         for (size_t i = 0; i < kSelfModifyingRegisterCount; ++i) {
            mean_initial_rate += values[i];
         }
         mean_initial_rate /= static_cast<double>(kSelfModifyingRegisterCount);
         scalar_memory->const_memory_[kSelfModifyingDecoyRegister](0, 0) =
             SelfModifyingProbabilityToRawTendency(mean_initial_rate);
         scalar_memory->working_memory_[kSelfModifyingDecoyRegister](0, 0) =
             scalar_memory->const_memory_[kSelfModifyingDecoyRegister](0, 0);
      } else {
         scalar_memory->const_memory_[kSelfModifyingDecoyRegister](0, 0) =
             SanitizeSelfModifyingRawTendency(
                 scalar_memory->const_memory_[kSelfModifyingDecoyRegister](0, 0));
      }
   }
   use_evolved_const_ = true;
}

void RegisterMachine::SeedSelfModifyingWorkingFromConstants(
    std::unordered_map<std::string, std::any> &params) {
   if (!SelfModifyingEnabled(params)) return;

   auto* scalar_memory = private_memory_[MemoryEigen::kScalarType_];
   if (scalar_memory->n_memories_ < kSelfModifyingMinScalarRegisters) {
      die(__FILE__, __FUNCTION__, __LINE__,
          "self_modifying requires scalar registers S0-S6.");
   }

   for (size_t i = 0; i < kSelfModifyingRegisterCount; ++i) {
      const size_t reg = kSelfModifyingFirstRegister + i;
      scalar_memory->working_memory_[reg] = scalar_memory->const_memory_[reg];
   }
   if (scalar_memory->n_memories_ > kSelfModifyingDecoyRegister) {
      scalar_memory->working_memory_[kSelfModifyingDecoyRegister] =
          scalar_memory->const_memory_[kSelfModifyingDecoyRegister];
   }
}

void RegisterMachine::CopyPrivateConstToWorkingMemoryPreservingSelfModifyingRegisters() {
   auto* scalar_memory = private_memory_[MemoryEigen::kScalarType_];
   const size_t first_reg = kSelfModifyingFirstRegister;
   const size_t end_reg = std::min(
       scalar_memory->n_memories_, kSelfModifyingDecoyRegister + 1);

   std::vector<MatrixDynamic> preserved;
   preserved.reserve(end_reg > first_reg ? end_reg - first_reg : 0);
   for (size_t reg = first_reg; reg < end_reg; ++reg) {
      preserved.push_back(scalar_memory->working_memory_[reg]);
   }

   CopyPrivateConstToWorkingMemory();

   for (size_t reg = first_reg; reg < end_reg; ++reg) {
      scalar_memory->working_memory_[reg] = preserved[reg - first_reg];
   }
}

void RegisterMachine::CopySelfModifyingOutputs(
    const RegisterMachine &source,
    std::unordered_map<std::string, std::any> &params) {
   if (!SelfModifyingEnabled(params)) return;

   auto* destination = private_memory_[MemoryEigen::kScalarType_];
   auto* source_memory = source.private_memory_[MemoryEigen::kScalarType_];
   if (destination->n_memories_ < kSelfModifyingMinScalarRegisters ||
       source_memory->n_memories_ < kSelfModifyingMinScalarRegisters) {
      die(__FILE__, __FUNCTION__, __LINE__,
          "self_modifying requires scalar registers S0-S6.");
   }

   const bool inherit_outputs_as_constants =
       LamarckismEvolvedConstantsEnabled(params) &&
       SelfModifyingConstantsEvolvable(params);
   for (size_t i = 0; i < kSelfModifyingRegisterCount; ++i) {
      const size_t reg = kSelfModifyingFirstRegister + i;
      const double output = SanitizeSelfModifyingRawTendency(
          source_memory->working_memory_[reg](0, 0));
      destination->working_memory_[reg](0, 0) = output;
      if (inherit_outputs_as_constants) {
         destination->const_memory_[reg](0, 0) = output;
      }
  }

   // Inherit the neutral decoy phenotype the same way as the rate registers.
   if (destination->n_memories_ > kSelfModifyingDecoyRegister &&
       source_memory->n_memories_ > kSelfModifyingDecoyRegister) {
      destination->working_memory_[kSelfModifyingDecoyRegister] =
          source_memory->working_memory_[kSelfModifyingDecoyRegister];
      if (inherit_outputs_as_constants) {
         destination->const_memory_[kSelfModifyingDecoyRegister] =
             source_memory->working_memory_[kSelfModifyingDecoyRegister];
      }
   }
}

void RegisterMachine::CopyObservationToMemoryBuff(state* obs, size_t mem_t) {
    auto memory_size = private_memory_[0]->memory_size_;

    int n_col = mem_t == MemoryEigen::kVectorType_ ? 1 : memory_size;

    MatrixDynamic mat(memory_size, n_col);
    
    if (mem_t == MemoryEigen::kVectorType_) {  // copy obs to vector
      int f = obs_index_ % obs->dim_;
      for (size_t row = 0; row < memory_size; row++) {
         mat(row, 0) = obs->stateValueAtIndex(f % obs->dim_);
         f++;
      }
   } else if (mem_t == MemoryEigen::kMatrixType_) {  // copy obs to matrix
      int fm = obs_index_ % obs->dim_;
      for (size_t row = 0; row < memory_size; row++) {
         for (size_t col = 0; col < memory_size; col++) {
            mat(row, col) = obs->stateValueAtIndex(fm % obs->dim_);
            fm++;
         }
      }
   } 
    AddToInputMemoryBuff(mat, mem_t);
}

void RegisterMachine::Run(EvalData& eval_data, int &time_step, const size_t &graph_depth,
                          bool &verbose) {

   bool any_stateless = false;
   for (const auto &per_type : register_stateful_) {
      for (bool f : per_type) {
         if (!f) { any_stateless = true; break; }
      }
      if (any_stateless) break;
   }

   if (!stateful_ || any_stateless) {
      if (use_evolved_const_) {
         CopyPrivateConstToWorkingMemory();
      } else {
         ClearWorkingMemory();
      }
   }

   bool copied_obs_vec = false;
   bool copied_obs_mat = false;

   //PrintInstructions(true);

   for (auto istr : instructions_effective_) {
      istr->out_ = private_memory_[istr->GetOutType()];
      istr->outIdxE_ = istr->outIdx_ % istr->out_->n_memories_;
      // Defensive checkpoint compatibility: S6 writes are no-ops even if an
      // older serialized program still addresses the decoy.
      if (self_modifying_ &&
          IsSelfModifyingDecoyAccess(istr->GetOutType(), istr->outIdx_,
                                     n_memories_)) {
         continue;
      }
      for (size_t in = 0; in < 2; in++) {
         // Check if this input is used in the operation.
         if (istr->GetInType(in) != -1) {
            if (istr->IsMemoryRef(in)) {
               istr->SetInMem(in, private_memory_[istr->GetInType(in)]);
               istr->SetInIdxE(
                   in, istr->GetInIdx(in) % istr->GetInMem(in)->n_memories_);

               // Input is a memory ref. Track read time
               istr->GetInMem(in)->read_time_[istr->GetInIdxE(in)] =
                   time_step + (graph_depth / MAX_GRAPH_DEPTH);
            } else {  // Input is an observation reference
               istr->SetInMem(in, observation_memory_buff_[istr->GetInType(in)]);
               istr->SetInIdxE(in, istr->GetInIdx(in) % istr->GetInMem(in)->n_memories_);
               // Copy to obs buff only once.
               if (istr->GetInType(in) == MemoryEigen::kVectorType_ && !copied_obs_vec) {
                  CopyObservationToMemoryBuff(eval_data.obs, MemoryEigen::kVectorType_);
                  copied_obs_vec = true;
               } else if (istr->GetInType(in) == MemoryEigen::kMatrixType_ && !copied_obs_mat) {
                  CopyObservationToMemoryBuff(eval_data.obs, MemoryEigen::kMatrixType_);
                  copied_obs_mat = true;
               }
            }
            // This copies scalar input data to temporary scalar variables
            if (istr->GetInType(in) == MemoryEigen::kScalarType_) {
               istr->SetupScalarIn(in, eval_data.obs);
               if (self_modifying_ && istr->IsMemoryRef(in) &&
                   IsSelfModifyingDecoyAccess(istr->GetInType(in),
                                              istr->GetInIdx(in), n_memories_)) {
                  (in == 0 ? istr->scalar_in1_ : istr->scalar_in2_) = 0.0;
               }
            }
         }
      }
      // Track write times for temporal memory
      istr->out_->write_time_[istr->outIdxE_] =
          time_step + (graph_depth / MAX_GRAPH_DEPTH);  
      MatrixDynamic prev_out;
      if (operations_additive_) {
         prev_out = istr->out_->working_memory_[istr->outIdxE_];
      }
      istr->exec(eval_data);  // Execute instruction
      if (operations_additive_) {
         istr->out_->working_memory_[istr->outIdxE_] += prev_out;
      }
   }
   bid_val_ =
       private_memory_[MemoryEigen::kScalarType_]
           ->working_memory_[kBidRegister](0, 0);
   // bid_val_ = 1;
}

void RegisterMachine::RunCoordination(EvalData& eval_data, int &time_step, const size_t &graph_depth,
                          bool &verbose) {
   bool any_stateless = false;
   for (const auto &per_type : register_stateful_) {
      for (bool f : per_type) {
         if (!f) { any_stateless = true; break; }
      }
      if (any_stateless) break;
   }

   if (!stateful_ || any_stateless) {
      if (use_evolved_const_) {
         CopyPrivateConstToWorkingMemory();
      } else {
         ClearWorkingMemory();
      }
   }

   if (last_activation_episode_ != eval_data.episode) {
       ResetActivationMatrix();
       last_activation_episode_ = eval_data.episode;
   }

   bool copied_obs_vec = false;
   bool copied_obs_mat = false;

   // std::cout << "\n=== Program before execution ===\n";
   //PrintInstructions(true);
   bool coord_active = false;


   //if modulation is turned on via settings, make M0 all ones
   if (matrix_modulation) {
      auto* Mat = private_memory_[MemoryEigen::kMatrixType_];
      if (Mat && Mat->n_memories_ > 0) {
         Mat->working_memory_[0].setOnes();
      }
   }
   // if via settings, modulation is not allowed and coordination is, always turn on coord_active
   if (coordination_allowed && !matrix_modulation) {
      coord_active = true;
   }

   for (auto istr : instructions_effective_) {
      // Debug: print the instruction before it is used
      // std::cout << "\n[Next Instruction] " << istr->ToString() << std::endl;

      if (self_modifying_ &&
          IsSelfModifyingDecoyAccess(istr->GetOutType(), istr->outIdx_,
                                     n_memories_)) {
         continue;
      }

      // Only update/print the activation matrix when coordination is active
      if (coord_active) {
         UpdateActivationMatrix(istr);
         // std::cout << "[Activation Matrix]" << std::endl;
         // std::cout << register_activations << std::endl;
      }

      // Set output memory and effective destination index
      istr->out_ = private_memory_[istr->GetOutType()];
      if (istr->GetOutType() == MemoryEigen::kScalarType_) {
         // Before M0 is written, behave normally; after that, use coordination's chosen column
         if (coord_active) {
            istr->outIdxE_ = chosen_col_ % istr->out_->n_memories_;
         } else {
            istr->outIdxE_ = istr->outIdx_ % istr->out_->n_memories_;
         }
      } else {
         istr->outIdxE_ = istr->outIdx_ % istr->out_->n_memories_;
      }

      // Setup inputs (unchanged from standard Run logic)
      for (size_t in = 0; in < 2; in++) {
         // Check if this input is used in the operation.
         if (istr->GetInType(in) != -1) {
            if (istr->IsMemoryRef(in)) {
               istr->SetInMem(in, private_memory_[istr->GetInType(in)]);
               istr->SetInIdxE(
                   in, istr->GetInIdx(in) % istr->GetInMem(in)->n_memories_);

               // Input is a memory ref. Track read time
               istr->GetInMem(in)->read_time_[istr->GetInIdxE(in)] =
                   time_step + (graph_depth / MAX_GRAPH_DEPTH);
            } else {  // Input is an observation reference
               istr->SetInMem(in,
                              observation_memory_buff_[istr->GetInType(in)]);
               istr->SetInIdxE(
                   in, istr->GetInIdx(in) % istr->GetInMem(in)->n_memories_);
               // Copy to obs buff only once.
               if (istr->GetInType(in) == MemoryEigen::kVectorType_ &&
                   !copied_obs_vec) {
                  CopyObservationToMemoryBuff(eval_data.obs, MemoryEigen::kVectorType_);
                  copied_obs_vec = true;
               } else if (istr->GetInType(in) == MemoryEigen::kMatrixType_ &&
                          !copied_obs_mat) {
                  CopyObservationToMemoryBuff(eval_data.obs, MemoryEigen::kMatrixType_);
                  copied_obs_mat = true;
               }
            }
            // This copies scalar input data to temporary scalar variables
            if (istr->GetInType(in) == MemoryEigen::kScalarType_) {
               istr->SetupScalarIn(in, eval_data.obs);
               if (self_modifying_ && istr->IsMemoryRef(in) &&
                   IsSelfModifyingDecoyAccess(istr->GetInType(in),
                                              istr->GetInIdx(in), n_memories_)) {
                  (in == 0 ? istr->scalar_in1_ : istr->scalar_in2_) = 0.0;
               }
            }
         }
      }

      // Track write times for temporal memory
      istr->out_->write_time_[istr->outIdxE_] =
          time_step + (graph_depth / MAX_GRAPH_DEPTH);
      MatrixDynamic prev_out;
      if (operations_additive_) {
         prev_out = istr->out_->working_memory_[istr->outIdxE_];
      }
      istr->exec(eval_data);
      if (operations_additive_) {
         istr->out_->working_memory_[istr->outIdxE_] += prev_out;
      }

      if ((istr->GetOutType() == MemoryEigen::kMatrixType_) && (matrix_modulation == 1)) {
         size_t outN = istr->out_->n_memories_;
         size_t effIdx = istr->outIdxE_ % outN;
         if (effIdx == 0) {
            coord_active = true; // turn on coordination
            MatrixDynamic &dstMat = istr->out_->working_memory_[effIdx];
            SigmoidInPlace(dstMat);
            MatrixDynamic act_factor = register_activations.unaryExpr(
                [](double a) {
                   return std::pow(1.1, a); // singalGP and GRN constant
                });

            // Finally apply the modulated activations to the destination.
            MultiplyActivationWith(act_factor, dstMat);
         }
      }
   }

   // Preserve existing placeholder bid behaviour in this path
   bid_val_ = 1;
   //ResetActivationMatrix();
}


void RegisterMachine::SetupMemory(std::unordered_map<std::string, int>& state,
                                  int n_memories, int memory_size, mt19937& rng) {
   // Clear any previous state (safety if this is ever reused)
   private_memory_.clear();
   observation_memory_buff_.clear();
   register_stateful_.clear();

   register_stateful_.resize(MemoryEigen::kNumMemoryType_);

   for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
      private_memory_.push_back(
          new MemoryEigen(mem_t, n_memories, memory_size));
      private_memory_.back()->RandomizeConst(rng);

      observation_memory_buff_.push_back(
          new MemoryEigen(mem_t, observation_buff_size_, memory_size));

      // All initial registers follow the global parameter
      register_stateful_[mem_t].assign(
          static_cast<size_t>(n_memories), stateful_ != 0);
   }

   if (coordination_allowed || matrix_modulation) {
      register_activations.resize(n_memories, n_memories);
      register_activations.setOnes();
   } else {
      register_activations.resize(0, 0);
   }
}

void RegisterMachine::ResizeMemory(
    std::unordered_map<std::string, std::any>& params,
    std::unordered_map<std::string, int>& state, int new_memory_size, int slot_size, mt19937& rng) {
   for (auto* m : private_memory_) {
      m->memory_size_ = new_memory_size;
      m->n_memories_ = slot_size;
      m->ResizeMemory();
      m->RandomizeConst(rng);
   }
   for (auto* m : observation_memory_buff_) {
      m->memory_size_ = new_memory_size;
      m->ResizeMemory();
   }
   ConfigureSelfModifyingRegisters(params, false);
   SeedSelfModifyingWorkingFromConstants(params);
}

void RegisterMachine::ResizeMemorySlots(
    std::unordered_map<std::string, std::any>& params,
    std::unordered_map<std::string, int>& state, int memory_size, int new_slot_size, mt19937& rng) {
   for (auto* m : private_memory_) {
      m->memory_size_ = memory_size;
      m->n_memories_ = new_slot_size;
      m->ResizeMemory();
   }
   for (auto* m : observation_memory_buff_) {
      m->memory_size_ = memory_size;
      m->ResizeMemory();
   }
   if (coordination_allowed || matrix_modulation) {
      const int nmem = new_slot_size;
      if (register_activations.rows() != nmem || register_activations.cols() != nmem) {
         register_activations.resize(nmem, nmem);
         register_activations.setOnes();
      }
   } else {
      if (register_activations.size() != 0) {
         register_activations.resize(0, 0);
      }
   }
   ConfigureSelfModifyingRegisters(params, false);
   SeedSelfModifyingWorkingFromConstants(params);
}

// void RegisterMachine::MutateObsBuffSize(size_t max_observation_buff_size,
//                                         mt19937 &rng) {
//   std::uniform_int_distribution<> dis(1, max_observation_buff_size - 1);
//   auto prev = observation_buff_size_;
//   do {
//     observation_buff_size_ = dis(rng);
//   } while (observation_buff_size_ == prev);
//   // ResizeMemory();
// }

void RegisterMachine::MutateMemorySize(
    std::unordered_map<std::string, std::any> &params,
    std::unordered_map<std::string, int> &state, mt19937 &rng) {

      int min_size = std::any_cast<int>(params["min_memory_size"]);
    int max_size = std::any_cast<int>(params["max_memory_size"]);
   std::uniform_int_distribution<> dis(
       min_size,max_size);
   size_t old_size = private_memory_[MemoryEigen::kScalarType_]->memory_size_;
   size_t new_size;

   int old_slot_size = static_cast<int>(
       private_memory_[MemoryEigen::kScalarType_]->n_memories_);
   do {
      new_size = dis(rng);
   } while (new_size == old_size);
   if (min_size == max_size) {
        if (old_size == static_cast<size_t>(min_size)) {
            return;  // no possible mutation
        } else {
            ResizeMemory(params, state, min_size, old_slot_size, rng);
            return;
        }
    }
}

void RegisterMachine::MutateMemorySlots(
    std::unordered_map<std::string, std::any>& params,
    std::unordered_map<std::string, int>& state, mt19937& rng) {

   const bool self_modifying = SelfModifyingEnabled(params);
   const int min_size = self_modifying
       ? std::max(std::any_cast<int>(params["min_memory_slots"]),
                  static_cast<int>(kSelfModifyingMinScalarRegisters))
       : std::any_cast<int>(params["min_memory_slots"]);
   const int max_size = std::any_cast<int>(params["max_memory_slots"]);
   if (self_modifying &&
       max_size < static_cast<int>(kSelfModifyingMinScalarRegisters)) {
      die(__FILE__, __FUNCTION__, __LINE__,
          "self_modifying requires max_memory_slots to be at least seven.");
   }

   int old_size = static_cast<int>(private_memory_[MemoryEigen::kScalarType_]->n_memories_);
   bool gene_duplicate_on = std::any_cast<int>(params["gene_clone"]);

   auto randomize_matrix_inplace = [&](MatrixDynamic &M) {
      std::uniform_real_distribution<double> ur(-1.0, 1.0);
      for (int r = 0; r < M.rows(); ++r) {
         for (int c = 0; c < M.cols(); ++c) {
            M(r, c) = ur(rng);
         }
      }
   };

   std::uniform_int_distribution<int> step_dist(-1, 1);

   // SIZE CHANGE
   int step = 0;
   do {step = step_dist(rng);} while (step == 0);

   int proposed = old_size + step;
   int new_size = std::clamp(proposed, min_size, max_size);

   if (new_size == old_size) { return;}

   if (register_stateful_.size() < MemoryEigen::kNumMemoryType_) {
      register_stateful_.resize(MemoryEigen::kNumMemoryType_);
   }

   // --- DELETION OF A GENE ---
   int deleted_idx = -1;
   if (new_size < old_size) {
      if (old_size <= min_size) {
         return;
      }

      auto norm_mod = [](int v, int m) -> int {
         int r = v % m;
         if (r < 0) r += m;
         return r;
      };

      // Count register usage: reads + writes.
      std::vector<size_t> usage(static_cast<size_t>(old_size), 0);
      for (auto* ins : instructions_) {
         // Destination usage (write)
         usage[static_cast<size_t>(norm_mod(ins->outIdx_, old_size))]++;

         // Source usage (read) only when the input is a memory reference
         for (int in = 0; in < 2; ++in) {
            if (ins->GetInType(in) != -1 && ins->IsMemoryRef(in)) {
               int idx = norm_mod(ins->GetInIdx(in), old_size);
               usage[static_cast<size_t>(idx)]++;
            }
         }
      }

      const double eps = 1.0;
      const double alpha = 1.5;
      //const double ratio = static_cast<double>(min_size) / static_cast<double>(old_size);
      //const double action_register_penalty = ratio;
      std::vector<double> weights(static_cast<size_t>(old_size), 0.0);
      for (int i = 0; i < old_size; ++i) {
         if (self_modifying &&
             (i == static_cast<int>(kActionRegister) ||
              (i >= static_cast<int>(kSelfModifyingFirstRegister) &&
               i < static_cast<int>(kSelfModifyingRequiredRegisterCount)) ||
              i == static_cast<int>(kSelfModifyingDecoyRegister))) {
            continue;
         }
         const double u = static_cast<double>(usage[static_cast<size_t>(i)]);
         double w = 1.0 / std::pow(u + eps, alpha);
         // if (i < min_size) {
         //    w *= action_register_penalty;
         // }
         weights[static_cast<size_t>(i)] = w;
      }

      std::discrete_distribution<int> pick(weights.begin(), weights.end());
      deleted_idx = pick(rng);

      // Delete the "module" for that register: remove any instruction that writes to OR reads from it.
      // (Gene deletion analogue: remove the gene and all its dependencies for symmetric shrinkage.)
      for (size_t i = 0; i < instructions_.size();) {
         instruction* ins = instructions_[i];
         bool uses_deleted = (norm_mod(ins->outIdx_, old_size) == deleted_idx);

         // Also check reads - remove instructions that read from the deleted register
         for (int in = 0; in < 2 && !uses_deleted; ++in) {
            if (ins->IsMemoryRef(in)) {
               int idx = norm_mod(ins->GetInIdx(in), old_size);
               if (idx == deleted_idx) {
                  uses_deleted = true;
               }
            }
         }

         if (uses_deleted) {
            delete ins;
            instructions_.erase(instructions_.begin() + static_cast<long>(i));
            continue;
         }
         ++i;
      }
   }

   // --- GROWTH OF THE GENOME: when enabled, snapshot a base register to clone ---
   int clone_base_idx = -1; 
   std::vector<MatrixDynamic> clone_base_const;
   if (new_size > old_size && old_size > 0 && gene_duplicate_on) {
      std::uniform_int_distribution<int> pick_base(0, old_size - 1);
      clone_base_idx = pick_base(rng);

      // Snapshot the base constants for all memory types so we can copy into the new slot.
      clone_base_const.resize(MemoryEigen::kNumMemoryType_);
      for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
         auto* bank = private_memory_[mem_t];
         if (bank && clone_base_idx >= 0 &&
             static_cast<size_t>(clone_base_idx) < bank->const_memory_.size()) {
            clone_base_const[mem_t] = bank->const_memory_[static_cast<size_t>(clone_base_idx)];
         }
      }
   }

   // --- Remap indices in remaining instructions to keep them in-range and consistent ---
   //DOUBLE CHECK HOW THE REMAPPING IS WORKING EVEN THO WE DELETE (DOES IT EVEN MATTER)
   if (new_size < old_size && deleted_idx >= 0) {
      // If something still references the deleted slot, redirect reads to a safe hidden register
      // if available, otherwise to register 0.
      int fallback = (min_size < new_size) ? min_size : 0;

      auto remap = [&](int &idx) {
         // Only remap indices that are within the old range. If idx is larger, clamp later.
         if (idx == deleted_idx) {
            idx = fallback;
         }
         if (idx > deleted_idx) {
            idx -= 1;
         }
         idx = std::clamp(idx, 0, std::max(0, new_size - 1));
      };

      for (auto* ins : instructions_) {
         remap(ins->outIdx_);

         // Only remap input indices when that input is a memory reference.
         // (Observation indices and element/offset indices should not be remapped here.)
         if (ins->IsMemoryRef(0)) {
            remap(ins->in0Idx_);
         }
         if (ins->IsMemoryRef(1)) {
            remap(ins->in1Idx_);
         }

         ins->memIndices_ = new_size;
      }
   }

   int memory_size = static_cast<int>(private_memory_[MemoryEigen::kScalarType_]->memory_size_);
   ResizeMemorySlots(params, state, memory_size, new_size, rng);

   n_memories_ = new_size;

   // --- Initialize the new slot's constants/working memory ---
   if (new_size > old_size) {
      const int new_reg_idx = new_size - 1;
      for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
         auto* bank = private_memory_[mem_t];
         if (!bank) continue;
         if (static_cast<size_t>(new_reg_idx) < bank->const_memory_.size()) {
            if (gene_duplicate_on && clone_base_idx >= 0 &&
                static_cast<size_t>(mem_t) < clone_base_const.size() &&
                clone_base_const[mem_t].size() > 0) {
                bank->const_memory_[static_cast<size_t>(new_reg_idx)] = clone_base_const[mem_t];
            } else {
               randomize_matrix_inplace(bank->const_memory_[static_cast<size_t>(new_reg_idx)]);
            }
         }
         if (static_cast<size_t>(new_reg_idx) < bank->working_memory_.size()) {
            if (static_cast<size_t>(new_reg_idx) < bank->const_memory_.size()) {
               bank->working_memory_[static_cast<size_t>(new_reg_idx)] =
                   bank->const_memory_[static_cast<size_t>(new_reg_idx)];
            } else {
               bank->working_memory_[static_cast<size_t>(new_reg_idx)].setZero();
            }
         }
      }
   }

   // --- On addition, perform gene-duplication style cloning for scalar registers (existing behavior) ---
   if (new_size > old_size && gene_duplicate_on && !instructions_.empty()) {
      auto* Scal = private_memory_[MemoryEigen::kScalarType_];
      int n_scalar = static_cast<int>(Scal->n_memories_);

      // New scalar register is the last slot after resize
      int new_reg_idx = n_scalar - 1;

      // Need at least one pre-existing register to clone from
      if (n_scalar > 1) {
         int base_idx = (clone_base_idx >= 0 && clone_base_idx <= (n_scalar - 2))
                          ? clone_base_idx
                          : std::uniform_int_distribution<int>(0, n_scalar - 2)(rng);

         // Snapshot the original instruction count so we insert clones
         // immediately after their source instruction (gene-duplication style).
         const size_t orig_count = instructions_.size();

         struct PendingInsert {
            size_t pos;            // insertion position in the ORIGINAL sequence
            instruction* instr;    // the cloned instruction to insert
         };
         std::vector<PendingInsert> pending;
         pending.reserve(orig_count);

         for (size_t i = 0; i < orig_count; ++i) {
            auto* ins = instructions_[i];
            bool uses_base = false;
            instruction* clone = new instruction(*ins);

            auto norm_mod = [](int v, int m) -> int {
               int r = v % m;
               if (r < 0) r += m;
               return r;
            };

            // Check destination (use modulo old_size to match execution-time semantics)
            if (clone->GetOutType() == MemoryEigen::kScalarType_ &&
                norm_mod(clone->outIdx_, old_size) == base_idx) {
               clone->outIdx_ = new_reg_idx;
               uses_base = true;
            }

            // Check scalar memory sources (also modulo old_size)
            for (int in = 0; in < 2; ++in) {
               if (clone->GetInType(in) == MemoryEigen::kScalarType_ &&
                   clone->IsMemoryRef(in)) {

                  int &idx_ref = (in == 0) ? clone->in0Idx_ : clone->in1Idx_;
                  if (norm_mod(idx_ref, old_size) == base_idx) {
                     idx_ref = new_reg_idx;
                     uses_base = true;
                  }
               }
            }

            if (uses_base) {
               clone->memIndices_ = n_memories_;
               pending.push_back({i + 1, clone}); // insert right after source ins
            } else {
               delete clone;
            }
         }

         size_t allowable = pending.size();
         int max_prog_size = std::any_cast<int>(params["max_prog_size"]);
         const size_t cur = instructions_.size();
         const size_t cap = (cur < static_cast<size_t>(max_prog_size))
               ? (static_cast<size_t>(max_prog_size) - cur): 0;
         allowable = std::min(allowable, cap);
         for (size_t k = allowable; k < pending.size(); ++k) {
            delete pending[k].instr;
         }
         pending.resize(allowable);

         size_t inserted = 0;
         for (const auto& p : pending) {
            const size_t pos = p.pos + inserted;
            if (pos <= instructions_.size()) {
               instructions_.insert(instructions_.begin() + static_cast<long>(pos), p.instr);
               ++inserted;
            } else {
               // Fallback: if something goes wrong with indexing, append.
               instructions_.push_back(p.instr);
            }
         }
      }
   }
}


// void RegisterMachine::ChangeStatefulFlag(){
//       const double p_hidden_stateful_mutation =
//        std::any_cast<double>(params["p_hidden_stateful_mutation"]);
//    const bool allow_flag_randomization = (p_hidden_stateful_mutation > 0.0);


//    // --- Update per-register stateful flags (must remove the correct slot, not always the last) ---
//    std::bernoulli_distribution reg_stateful_dist(0.5);
//    for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; ++mem_t) {
//       auto &flags = register_stateful_[mem_t];

//       if (static_cast<int>(flags.size()) < old_size) {
//          flags.resize(static_cast<size_t>(old_size), stateful_ != 0);
//       }

//       if (new_size < old_size) {
//          if (deleted_idx >= 0 && deleted_idx < static_cast<int>(flags.size())) {
//             flags.erase(flags.begin() + deleted_idx);
//          } else {
//             // Fallback: if something is inconsistent, trim from the end.
//             flags.resize(static_cast<size_t>(new_size));
//          }
//       } else if (new_size > old_size) {
//          if (!allow_flag_randomization) {
//             flags.resize(static_cast<size_t>(new_size), stateful_ != 0);
//          } else {
//             size_t extra = static_cast<size_t>(new_size - old_size);
//             for (size_t k = 0; k < extra; ++k) {
//                bool is_stateful_new = reg_stateful_dist(rng);
//                flags.push_back(is_stateful_new);
//             }
//          }
//       }

//       if (flags.size() > static_cast<size_t>(new_size)) {
//          flags.resize(static_cast<size_t>(new_size));
//       }
//       if (flags.size() < static_cast<size_t>(new_size)) {
//          flags.resize(static_cast<size_t>(new_size), stateful_ != 0);
//       }
//    }
// }


//Sinkhorn-Knopp algorithm (ensures everything equals 1). However, constraint then is that there should be atleast
//one positive number per row and column. We can solve this by a slack variable.
// https://gist.github.com/MurageKibicho/955133f061926b3666d314359a868081 
void RegisterMachine::UpdateActivationMatrix(auto ins){
   if (!coordination_allowed && !matrix_modulation) return;
   if (register_activations.size() == 0) return;
   // Only scalar destinations are relevant
   if (ins->GetOutType() != MemoryEigen::kScalarType_) return;

   auto* Scal = private_memory_[MemoryEigen::kScalarType_];
   const size_t N   = Scal->n_memories_;
   const size_t dst = static_cast<size_t>(ins->outIdx_) % N;
   if (self_modifying_ && dst == kSelfModifyingDecoyRegister) return;
   // Keep activation matrix dimensions consistent with scalar register count
   if (register_activations.rows() != static_cast<int>(N) ||
       register_activations.cols() != static_cast<int>(N)) {
       register_activations.resize(N, N);
       register_activations.setOnes();
   }
   // sources 
   std::vector<size_t> srcs; srcs.reserve(2);
   for (int in = 0; in < 2; ++in) {
      if (ins->GetInType(in) == MemoryEigen::kScalarType_ && ins->IsMemoryRef(in)) {
         const size_t src = static_cast<size_t>(ins->GetInIdx(in)) % N;
         if (!(self_modifying_ && src == kSelfModifyingDecoyRegister)) {
            srcs.push_back(src);
         }
      }
   }
   if (srcs.empty()) return;

   auto row_near_uniform = [&](size_t r) -> bool {
      const int rr = static_cast<int>(r);
      double mx   = register_activations.row(rr).maxCoeff();
      double mn   = register_activations.row(rr).minCoeff();
      double mean = register_activations.row(rr).mean();
      // Treat the row as "uninformative" if all entries are (near) equal.
      // Relative tolerance scales with the row's magnitude; abs guard handles tiny means.
      const double tol = std::max(1e-9, 1e-3 * std::abs(mean));
      return (mx - mn) <= tol;
   };

   // Return (argmax_col, max_val). Tie-breaker prefers current dst column.
   auto row_argmax = [&](size_t r) -> std::pair<size_t, double> {
      size_t best_c = dst;
      double best_v = register_activations(r, best_c);
      for (size_t c = 0; c < N; ++c) {
         if (self_modifying_ && c == kSelfModifyingDecoyRegister) continue;
         double v = register_activations(r, c);
         if (v > best_v) { best_v = v; best_c = c; }
      }
      return {best_c, best_v};
   };

   std::vector<size_t> chosen_cols;
   chosen_cols.reserve(2);

   if (srcs.size() == 1) {
      auto [c0, v0] = row_argmax(srcs[0]);
      if (row_near_uniform(srcs[0])) {
         chosen_cols.push_back(dst);
      } else {
         chosen_cols.push_back(c0);
      }
   } else {
      bool both_all_uniform = row_near_uniform(srcs[0]) && row_near_uniform(srcs[1]);
      if (both_all_uniform) {
         // Both rows are essentially uniform: use the destination column only
         chosen_cols.push_back(dst);
      } else {
         auto [c0, v0] = row_argmax(srcs[0]);
         auto [c1, v1] = row_argmax(srcs[1]);

         constexpr double tie_tol = 1e-12;
         if (std::fabs(v1 - v0) <= tie_tol) {
            // Tie between sources: use both columns (if distinct)
            chosen_cols.push_back(c0);
            if (c1 != c0) {
               chosen_cols.push_back(c1);
            }
         } else if (v1 > v0) {
            chosen_cols.push_back(c1);
         } else {
            chosen_cols.push_back(c0);
         }
      }
   }

   if (chosen_cols.empty()) {
      // Should not happen, but guard against it.
      return;
   }

   for (size_t r : srcs) {
      for (size_t c : chosen_cols) {
         double base_v = register_activations(r, c);
         double new_v  = base_v + learning_rate_;
         register_activations(r, c) = new_v;
      }
      RowSoftmaxClampInPlace(register_activations, r, 1e-12);
   }

   // Use the first chosen column as the effective destination index for routing.
   chosen_col_ = chosen_cols.front();
}

void RegisterMachine::ResetActivationMatrix(){
   if (register_activations.size() == 0) return;
   register_activations.setOnes();
}
