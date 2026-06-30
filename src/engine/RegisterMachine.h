#ifndef RegisterMachine_h
#define RegisterMachine_h

#include "instruction.h"

class EvalData;

#define MAX_GRAPH_DEPTH 1000.0

class RegisterMachine {
  public:
   bool from_string_ = false;        // TODO(skelly):remove
   bool use_evolved_const_ = false;  // Fixed parameter: whether to use constants
   bool stateful_ = false;           // Fixed parameter: whether memories maintain state
   int last_activation_episode_ = -1;
   int action_ = -1;                 // Mutable: discrete action
   std::vector<std::vector<bool>> register_stateful_;
   double learning_rate_ = 0.0; //individual learning rate of program
   double initial_heb_noise_ = 0.0;
   bool coordination_allowed = false;
   bool matrix_modulation = false;
   bool operations_additive_ = false;
   bool self_modifying_ = false;     // Reserves S2-S5 for rates and isolates neutral S6
   int obs_index_ = 0;              // Mutable: index into observation space
   int observation_buff_size_ = 1;  // Mutable: observation buff size
   int n_memories_ = 1; // Mutable: memory slots 
   
   MatrixDynamic register_activations;
   size_t chosen_col_ = 0; //dynamic selection

   std::vector<instruction *> instructions_;
   std::vector<instruction *> instructions_effective_;
   vector<int> op_counts_;  // Count each op in instructions_effective_
   int n_effective_registers_ = 0;

   void MutateRegisterStatefulFlags(mt19937 &rng);
   void ChangeStatefulFlag();
   // Vector storing 1 MemoryEigen* of each type (SCALAR, VECTOR, MATRIX)
   vector<MemoryEigen *> private_memory_;

   // Vector storing id of each private memory (required for ToString)
   vector<long> private_memory_ids_;

   // Vector storing 1 MemoryEigen* of each type (SCALAR, VECTOR, MATRIX)
   vector<MemoryEigen *> observation_memory_buff_;

   // Team memory
   //sharedMemoryEigen *team_memory_;

   long id_ = -1;             // Unique id
   double bid_val_ = 0.0;      // Temporarily store the most recent bid value
   set<long> features_;  // Features indexed by this RegisterMachine
   long gtime_ = 0;          // Generation created
   int nrefs_ = 0;           // Number of references by teams

   // Create arbitrary RegisterMachine
   RegisterMachine(long action,
                   int team_obs_index, std::unordered_map<std::string, std::any> &params,
                   std::unordered_map<std::string, int> &state, mt19937 &rng,
                   std::vector<bool> &legalOps);

   // Create RegisterMachine and copy instructions
   // If inherited_n_memories > 0, use that value instead of sampling from config
   RegisterMachine(long action, std::vector<instruction*> &instructions,
                   std::unordered_map<std::string, std::any>& params,
                   std::unordered_map<std::string, int>& state, mt19937& rng,
                   std::vector<bool>& legal_ops, int inherited_n_memories = -1);

   // Copy contructor
   RegisterMachine(RegisterMachine &rm);

   // Copy assignment operator
   RegisterMachine &operator=(RegisterMachine &rm);

    // Clone RegisterMachine with new id
   RegisterMachine(RegisterMachine &plr,
                   std::unordered_map<std::string, std::any> &params,
                   std::unordered_map<std::string, int> &state,
                   mt19937 &rng);

   // Create RegisterMachine from checkpoint string
   RegisterMachine(std::vector<std::string> outcomeFields,
                   std::vector<std::map<long, MemoryEigen *>> &memory_maps,
                   std::unordered_map<std::string, std::any> &params,
                   mt19937 &rng);

   ~RegisterMachine();

   //inline void SetMemory(sharedMemoryEigen *team_memory) { team_memory_ = team_memory; }

   inline void AddToInputMemoryBuff(MatrixDynamic &mat, int mem_t) {
      observation_memory_buff_[mem_t]->working_memory_.push_front(mat);
      observation_memory_buff_[mem_t]->working_memory_.pop_back();
   }

   inline void ClearWorkingMemory() {
      // Make sure the flag structure matches the memory banks
      if (register_stateful_.size() < private_memory_.size()) {
         register_stateful_.resize(private_memory_.size());
      }

      // Reset only *stateless* private registers
      for (size_t mem_t = 0; mem_t < private_memory_.size(); ++mem_t) {
         auto* memory = private_memory_[mem_t];
         auto &flags  = register_stateful_[mem_t];

         const size_t n = memory->n_memories_;
         if (flags.size() < n) {
            // Default any missing flags to the global parameter
            flags.resize(n, stateful_ != 0);
         }

         for (size_t i = 0; i < n; ++i) {
            if (!flags[i]) {
               // Stateless: reset each step
               memory->working_memory_[i].setZero();
            }
         }
      }

      // Observation buffer stays fully stateless (always cleared)
      for (auto memory : observation_memory_buff_) memory->ClearWorking();
   }

   inline void CopyPrivateConstToWorkingMemory() {
      // Make sure the flag structure matches the memory banks
      if (register_stateful_.size() < private_memory_.size()) {
         register_stateful_.resize(private_memory_.size());
      }

      // Reset only *stateless* private registers to their evolved constants
      for (size_t mem_t = 0; mem_t < private_memory_.size(); ++mem_t) {
         auto* memory = private_memory_[mem_t];
         auto &flags  = register_stateful_[mem_t];

         const size_t n = memory->n_memories_;
         if (flags.size() < n) {
            flags.resize(n, stateful_ != 0);
         }

         for (size_t i = 0; i < n; ++i) {
            if (!flags[i]) {
               memory->working_memory_[i] = memory->const_memory_[i];
            }
         }
      }
   }

   // Reset ordinary stateless registers at an episode boundary while retaining
   // the self-modifying rate phenotype in S2-S5 and its neutral S6 control.
   void CopyPrivateConstToWorkingMemoryPreservingSelfModifyingRegisters();
   
   void UpdateActivationMatrix(auto ins);

   void CopyObservationToMemoryBuff(state *obs, size_t memory_type);

   // Determine which observation variables of used in effetive instructions,
   // store in features_
   void MarkFeatures(instruction *istr, int in);

   // Determine which instructions are effective,
   // store in instructions_effective_
   void MarkIntrons(std::unordered_map<std::string, std::any> &params_, int n_memories);
   double ComputeMeanPairwiseRegisterSimilarity() const;
   double ComputeMaxPairwiseRegisterSimilarity() const;

   // Execute this register machine
   void Run(EvalData& eval_data, int &time_step, const size_t &graph_depth, bool &verbose);
   void RunCoordination(EvalData& eval_data, int &time_step, const size_t &graph_depth, bool &verbose);

   std::string ToString(bool);
   std::string ToStringMemory();
   void PrintInstructions(bool effective_only) const;
   void ResetActivationMatrix();

   void Mutate(std::unordered_map<std::string, std::any> &params,
               std::unordered_map<std::string, int> &state, mt19937 &rng,
               std::vector<bool> &legal_ops);

   void MutateMemorySize(std::unordered_map<std::string, std::any> &params,
                         std::unordered_map<std::string, int> &state,
                         mt19937 &rng);

   void MutateMemorySlots(std::unordered_map<std::string, std::any> &params,
                         std::unordered_map<std::string, int> &state,
                         mt19937 &rng);

   void ResizeMemory(std::unordered_map<std::string, std::any> &params,
                     std::unordered_map<std::string, int> &state,
                     int new_memory_size, int slot_size, mt19937 &rng);

   void ResizeMemorySlots(std::unordered_map<std::string, std::any> &params,
                     std::unordered_map<std::string, int> &state,
                     int memory_size, int new_slot_size, mt19937 &rng);

   void ConfigureSelfModifyingRegisters(
       std::unordered_map<std::string, std::any> &params,
       bool initialize_working_memory = true);

   void SeedSelfModifyingWorkingFromConstants(
       std::unordered_map<std::string, std::any> &params);

   void CopySelfModifyingOutputs(
       const RegisterMachine &source,
       std::unordered_map<std::string, std::any> &params);

   void SetupMemory(std::unordered_map<std::string, int> &state, int n_memories,
                    int memory_size, mt19937 &rng);

   // Copy constants from prog to this register machine
   void CopyEvolvedConstants(RegisterMachine &prog) {
      for (size_t m = 0; m < private_memory_.size(); m++) {
         for (size_t i = 0; i < private_memory_[m]->n_memories_; i++) {
            private_memory_[m]->const_memory_[i] =
                prog.private_memory_[m]->const_memory_[i];
         }
      }
   }
};

struct RegisterMachineBidLexicalCompare {
   bool operator()(RegisterMachine *rm1, RegisterMachine *rm2) const {
      if (rm1->bid_val_ != rm2->bid_val_) {
         return rm1->bid_val_ > rm2->bid_val_;
      } else {
         return rm1->id_ < rm2->id_;
      }
   }
};

struct RegisterMachineIdComp {
   bool operator()(RegisterMachine *rm1, RegisterMachine *rm2) const {
      return rm1->id_ < rm2->id_;
   }
};

#endif
