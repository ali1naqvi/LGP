/****************************************************************************/
// // linear crossover with two parent programs (i.e. two teams with one program
// // each)
// if (crossover && GetParam<double>("p_atomic") == 1.0 && pm1->size() == 1 &&
//     pm2->size() == 1 &&
//     real_dist_(_rngs[TPG_SEED]) <
//         GetParam<double>("p_instructions_xover")) {
//   phylo_graph_[(*cm)->id_].ancestorIds.insert(pm2->id_);
//   (*cm)->addAncestorId(pm2->id_);

//   linearCrossover = true;
//   vector<program *> pm1Programs;
//   pm1->members(pm1Programs);
//   vector<program *> pm2Programs;
//   pm1->members(pm2Programs);
//   linearM *c1;  // = NULL;
//   linearM *c2;  // = NULL;
//   programCrossover(dynamic_cast<linearM *>(pm1Programs[0]),
//                    dynamic_cast<linearM *>(pm2Programs[0]), &c1, &c2,
//                    _rngs[TPG_SEED]);

//   cm2 = new team(GetState("t_current"), state_["team_count"]++);
//   numNewTeams++;

//   phylo_graph_[(cm2)->id_].ancestorIds.insert(pm1->id_);
//   (cm2)->addAncestorId(pm1->id_);
//   phylo_graph_[(cm2)->id_].ancestorIds.insert(pm2->id_);
//   (cm2)->addAncestorId(pm2->id_);

//   if (real_dist_(_rngs[TPG_SEED]) < 0.5) {
//     (*cm)->addProgram(c1);
//     (cm2)->addProgram(c2);
//   } else {
//     (*cm)->addProgram(c2);
//     (cm2)->addProgram(c1);
//   }

//   addTeam(cm2);
//   _Mroot.insert(cm2);
//   phylo_graph_.insert(pair<long, phyloRecord>(cm2->id_, phyloRecord()));
//   phylo_graph_[cm2->id_].gtime = GetState("t_current");
//   phylo_graph_[cm2->id_].root = true;
// }
/****************************************************************************/

/****************************************************************************/
// // team crossover
// if (crossover && (pm1->size() > 1 || pm2->size() > 1)) {
//   phylo_graph_[(*cm)->id_].ancestorIds.insert(pm2->id_);
//   (*cm)->addAncestorId(pm2->id_);

//   pm2->getMembersRef(p2programs);
//   auto p2liter = p2programs->begin();
//   while (p1liter != p1programs->end() || p2liter != p2programs->end()) {
//     if (p1liter != p1programs->end() &&
//         (int)(*cm)->size() < GetParam<int>("max_team_size") &&
//         (((*p1liter)->action_ < 0 && (*cm)->numAtomic_ < 1) ||
//          find(p2programs->begin(), p2programs->end(), *p1liter) !=
//              p2programs->end()))
//       (*cm)->addProgram(*p1liter);
//     else if ((int)(*cm)->size() < GetParam<int>("max_team_size") &&
//              p1liter != p1programs->end() &&
//              real_dist_(_rngs[TPG_SEED]) < 0.5)
//       (*cm)->addProgram(*p1liter);
//     if ((int)(*cm)->size() < GetParam<int>("max_team_size") &&
//         p2liter != p2programs->end() &&
//         real_dist_(_rngs[TPG_SEED]) < 0.5)
//       (*cm)->addProgram(*p2liter);
//     if (p1liter != p1programs->end()) p1liter++;
//     if (p2liter != p2programs->end()) p2liter++;
//   }
//   if ((*cm)->numAtomic_ < 1)
//     die(__FILE__, __FUNCTION__, __LINE__,
//         "Crossover must leave the fail-safe atomic program!");
// } else
//   for (p1liter = p1programs->begin(); p1liter != p1programs->end();
//   p1liter++)
//     (*cm)->addProgram(*p1liter);
/****************************************************************************/

// if (instructions_.size() != instructions_effective_.size()) {
//   cerr << endl << "dbg Mark instructions_s " << instructions_.size() << " bidE_s "
//        << instructions_effective_.size() << endl;

//   cerr << "instructions_ exec" << endl;
//   for (auto i : instructions_) {
//     i->exec(true);
//     cerr << endl;
//   }
//   cerr << "exec done" << endl;

//   cerr << "bidE_ exec" << endl;
//   for (auto i : instructions_effective_) {
//     i->exec(true);
//     cerr << endl;
//   }
//   cerr << "exec done" << endl;
// }

/******************************************************************************/
double linearM::run(state *obs, int timeStep, int graphDepth, mt19937 &rng) {
  (void)rng;
  bool dbg = false;
  // ofstream dbg_file("prog-"+std::to_string(id_) + ".txt");
  // if (dbg) dbg_file << "id: " << id_ << " run:" << endl;

  // reset memory
  if (!stateful_) CopySharedConstToWorking();

  private_memory_[memoryEigen::kScalarType_]->working_memory_[0].setZero();
  private_memory_[memoryEigen::kScalarType_]->working_memory_[1].setZero();

  for (auto istr : instructions_effective_) {
    // read inputs
    size_t idx = 0;
    for (size_t in = 0; in < 2; in++) {
      if (istr->inType(in) != memoryEigen::kNAType) {
        if (istr->IsInput(in)) {
          // if (dbg) dbg_file << "in" << in << " fRef ";
          // in this case inMem(in) will be inputMemory_ and we use index 0
          if (istr->inType(in) == memoryEigen::kScalarType_)
            istr->inMem(in)->working_memory_[idx](0, 0) =
                obs->stateValueAtIndex(istr->inIdx(in));
          else if (istr->inType(in) == memoryEigen::kVectorType_)
            for (size_t f = istr->inIdx(in), row = 0;
                 row < istr->inMem(in)->memoryRows(); row++)
              istr->inMem(in)->working_memory_[idx](row, 0) =
                  obs->stateValueAtIndex(
                      f++ % num_input_);  //(*feature)[f++ % num_input_];
          else if (istr->inType(in) == memoryEigen::kMatrixType_)
            for (size_t f = istr->inIdx(in), row = 0;
                 row < istr->inMem(in)->memoryRows(); row++)
              for (size_t col = 0; col < istr->inMem(in)->memoryCols(); col++)
                istr->inMem(in)->working_memory_[idx](row, col) =
                    obs->stateValueAtIndex(f++ % num_input_);
          istr->inIdxE(in, idx);  // reset inIdxE to zero for input ref
        } else {                  // this input is a memory ref
          // if (dbg) dbg_file << "in" << in << " mRef ";
          // track read time for temporal memory
          istr->inMem(in)->getReadTimeE()(istr->inIdx(in), 0) =
              timeStep + (graphDepth / MAX_GRAPH_DEPTH);
        }
      }
    }
    // if (dbg) dbg_file << endl;
    // track write times for temporal memory
    istr->out_->getWriteTimeE()(istr->outIdx_, 0) =
        timeStep + (graphDepth / MAX_GRAPH_DEPTH);

    istr->exec(dbg);
  }
  // if (dbg) {
  //   dbg_file << "id: " << id_ << " outs ";
  //   dbg_file << private_memory_[memoryEigen::kScalarType_]
  //               ->working_memory_[0](0, 0);
  //   dbg_file << " ";
  //   dbg_file << private_memory_[memoryEigen::kScalarType_]
  //               ->working_memory_[1](0, 0);
  //   dbg_file << endl;
  // }
  // dbg_file.close();
  return private_memory_[memoryEigen::kScalarType_]->working_memory_[0](0, 0);
}

// /******************************************************************************
//  * Markus F. Brameier and Wolfgang Banzhaf. 2010.
//  * Linear Genetic Programming (1st. ed.). Springer Publishing Company, Inc.
//  * Algorithm 3.1 (detection of structural introns)
//  */
// void linearM::MarkIntrons(std::unordered_map<std::string, std::any> &params)
// {
//   fill(op_counts_.begin(), op_counts_.end(), 0);  // count occurance of each
//   op

//   // keep track of which memories are effective with a map
//   map<int, vector<bool> > Reff;  // maps [memory type][index]->true/false

//   for (int mem_t = 0; mem_t < memoryEigen::kNumMemoryType_; mem_t++)
//     Reff[mem_t] =
//         vector<bool>(std::any_cast<int>(params["n_memories"]), false);

//   Reff[memoryEigen::kScalarType_][0] = true;  // mark bid output memory

//   if (std::any_cast<int>(params["continuous_output"]))
//     Reff[memoryEigen::kScalarType_][1] = true;  // mark continuous output
//     memory

//   features_.clear();
//   instructions_effective_.clear();

//   // From last to first instruction.
//   // vector<instruction *>::reverse_iterator riter;
//   for (auto riter = instructions_.rbegin(); riter != instructions_.rend(); riter++) {
//     if (!skipIntrons_ || Reff[(*riter)->outType()][(*riter)->outIdx_]) {
//       instructions_effective_.insert(instructions_effective_.begin(), *riter);
//       op_counts_[(*riter)->op_]++;
//       // output TODO(spkelly) this is always true now
//       (*riter)->out_ = private_memory_[(*riter)->outType()];
//       // inputs
//       for (int in = 0; in < 2; in++) {
//         // if this input is actually used for this op
//         if ((*riter)->inType(in) != memoryEigen::kNAType) {
//           if ((*riter)->IsInput(in)) {  // this input is a feature ref
//             (*riter)->inMem(in,
//             inputMemoryPointers_[in][(*riter)->inType(in)]);
//             // mark features (accounting, should have no effect behaviour)
//             if ((*riter)->inType(in) == memoryEigen::kScalarType_) {
//               features_.insert((*riter)->inIdx(in));
//             } else if ((*riter)->inType(in) == memoryEigen::kVectorType_) {
//               for (size_t f = (*riter)->inIdx(in), row = 0;
//                    row < (*riter)->inMem(in)->memoryRows(); row++) {
//                 features_.insert(f++ % num_input_);  // toroidal
//               }
//             } else if ((*riter)->inType(in) == memoryEigen::kMatrixType_) {
//               for (size_t f = (*riter)->inIdx(in), row = 0;
//                    row < (*riter)->inMem(in)->memoryRows(); row++) {
//                 for (size_t col = 0; col < (*riter)->inMem(in)->memoryCols();
//                      col++) {
//                   features_.insert(f++ % num_input_);  // toroidal
//                 }
//               }
//             }
//           } else {  // this input is a memory ref
//             (*riter)->inMem(in,
//             private_memory_[(*riter)->inType(in)]);
//             Reff[(*riter)->inType(in)][(*riter)->inIdx(in)] = true;
//           }
//         }
//       }
//     }
//   }
// }





// #include "RegisterMachine.h"
// #include "EvalData.h"

// #include <stacktrace>

// /******************************************************************************/
// std::string RegisterMachine::ToString(bool effective_only) {
//    std::ostringstream oss;
//    oss << "RegisterMachine:" << id_ << ":" << gtime_ << ":" << action_ << ":"
//        << stateful_ << ":" << nrefs_ << ":" << observation_buff_size_ << ":"
//        << learning_rate_ << ":" << initial_heb_noise_ << ":" << obs_index_;
//    // for (auto &i : private_memory_ids_) oss << ":" << i;
//    auto prog = effective_only ? instructions_effective_ : instructions_;
//    for (auto istr : prog) oss << ":" << istr->ToString();
//    oss << endl;
//    for (auto* m : private_memory_)
//       oss << m->ToString(id_);
//    return oss.str();
// }

// /******************************************************************************/
// std::string RegisterMachine::ToStringMemory() {
//    std::ostringstream oss;
//    for (auto* m : private_memory_)
//       oss << m->ToString(id_);
//    return oss.str();
// }

// void RegisterMachine::PrintInstructions(bool effective_only) const {
//     const auto& base = effective_only ? instructions_effective_ : instructions_;
//     const bool used_effective = effective_only && !instructions_effective_.empty();
//     const auto& instrs = used_effective ? instructions_effective_ : instructions_;

//     auto TypeLabel = [](int memType) -> const char* {
//         static const char* lbl[] = {"S","V","M"};
//         if (memType < 0 || memType > 2) return "?";
//         return lbl[memType];
//     };

//     auto RegLabel = [&](int memType, size_t idx) -> std::string {
//         std::ostringstream os; os << TypeLabel(memType) << idx; return os.str();
//     };

//     auto PrintMatrixCompact = [&](const MatrixDynamic& mat) {
//         size_t R = mat.rows();
//         size_t C = mat.cols();
//         std::cout.setf(std::ios::fixed); std::cout.precision(4);
//         if (R == 1 && C == 1) {
//             std::cout << mat(0,0);
//             return;
//         }
//         if (C == 1) {
//             std::cout << "[";
//             size_t show = std::min<size_t>(8, R);
//             for (size_t r = 0; r < show; ++r) {
//                 std::cout << mat(r,0);
//                 if (r + 1 < show) std::cout << ", ";
//             }
//             if (R > show) std::cout << ", …";
//             std::cout << "]";
//             return;
//         }
//         size_t showR = std::min<size_t>(3, R);
//         size_t showC = std::min<size_t>(3, C);
//         std::cout << "[";
//         for (size_t r = 0; r < showR; ++r) {
//             if (r) std::cout << "; ";
//             std::cout << "[";
//             for (size_t c = 0; c < showC; ++c) {
//                 std::cout << mat(r,c);
//                 if (c + 1 < showC) std::cout << ", ";
//             }
//             if (C > showC) std::cout << ", …";
//             std::cout << "]";
//         }
//         if (R > showR) std::cout << "; …";
//         std::cout << "]";
//     };

//     auto OperandLabel = [this](const instruction* inst, int in) -> std::string {
//         if (inst->IsMemoryRef(in)) {
//             int memType = inst->GetInType(in);
//             int idx = inst->GetInIdx(in) % private_memory_[memType]->n_memories_;
//             static const char* lbl[] = {"S","V","M"};
//             std::ostringstream os; os << lbl[memType] << idx; return os.str();
//         } else if (inst->IsObs(in)) {
//             int t = inst->GetInType(in);
//             size_t total = 0;
//             if (t == MemoryEigen::kScalarType_) {
//                 total = static_cast<size_t>(observation_buff_size_);
//             } else {
//                 total = observation_memory_buff_[t]->n_memories_;
//             }
//             int raw = inst->GetInIdx(in);
//             int idx = (total > 0) ? (raw % static_cast<int>(total)) : 0;
//             static const char* lbl[] = {"S","V","M"};
//             std::ostringstream os;
//             if (total > 0) {
//                 os << "OBS" << raw << "->obs_" << lbl[t] << "[" << idx << "/" << total << "]";
//             } else {
//                 os << "OBS" << raw << "->obs_" << lbl[t] << "[" << idx << "]";
//             }
//             return os.str();
//         } else {
//             return std::string("IMM");
//         }
//     };

//     std::cout << "prog: " << id_ << endl;
//     std::cout << "discrete action: "<<  action_ << endl; 
//     std::cout << "  instr_count: " << instrs.size();
//     std::cout << "  effective: " << (used_effective ? 1 : 0);
//     if (effective_only && !used_effective) std::cout << " (fallback)";
//     std::cout << std::endl;

//     {
//         std::cout.setf(std::ios::fixed);
//         std::cout.precision(4);
//         std::cout << "evolved: lr=" << learning_rate_
//                   << "  init_heb_noise=" << initial_heb_noise_
//                   << "  use_evolved_const=" << (use_evolved_const_ ? 1 : 0)
//                   << "  obs_index=" << obs_index_
//                   << "  obs_buff_size=" << observation_buff_size_
//                   << std::endl;
//     }

//     if (use_evolved_const_) {
//         std::cout << "[Evolved Constants]" << std::endl;
//         for (int t = 0; t < MemoryEigen::kNumMemoryType_; ++t) {
//             const auto* bank = private_memory_[t];
//             for (size_t idx = 0; idx < bank->n_memories_; ++idx) {
//                 std::cout << RegLabel(t, idx) << ": ";
//                 PrintMatrixCompact(bank->const_memory_[idx]);
//                 std::cout << std::endl;
//             }
//         }
//         std::cout << std::endl;
//     }
//     auto RenderImmSuffix = [&](const instruction* ins) -> std::string {
//         std::ostringstream os;
//         bool has = false;
//         for (int k = 0; k < 2; ++k) {
//             if (ins->GetInType(k) != -1 && !ins->IsMemoryRef(k) && !ins->IsObs(k)) {
//                 if (!has) { os << "  {"; has = true; }
//                 os << (k==0?"imm0":"imm1") << "=" << ins->GetInIdx(k);
//                 os << "; ";
//             }
//         }
//         if (has) {
//             std::string s = os.str();
//             if (!s.empty() && s.back() == ' ') { s.pop_back(); }
//             if (!s.empty() && s.back() == ';') { s.pop_back(); }
//             return s + "}";
//         }
//         return std::string();
//     };

//     auto RenderInstr = [&](const instruction* ins) -> std::string {
//         std::ostringstream os;
//         std::string op_name = instruction::op_names_[ins->op_];

//         std::string in0 = (ins->GetInType(0) != -1) ? OperandLabel(ins, 0) : std::string("-");
//         std::string in1 = (ins->GetInType(1) != -1) ? OperandLabel(ins, 1) : std::string("-");

//         os << op_name << "(" << in0;
//         if (ins->GetInType(1) != -1) os << ", " << in1;
//         os << ")";

//         std::string imm = RenderImmSuffix(ins);
//         if (!imm.empty()) os << imm;

//         if (op_name == "SCALAR_VECTOR_ASSIGN_OP") {
//             os << "  {i=" << ins->in1Idx_ << "}";
//         }

//         return os.str();
//     };

//     auto key64 = [](int t, size_t i){ return (uint64_t(t) << 32) | uint64_t(i); };
//     std::unordered_map<uint64_t, std::string> resident_instr; 

//     for (size_t i = 0; i < instrs.size(); ++i) {
//         const auto* ins = instrs[i];
//         int tOut = ins->GetOutType();
//         size_t oIdx = ins->outIdx_ % private_memory_[tOut]->n_memories_;
//         resident_instr[key64(tOut, oIdx)] = RenderInstr(ins);
//     }

//     std::cout << "[Register Instructions]" << std::endl;
//     for (size_t t = 0; t < MemoryEigen::kNumMemoryType_; ++t) {
//         const auto* bank = private_memory_[t];
//         for (size_t idx = 0; idx < bank->n_memories_; ++idx) {
//             auto it = resident_instr.find(key64((int)t, idx));
//             std::cout << "[(U)" << RegLabel((int)t, idx) << "] = "
//                       << (it == resident_instr.end() ? std::string("-") : it->second)
//                       << std::endl;
//         }
//     }

//     std::cout << "\n[Instructions]" << std::endl;
//     for (size_t i = 0; i < instrs.size(); ++i) {
//         const auto* instr = instrs[i];
//         int outType = instr->GetOutType();
//         size_t outIdx = instr->outIdx_ % private_memory_[outType]->n_memories_;

//         std::string in0 = (instr->GetInType(0) != -1) ? OperandLabel(instr, 0) : std::string("-");
//         std::string in1 = (instr->GetInType(1) != -1) ? OperandLabel(instr, 1) : std::string("-");

//         std::cout << "[" << i << "] "
//                   << RegLabel(outType, outIdx) << " = "
//                   << instruction::op_names_[instr->op_] << "(" << in0;
//         if (instr->GetInType(1) != -1) std::cout << ", " << in1;
//         std::cout << ")";
//         std::string imm = RenderImmSuffix(instr);
//         if (!imm.empty()) std::cout << imm;
//         std::cout << std::endl;
//     }
//     auto* Scal = private_memory_[MemoryEigen::kScalarType_];
//     if (Scal->n_memories_ >= 1) {
//         std::cout << "\n[Return] bid(s0): ";
//         PrintMatrixCompact(Scal->working_memory_[0]);
//         std::cout << std::endl;
//     }
//     if (Scal->n_memories_ >= 2) {
//         std::cout << "[Return] action(s1): ";
//         PrintMatrixCompact(Scal->working_memory_[1]);
//         std::cout << std::endl;
//     }
// }

// /******************************************************************************/
// // Create arbitrary RegisterMachine
// RegisterMachine::RegisterMachine(
//     long action, int team_obs_index, std::unordered_map<std::string, std::any>& params,
//     std::unordered_map<std::string, int>& state, mt19937& rng,
//     std::vector<bool>& legal_ops) {
//    action_ = action;
//    stateful_ = std::any_cast<int>(params["stateful"]);
//    gtime_ = state["t_current"];
//    id_ = state["program_count"]++;
//    nrefs_ = 0;

//    //before lr = 0.0 - 0.25
//    //before lr = 0.0 - 0.50
//    std::normal_distribution<double> lr_dist(0.25, 0.25);
//    std::lognormal_distribution<double> noise_dist(0.0, 0.5); 

//    learning_rate_ = std::any_cast<double>(params["static_learning_rate"]);
//    if (learning_rate_ == 0.0) {
//             learning_rate_    = std::clamp(lr_dist(rng), 0.0, 1.0);}
//    initial_heb_noise_ = std::clamp(noise_dist(rng), 0.0, 1.0);
      

//    observation_buff_size_ = std::any_cast<int>(params["observation_buff_size"]);
//    obs_index_ = team_obs_index;
//    uniform_int_distribution<int> disP(
//        std::any_cast<int>(params["min_initial_prog_size"]), std::any_cast<int>(params["max_initial_prog_size"]));
//    int prog_size = disP(rng);
//    for (int i = 0; i < prog_size; i++) {
//       auto in = new instruction(params, rng);
//       in->Mutate(true, legal_ops, observation_buff_size_, rng);
//       instructions_.push_back(in);
//    }
//    op_counts_.resize(instruction::NUM_OP);
//    if (!isEqual(std::any_cast<double>(params["p_memory_mu_const"]),
//                 0.0)) {
//       use_evolved_const_ = true;
//    }
//    SetupMemory(state, std::any_cast<int>(params["n_memories"]),
//                std::any_cast<int>(params["memory_size"]));
// }

// // Create RegisterMachine and copy instructions
// RegisterMachine::RegisterMachine(
//     long action, std::vector<instruction*> &instructions,
//     std::unordered_map<std::string, std::any> &params,
//     std::unordered_map<std::string, int> &state, mt19937 &rng,
//     std::vector<bool> &legal_ops) {
//    action_ = action;
//    stateful_ = std::any_cast<int>(params["stateful"]);
//    gtime_ = state["t_current"];
//    id_ = state["program_count"]++;
//    nrefs_ = 0;

   
//    std::lognormal_distribution<double> lr_dist(0.0, 0.25);
//    std::lognormal_distribution<double> noise_dist(0.0, 0.5); 
//    learning_rate_    = std::clamp(lr_dist(rng),    0.0, 1.0);
//    initial_heb_noise_ = std::clamp(noise_dist(rng), 0.0, 1.0);

//    observation_buff_size_ = std::any_cast<int>(params["observation_buff_size"]);
//    obs_index_ = 0;
//    for (auto i : instructions) {
//       instructions_.push_back(new instruction(*i));
//    }
//    op_counts_.resize(instruction::NUM_OP);
//    if (!isEqual(std::any_cast<double>(params["p_memory_mu_const"]),
//                 0.0)) {
//       use_evolved_const_ = true;
//    }
//    SetupMemory(state, std::any_cast<int>(params["n_memories"]),
//                std::any_cast<int>(params["memory_size"]));
// }

// // Copy Contructor
// RegisterMachine::RegisterMachine(RegisterMachine& rm) {
//    action_ = rm.action_;
//    gtime_ = rm.gtime_;
//    id_ = rm.id_;
//    obs_index_ = rm.obs_index_;
//    bid_val_ = -(numeric_limits<double>::max());
//    nrefs_ = 0;
//    stateful_ = rm.stateful_;
//    learning_rate_ = rm.learning_rate_;
//    initial_heb_noise_ = rm.initial_heb_noise_;
//    observation_buff_size_ = rm.observation_buff_size_;
//    use_evolved_const_ = rm.use_evolved_const_;
//    op_counts_.resize(instruction::NUM_OP);
//    for (size_t i = 0; i < MemoryEigen::kNumMemoryType_; i++) {
//       private_memory_.push_back(new MemoryEigen(*(rm.private_memory_[i])));
//       observation_memory_buff_.push_back(
//           new MemoryEigen(*(rm.observation_memory_buff_[i])));
//    }
//    private_memory_ids_ = rm.private_memory_ids_;
//    for (auto i : rm.instructions_) {
//       instructions_.push_back(new instruction(*i));
//    }
// }

// // Clone RegisterMachine with new id
// RegisterMachine::RegisterMachine(
//     RegisterMachine& rm, std::unordered_map<std::string, std::any>& params,
//     std::unordered_map<std::string, int>& state) {
//    action_ = rm.action_;
//    gtime_ = state["t_current"];
//    id_ = state["program_count"]++;
//    obs_index_ = rm.obs_index_;
//    bid_val_ = -(numeric_limits<double>::max());
//    nrefs_ = 0;
//    stateful_ = rm.stateful_;
//    observation_buff_size_ = rm.observation_buff_size_;
//    learning_rate_ = rm.learning_rate_;
//    initial_heb_noise_ = rm.initial_heb_noise_;
//    use_evolved_const_ = rm.use_evolved_const_;
//    op_counts_.resize(instruction::NUM_OP);
//    for (auto instr : rm.instructions_) {
//       instructions_.push_back(new instruction(*instr));
//    }
//    SetupMemory(state, std::any_cast<int>(params["n_memories"]),
//                rm.private_memory_[MemoryEigen::kScalarType_]->memory_size_);
//    CopyEvolvedConstants(rm);
// }

// // Create RegisterMachine from string
// RegisterMachine::RegisterMachine(
//     std::vector<std::string> outcomeFields,
//     std::vector<std::map<long, MemoryEigen*>>& memory_maps,
//     std::unordered_map<std::string, std::any>& params, mt19937& rng) {
//    int f = 1;
//    id_ = atoi(outcomeFields[f++].c_str());
//    gtime_ = atoi(outcomeFields[f++].c_str());
//    action_ = atoi(outcomeFields[f++].c_str());
//    stateful_ = atoi(outcomeFields[f++].c_str());
//    nrefs_ = atoi(outcomeFields[f++].c_str());
//    observation_buff_size_ = atoi(outcomeFields[f++].c_str());
//    learning_rate_ = stod(outcomeFields[f++].c_str());
//    initial_heb_noise_ = stod(outcomeFields[f++].c_str());
//    obs_index_ = atoi(outcomeFields[f++].c_str());
//    // SetupMemry() but from existing memory pointers
//    for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
//       // long id = atoi(outcomeFields[f++].c_str());
//       private_memory_.push_back(memory_maps[mem_t][id_]);
//       // private_memory_ids_.push_back(id);
//       auto memory_size = memory_maps[mem_t][id_]->memory_size_;
//       observation_memory_buff_.push_back(
//           new MemoryEigen(mem_t, observation_buff_size_, memory_size));
//    }
//    for (size_t ii = f; ii < outcomeFields.size(); ii++) {
//       vector<string> instructionString;
//       SplitString(outcomeFields[ii], '_', instructionString);
//       instruction* in = new instruction(params, rng);
//       in->in1Src_ = stringToInt(instructionString[0]);
//       in->in2Src_ = stringToInt(instructionString[1]);
//       in->outIdx_ = stringToInt(instructionString[2]);
//       in->op_ = stringToInt(instructionString[3]);
//       in->in0Idx_ = stringToInt(instructionString[4]);
//       in->in1Idx_ = stringToInt(instructionString[5]);
//       in->in2Idx_ = stringToInt(instructionString[6]);
//       in->in3Idx_ = stringToInt(instructionString[7]);
//       instructions_.push_back(in);
//    }
//    instructions_effective_ = instructions_;
//    op_counts_.resize(instruction::NUM_OP);
//    from_string_ = true;
// }

// RegisterMachine::~RegisterMachine() {
//    for (auto* i : instructions_)
//       delete i;
//    for (auto* m : private_memory_)
//       delete m;
//    for (auto* m : observation_memory_buff_)
//       delete m;
// }

// // TODO(skelly): WARNING: confirm this function works as expected.
// void RegisterMachine::MarkFeatures(instruction* istr, int in) {
//    // Setting input memory pointer is required for MarkIntrons.
//    istr->SetInMem(in, observation_memory_buff_[istr->GetInType(in)]);

//    features_.clear();
//    if (istr->GetInType(in) == MemoryEigen::kScalarType_) {
//       features_.insert(istr->GetInIdx(in));
//    } else if (istr->GetInType(in) == MemoryEigen::kVectorType_) {
//       for (size_t f = istr->GetInIdx(in), row = 0;
//            row < istr->GetInMem(in)->memory_size_; row++) {
//          // features_.insert(f++ % num_input_);  // toroidal
//          features_.insert(f++);
//       }
//    } else if (istr->GetInType(in) == MemoryEigen::kMatrixType_) {
//       for (size_t f = istr->GetInIdx(in), row = 0;
//            row < istr->GetInMem(in)->memory_size_; row++) {
//          for (size_t col = 0; col < istr->GetInMem(in)->memory_size_; col++) {
//             // features_.insert(f++ % num_input_);  // toroidal
//             features_.insert(f++);
//          }
//       }
//    }
// }

// void RegisterMachine::MarkIntrons(
//     std::unordered_map<std::string, std::any>& params) {
//    // memories_effective keeps track of which memories are effective, i.e. used
//    // in the program. memories_effective maps [memory type][index]->true/false.
//    auto n_memories = std::any_cast<int>(params["n_memories"]);
//    map<int, vector<bool>> memories_effective;
//    memories_effective[MemoryEigen::kScalarType_] =
//        vector<bool>(n_memories, false);
//    memories_effective[MemoryEigen::kVectorType_] =
//        vector<bool>(n_memories, false);
//    memories_effective[MemoryEigen::kMatrixType_] =
//        vector<bool>(n_memories, false);

//    // Mark bid output memory
//    memories_effective[MemoryEigen::kScalarType_][0] = true;

//    // Mark continuous output memory
//    if (std::any_cast<int>(params["continuous_output"]) == 1)
//       memories_effective[MemoryEigen::kScalarType_][1] = true;
//    else if (std::any_cast<int>(params["continuous_output"]) == 2)
//       memories_effective[MemoryEigen::kVectorType_][1] = true;
//    else if (std::any_cast<int>(params["continuous_output"]) == 3)
//       memories_effective[MemoryEigen::kMatrixType_][1] = true;

//    // Backward pass to find effective instructions when stateless
//    std::vector<instruction*> instructions_effective_stateless;
//    for (auto riter = instructions_.rbegin(); riter != instructions_.rend();
//         riter++) {
//       auto istr = *riter;
//       if (memories_effective[istr->GetOutType()][istr->outIdx_ % n_memories]) {
//          instructions_effective_stateless.push_back(istr);
//          for (int in = 0; in < 2; in++) {
//             if (istr->IsMemoryRef(in)) {
//                memories_effective[istr->GetInType(in)]
//                                  [istr->GetInIdx(in) % n_memories] = true;
//             }
//          }
//       }
//    }

//    // TODO(skelly): Is this the most efficient method? Currently O(n^2)
//    for (size_t t = 0; t < instructions_.size(); t++) {
//       instructions_effective_.clear();
//       // Count occurance of each op.
//       std::fill(op_counts_.begin(), op_counts_.end(), 0);
//       for (auto istr : instructions_) {
//          if (memories_effective[istr->GetOutType()]
//                                [istr->outIdx_ % n_memories] ||
//              std::find(instructions_effective_stateless.begin(),
//                        instructions_effective_stateless.end(),
//                        istr) != instructions_effective_stateless.end()) {
//             instructions_effective_.push_back(istr);
//             op_counts_[istr->op_]++;
//             for (int in = 0; in < 2; in++) {
//                if (istr->IsMemoryRef(in)) {
//                   memories_effective[istr->GetInType(in)]
//                                     [istr->GetInIdx(in) % n_memories] = true;
//                } else if (istr->IsObs(in)) {
//                   MarkFeatures(istr, in);
//                }
//             }
//          }
//       }
//    }
// }

// void RegisterMachine::Mutate(std::unordered_map<std::string, std::any> &params,
//                              std::unordered_map<std::string, int> &state,
//                              mt19937 &rng, vector<bool> &legal_ops) {
//    uniform_real_distribution<> dis_real(0, 1.0);

//       // Remove random instruction
//       if (instructions_.size() > 1 &&
//           dis_real(rng) <
//               std::any_cast<double>(params["p_instructions_delete"])) {
//          uniform_int_distribution<int> disBid(0, instructions_.size() - 1);
//          int i = disBid(rng);
//          delete *(instructions_.begin() + i);
//          instructions_.erase(instructions_.begin() + i);
//       }

//       // Insert a new random instruction
//       if ((int)instructions_.size() <
//               std::any_cast<int>(params["max_prog_size"]) &&
//           dis_real(rng) < std::any_cast<double>(params["p_instructions_add"])) {
//          instruction *instr = new instruction(params, rng);
//          instr->Mutate(true, legal_ops, observation_buff_size_, rng);
//          uniform_int_distribution<int> disBid(0, instructions_.size());
//          int i = disBid(rng);
//          instructions_.insert(instructions_.begin() + i, instr);
//       }

//       // Mutate a randomly selected instruction
//       if (dis_real(rng) <
//           std::any_cast<double>(params["p_instructions_mutate"])) {
            
//          uniform_int_distribution<int> disBid(0, instructions_.size() - 1);
//          auto i = disBid(rng);
//          instructions_[i]->Mutate(false, legal_ops,
//                                             observation_buff_size_, rng);
//       }

//       // Mutate constants
//       if (use_evolved_const_ && dis_real(rng) <
//           std::any_cast<double>(params["p_memory_mu_const"])) {
//          for (auto m : private_memory_) {
//             m->MutateConstants(rng);
//          }
//       }

      
//       std::uniform_real_distribution<double> coin(0.0, 1.0);
//       double mut_lr    = std::any_cast<double>(params["mut_learning_rate"]);

//       static constexpr double sd_meta_lr    = 0.1;
//       static constexpr double sd_meta_noise = 0.2;

//       if (coin(rng) < mut_lr) {
//          std::normal_distribution<double> gauss_lr(0.0, sd_meta_lr);
//          double factor = std::exp(gauss_lr(rng)); 
//          learning_rate_ = std::clamp(learning_rate_ * factor, 0.0, 1.0);
//       }


//       // Swap positions of two instructions
//       if (instructions_.size() > 1 &&
//           dis_real(rng) <
//               std::any_cast<double>(params["p_instructions_swap"])) {
//          uniform_int_distribution<int> disBid(0, instructions_.size() - 1);
//          int i = disBid(rng);
//          int j;
//          do {
//             j = disBid(rng);
//          } while (i == j);
//          std::swap(instructions_[i], instructions_[j]);
//       }

//       // Change observation buff size
//       // if (dis_real(rng) <
//       //     std::any_cast<double>(params["p_observation_buff_size"])) {
//       //   MutateObsBuffSize(std::any_cast<int>(params["max_observation_buff_size"]),
//       //                     rng);
//       // }

//       // Change memory size
//       if (dis_real(rng) < std::any_cast<double>(params["p_memory_size"])) {
//          MutateMemorySize(params, state, rng);
//       }

//       //Change observation index
//       if (dis_real(rng) <
//           std::any_cast<double>(params["p_observation_index"])) {
//          const int max_index = 1000000;  // TODO(skelly): fix magic #
//          uniform_int_distribution<int> dis(0, max_index);
//          auto prev = obs_index_;
//          do {
//             obs_index_ = dis(rng);
//          } while (obs_index_ == prev);
//       }
// }

// void RegisterMachine::CopyObservationToMemoryBuff(state* obs, size_t mem_t) {
//     auto memory_size = private_memory_[0]->memory_size_;

//     int n_col = mem_t == MemoryEigen::kVectorType_ ? 1 : memory_size;

//     MatrixDynamic mat(memory_size, n_col);
    
//     if (mem_t == MemoryEigen::kVectorType_) {  // copy obs to vector
//       int f = obs_index_ % obs->dim_;
//       for (size_t row = 0; row < memory_size; row++) {
//          mat(row, 0) = obs->stateValueAtIndex(f % obs->dim_);
//          f++;
//       }
//    } else if (mem_t == MemoryEigen::kMatrixType_) {  // copy obs to matrix
//       int fm = obs_index_ % obs->dim_;
//       for (size_t row = 0; row < memory_size; row++) {
//          for (size_t col = 0; col < memory_size; col++) {
//             mat(row, col) = obs->stateValueAtIndex(fm % obs->dim_);
//             fm++;
//          }
//       }
//    } 
//     AddToInputMemoryBuff(mat, mem_t);
// }

// void RegisterMachine::Run(EvalData& eval_data, int &time_step, const size_t &graph_depth,
//                           bool &verbose) {
//    // Clear working memory prior to execution, making this program stateless
//    if (!stateful_) {
//       if (use_evolved_const_) {
//             CopyPrivateConstToWorkingMemory();
//       } else {
//          ClearWorkingMemory();
//       }
//    }

//    bool copied_obs_vec = false;
//    bool copied_obs_mat = false;

//    // Local helpers for register/memory labeling and compact value printing
//    auto TypeLabel = [](int memType) -> const char* {
//        static const char* lbl[] = {"S","V","M"};
//        if (memType < 0 || memType > 2) return "?"; return lbl[memType];
//    };
//    auto RegLabel = [&](int memType, size_t idx) -> std::string {
//        std::ostringstream os; os << TypeLabel(memType) << idx; return os.str();
//    };
//    auto PrintMatrixCompact = [&](const MatrixDynamic& mat) {
//        size_t R = mat.rows(); size_t C = mat.cols();
//        std::cout.setf(std::ios::fixed); std::cout.precision(4);
//        if (R == 1 && C == 1) { std::cout << mat(0,0); return; }
//        if (C == 1) { // vector
//            std::cout << "[";
//            size_t show = std::min<size_t>(8, R);
//            for (size_t r = 0; r < show; ++r) { std::cout << mat(r,0); if (r+1<show) std::cout << ", "; }
//            if (R > show) std::cout << ", …"; std::cout << "]"; return;
//        }
//        size_t showR = std::min<size_t>(3, R), showC = std::min<size_t>(3, C);
//        std::cout << "[";
//        for (size_t r = 0; r < showR; ++r) {
//            if (r) std::cout << "; "; std::cout << "[";
//            for (size_t c = 0; c < showC; ++c) { std::cout << mat(r,c); if (c+1<showC) std::cout << ", "; }
//            if (C > showC) std::cout << ", …"; std::cout << "]";
//        }
//        if (R > showR) std::cout << "; …"; std::cout << "]";
//    };
//    auto OperandLabel = [this](const instruction* inst, int in) -> std::string {
//        if (inst->GetInType(in) == -1) return "-";
//        if (inst->IsMemoryRef(in)) {
//            int memType = inst->GetInType(in);
//            int idx = inst->GetInIdx(in) % private_memory_[memType]->n_memories_;
//            static const char* lbl[] = {"S","V","M"};
//            std::ostringstream os; os << lbl[memType] << idx; return os.str();
//        } else { // observation or immediate
//            if (inst->IsObs(in)) {
//                int t = inst->GetInType(in);
//                size_t n = observation_memory_buff_[t]->n_memories_;
//                int idx = n ? (inst->GetInIdx(in) % n) : 0;
//                static const char* lbl[] = {"S","V","M"};
//                std::ostringstream os; 
//                os << "OBS" << inst->GetInIdx(in) << "->obs_" << lbl[t] << "[" << idx << "]"; 
//                return os.str();
//            }
//            return std::string("IMM");
//        }
//    };

//    // Structures for tracking last writer per register for verbose mode
//    struct LastWriteInfo {
//        int op;                 // op code
//        int tOut;               // memory type of output
//        size_t oIdx;            // effective out index
//        std::string in0;        // rendered operand label 0
//        std::string in1;        // rendered operand label 1 (empty if unused)
//    };
//    auto key64 = [](int t, size_t i){ return (uint64_t(t) << 32) | uint64_t(i); };
//    std::unordered_map<uint64_t, LastWriteInfo> last_write; // last writer per (type,idx)

//    // Index-based loop for per-step tracing
//    for (size_t ii = 0; ii < instructions_effective_.size(); ++ii) {
//       auto* istr = instructions_effective_[ii];
//       istr->out_ = private_memory_[istr->GetOutType()];
//       istr->outIdxE_ = istr->outIdx_ % istr->out_->n_memories_;
//       for (size_t in = 0; in < 2; in++) {
//          if (istr->GetInType(in) != -1) {
//             if (istr->IsMemoryRef(in)) {
//                istr->SetInMem(in, private_memory_[istr->GetInType(in)]);
//                istr->SetInIdxE(in, istr->GetInIdx(in) % istr->GetInMem(in)->n_memories_);
//                istr->GetInMem(in)->read_time_[istr->GetInIdxE(in)] =
//                    time_step + (graph_depth / MAX_GRAPH_DEPTH);
//             } else {
//                istr->SetInMem(in, observation_memory_buff_[istr->GetInType(in)]);
//                istr->SetInIdxE(in, istr->GetInIdx(in) % istr->GetInMem(in)->n_memories_);
//                if (istr->GetInType(in) == MemoryEigen::kVectorType_ && !copied_obs_vec) {
//                    CopyObservationToMemoryBuff(eval_data.obs, MemoryEigen::kVectorType_); copied_obs_vec = true;
//                } else if (istr->GetInType(in) == MemoryEigen::kMatrixType_ && !copied_obs_mat) {
//                    CopyObservationToMemoryBuff(eval_data.obs, MemoryEigen::kMatrixType_); copied_obs_mat = true;
//                }
//             }
//             if (istr->GetInType(in) == MemoryEigen::kScalarType_) {
//                istr->SetupScalarIn(in, eval_data.obs);
//             }
//          }
//       }
//       istr->out_->write_time_[istr->outIdxE_] = time_step + (graph_depth / MAX_GRAPH_DEPTH);
//       istr->exec(eval_data);

//       if (verbose) {
//           int tOut = istr->GetOutType();
//           size_t oIdx = istr->outIdxE_;
//           LastWriteInfo info;
//           info.op = istr->op_;
//           info.tOut = tOut;
//           info.oIdx = oIdx;
//           info.in0 = OperandLabel(istr, 0);
//           info.in1 = (istr->GetInType(1) != -1) ? OperandLabel(istr, 1) : std::string();
//           last_write[key64(tOut,oIdx)] = std::move(info);
//       }
//    }
//    if (verbose) {
//        std::cout << "[Final Register Values]" << std::endl;
//        for (int t = 0; t < MemoryEigen::kNumMemoryType_; ++t) {
//            auto* bank = private_memory_[t];
//            for (size_t idx = 0; idx < bank->n_memories_; ++idx) {
//                auto it = last_write.find(key64(t, idx));
//                if (it == last_write.end()) continue; // only show registers that were written by some op
//                const auto& lw = it->second;
//                std::cout << RegLabel(t, idx) << " = "
//                          << instruction::op_names_[lw.op] << "(" << lw.in0;
//                if (!lw.in1.empty()) std::cout << ", " << lw.in1;
//                std::cout << ")  =>  " << RegLabel(t, idx) << ": ";
//                PrintMatrixCompact(bank->working_memory_[idx]);
//                std::cout << std::endl;
//            }
//        }
//    }
//    bid_val_ =
//        private_memory_[MemoryEigen::kScalarType_]->working_memory_[0](0, 0);
// }


// void RegisterMachine::SetupMemory(std::unordered_map<std::string, int>& state,
//                                   int n_memories, int memory_size) {
//    for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
//       private_memory_.push_back(
//           new MemoryEigen(mem_t, n_memories, memory_size));
//       private_memory_.back()->RandomizeConst();
//       observation_memory_buff_.push_back(
//           new MemoryEigen(mem_t, observation_buff_size_, memory_size));
//    }
// }

// void RegisterMachine::ResizeMemory(
//     std::unordered_map<std::string, std::any>& params,
//     std::unordered_map<std::string, int>& state, int new_memory_size) {
//    for (auto* m : private_memory_) {
//       m->memory_size_ = new_memory_size;
//       m->n_memories_ = std::any_cast<int>(params["n_memories"]);
//       m->ResizeMemory();
//       m->RandomizeConst();
//    }
//    for (auto* m : observation_memory_buff_) {
//       m->memory_size_ = new_memory_size;
//       m->n_memories_ = std::any_cast<int>(params["n_memories"]);
//       m->ResizeMemory();
//    }
// }

// // void RegisterMachine::MutateObsBuffSize(size_t max_observation_buff_size,
// //                                         mt19937 &rng) {
// //   std::uniform_int_distribution<> dis(1, max_observation_buff_size - 1);
// //   auto prev = observation_buff_size_;
// //   do {
// //     observation_buff_size_ = dis(rng);
// //   } while (observation_buff_size_ == prev);
// //   // ResizeMemory();
// // }

// void RegisterMachine::MutateMemorySize(
//     std::unordered_map<std::string, std::any>& params,
//     std::unordered_map<std::string, int>& state, mt19937& rng) {
//    std::uniform_int_distribution<> dis(
//        std::any_cast<int>(params["min_memory_size"]),
//        std::any_cast<int>(params["max_memory_size"]));
//    size_t old_size = private_memory_[MemoryEigen::kScalarType_]->memory_size_;
//    size_t new_size;
//    do {
//       new_size = dis(rng);
//    } while (new_size == old_size);
//    ResizeMemory(params, state, new_size);
// }

