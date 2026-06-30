#include <string>
#include "team.h"
#include "EvalData.h"

#include <algorithm>
#include <limits>
#include <cmath>

// Allow duplicates
// void team::AddProgram(RegisterMachine* prog, int position) {
//    prog->obs_index_ = obs_index_;
//    auto it = members_.begin();
//    advance(it, position);
//    members_.insert(it, prog);
//    if (prog->action_ < 0)
//       n_atomic_++;
//    members_run_.resize(members_.size());  // put in mark introns
//    prog->nrefs_++;
// }

void team::AddProgram(RegisterMachine* prog, int position) {
   if (std::any_of(members_.begin(), members_.end(),
                   [&](const RegisterMachine* p){ return p->id_ == prog->id_; }))
      return;
   prog->obs_index_ = obs_index_;
   auto it = members_.begin();
   std::advance(it, position);
   members_.insert(it, prog);
   if (prog->action_ < 0)
      n_atomic_++;
   members_run_.resize(members_.size());  // put in mark introns
   prog->nrefs_++;
}

/******************************************************************************/
string team::ToString() const {
   ostringstream oss;

   oss << "team:" << id_ << ":" << gtime_ << ":" << obs_index_ << ":" << lambda_td_ << ":" << decay_factor_ << ":" << _n_eval;
   for (auto prog : members_) {
      oss << ":" << prog->id_;
   }
   oss << endl;

   if (incomingPrograms_.size() > 0) {
      oss << "incoming_progs:" << id_;
      for (auto& ip : incomingPrograms_) {
         oss << ":" << ip;
      }
      oss << endl;
   }
   
   // if (!HebbianMap.weights.empty()) {
   //  for (const auto& [prevProgID, currMap] : HebbianMap.weights) {
   //      for (const auto& [currProgID, weight] : currMap) {
   //          oss << "weights:" << id_ << ":" << prevProgID << ":" << currProgID 
   //              << ":" << std::setprecision(4) << weight << "\n";
   //      }
   //  }
   //  oss << endl;
   // }
   
   return oss.str();
}


/******************************************************************************/
void team::InitMemory(map<long, team*>& teamMap,
                      std::unordered_map<std::string, std::any>& params) {
   set<team*, teamIdComp> teams;
   set<RegisterMachine*, RegisterMachineIdComp> RegisterMachines;
   GetAllNodes(teamMap, teams, RegisterMachines);
   const auto self_modifying = params.find("self_modifying");
   const bool use_self_modifying_constants = self_modifying != params.end() && std::any_cast<int>(self_modifying->second) != 0;
   for (auto prog : RegisterMachines) {
      if (use_self_modifying_constants ||
          !isEqual(std::any_cast<double>(params["p_memory_mu_const"]), 0.0)) {
         prog->use_evolved_const_ = true;
         prog->ConfigureSelfModifyingRegisters(params, false);
         if (use_self_modifying_constants) {
            // Preserve the rate phenotype generated in the preceding episode.
            prog->CopyPrivateConstToWorkingMemoryPreservingSelfModifyingRegisters();
         } else {
            prog->CopyPrivateConstToWorkingMemory();
         }
      } else {
         // Initialize working memory with zeros
         prog->use_evolved_const_ = false;
         prog->ClearWorkingMemory();
      }
   }
      if(std::any_cast<int>(params["reset_per_episode"]) == 1){
         for (auto* t : teams) {
            t->HebbianMap.resetWeights();

            }
      }
      for (auto* t : teams){
            t->HebbianMap.resetElig();} 
}

/******************************************************************************/
void team::clone(map<long, phyloRecord>& phyloGraph, team** tm) {
   phyloGraph[(*tm)->id_].ancestorIds.insert(id_);
   for (auto prog : members_) {
      (*tm)->AddProgram(prog);
   }
   (*tm)->fitnessBins(fitnessBins_);
   (*tm)->cloneId_ = id_;
   (*tm)->obs_index_ = obs_index_; //TODO(ali) double check
   clones_++;
}

/******************************************************************************/
void team::features(set<long>& F) const {
   if (F.empty() == false)
      die(__FILE__, __FUNCTION__, __LINE__, "feature set not empty");

   for (auto prog : members_)
      F.insert(prog->features_.begin(), prog->features_.end());
}


/******************************************************************************/
double team::novelty(int type, int kNN) const {
   multiset<double>::iterator it;
   double nov = 0;
   int i = 0;
   if (type == 0) {
      for (it = distances_0_.begin(); it != distances_0_.end() && i <= kNN;
           ++it, i++)
         nov += *it;
      return nov / i;
   } else if (type == 1) {
      for (it = distances_1_.begin(); it != distances_1_.end() && i <= kNN;
           ++it, i++)
         nov += *it;
      return nov / i;
   } else if (type == 2) {
      for (it = distances_2_.begin(); it != distances_2_.end() && i <= kNN;
           ++it, i++)
         nov += *it;
      return nov / i;
   } else
      return -1;
}

/******************************************************************************/
void team::updateComplexityRecord(map<long, team*>& teamMap, int rtcIndex) {
   (void)teamMap;
   // set <team *, teamIdComp> teams;
   // set <RegisterMachine *, RegisterMachineIdComp> RegisterMachines;
   // set <MemoryEigen *, MemoryEigenIdComp> memories;
   // GetAllNodes(teamMap, teams, RegisterMachines, memories, false);//not just
   // active RegisterMachines _numActiveTeams = teams.size(); _numActivePrograms
   // = RegisterMachines.size(); _numEffectiveInstructions = 0;
   // _numActiveFeatures = 0; for (auto leiter = RegisterMachines.begin();
   // leiter != RegisterMachines.end(); leiter++){
   //    _numEffectiveInstructions += (*leiter)->SizeEffective();
   //    _numActiveFeatures += (*leiter)->numFeatures();
   // }
   runTimeComplexityIns_ = GetMeanOutcome(_TRAIN_PHASE, 0, rtcIndex);
   runTimeComplexityTms_ = GetMeanOutcome(_TRAIN_PHASE, 0, rtcIndex - 1);
}

/******************************************************************************/
void team::updateComplexityRecord(map<long, team*>& teamMap, int rtcIndex,
                                  int auxInt, long auxIntMatch, int phase) {
   (void)teamMap;
   // set <team *, teamIdComp> teams;
   // set <RegisterMachine *, RegisterMachineIdComp> RegisterMachines;
   // set <MemoryEigen *, MemoryEigenIdComp> memories;
   // GetAllNodes(teamMap, teams, RegisterMachines, memories, false);//not just
   // active RegisterMachines _numActiveTeams = teams.size(); _numActivePrograms
   // = RegisterMachines.size(); _numEffectiveInstructions = 0;
   // _numActiveFeatures = 0; for (auto leiter = RegisterMachines.begin();
   // leiter != RegisterMachines.end(); leiter++){
   //    _numEffectiveInstructions += (*leiter)->SizeEffective();
   //    _numActiveFeatures += (*leiter)->numFeatures();
   // }
   runTimeComplexityIns_ =
       GetMeanOutcome(phase, 0, rtcIndex, auxInt, auxIntMatch);
   runTimeComplexityTms_ =
       GetMeanOutcome(phase, 0, rtcIndex - 1, auxInt, auxIntMatch);
}

// TODO(skelly): remove shared memory code
/******************************************************************************/
// void team::GetAllMemories(
//     map<long, team *> &teamMap, set<team *, teamIdComp> &visitedTeams,
//     set<MemoryEigen *, MemoryEigenIdComp> &memories) const {
//   visitedTeams.insert(teamMap[id_]);

//   for (auto prog : members_) {
//     for (int mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
//       memories.insert(prog->MemGet(mem_t));
//     }
//     if (prog->action_ >= 0 &&
//         find(visitedTeams.begin(), visitedTeams.end(),
//              teamMap[prog->action_]) == visitedTeams.end())
//       teamMap[prog->action_]->GetAllMemories(teamMap, visitedTeams,
//       memories);
//   }
// }
/******************************************************************************/
// this version returns partial graph up to team tm
void team::GetAllNodes(map<long, team*>& teamMap,
                       set<team*, teamIdComp>& visitedTeams, long stopId,
                       bool skipRoot) const {
   if (!skipRoot || !root_)
      visitedTeams.insert(teamMap[id_]);
   for (auto prog : members_) {
      if (prog->action_ >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[prog->action_]) == visitedTeams.end() &&
          prog->action_ != stopId)
         teamMap[prog->action_]->GetAllNodes(teamMap, visitedTeams, stopId,
                                             skipRoot);
   }
}

/******************************************************************************/
void team::GetAllNodes(
    map<long, team*>& teamMap, set<team*, teamIdComp>& visitedTeams,
    set<RegisterMachine*, RegisterMachineIdComp>& RegisterMachines) const
{
    // Print debug message indicating we've entered this function

    visitedTeams.insert(teamMap[id_]);
    for (auto prog : members_) {
        RegisterMachines.insert(prog);
        // Print debug message about which RegisterMachine we're adding

        if ((prog)->action_ >= 0 &&
            find(visitedTeams.begin(), visitedTeams.end(),
                 teamMap[(prog)->action_]) == visitedTeams.end())
        {
            // Print debug message about the recursive call
            teamMap[(prog)->action_]->GetAllNodes(teamMap, visitedTeams,
                                                  RegisterMachines);
        }
    }
}

/******************************************************************************/
void team::GetAllNodes(
    map<long, team*>& teamMap, set<team*, teamIdComp>& visitedTeams,
    set<RegisterMachine*, RegisterMachineIdComp>& RegisterMachines,
    set<MemoryEigen*>& memories) const {
   visitedTeams.insert(teamMap[id_]);
   for (auto prog : members_) {
      RegisterMachines.insert(prog);
      for (auto m : prog->private_memory_) {
         memories.insert(m);
      }
      if (prog->action_ >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[prog->action_]) == visitedTeams.end())
         teamMap[prog->action_]->GetAllNodes(teamMap, visitedTeams,
                                             RegisterMachines, memories);
   }
}

/******************************************************************************/
void team::updatePolicyRoot(map<long, team*>& teamMap,
                            set<team*, teamIdComp>& visitedTeams,
                            long& rootId) {
   visitedTeams.insert(teamMap[id_]);
   addPolicyRootId(rootId);  // add even if this is the root

   for (auto prog : members_) {
      if (prog->action_ >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[prog->action_]) == visitedTeams.end())
         teamMap[prog->action_]->updatePolicyRoot(teamMap, visitedTeams,
                                                  rootId);
   }
}

/******************************************************************************/
// Fill F with every feature indexed by every RegisterMachine in this policy
// (tree).If we ever build massive policy tress, this should be changed to a
// more efficient traversal. For now just look at every node.
int team::policyFeatures(map<long, team*>& teamMap,
                         set<team*, teamIdComp>& visitedTeams, set<long>& F,
                         bool active) const {
   visitedTeams.insert(teamMap[id_]);

   set<long> featuresSingle;
   int numProgramsInPolicy = 0;
   for (auto prog : members_) {
      numProgramsInPolicy++;
      featuresSingle = prog->features_;
      F.insert(featuresSingle.begin(), featuresSingle.end());
      if (prog->action_ >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[prog->action_]) == visitedTeams.end())
         numProgramsInPolicy += teamMap[prog->action_]->policyFeatures(
             teamMap, visitedTeams, F, active);
   }
   return numProgramsInPolicy;
}

/******************************************************************************/
void team::policyInstructions(
    map<long, team*>& teamMap, set<team*, teamIdComp>& visitedTeams,
    vector<int>& RegisterMachineInstructionCounts,
    vector<int>& effectiveProgramInstructionCounts) const {
   visitedTeams.insert(teamMap[id_]);

   for (auto prog : members_) {
      RegisterMachineInstructionCounts.push_back(
          static_cast<int>(prog->instructions_.size()));
      effectiveProgramInstructionCounts.push_back(
          static_cast<int>(prog->instructions_effective_.size()));

      if (prog->action_ >= 0 &&
          find(visitedTeams.begin(), visitedTeams.end(),
               teamMap[prog->action_]) == visitedTeams.end())
         teamMap[prog->action_]->policyInstructions(
             teamMap, visitedTeams, RegisterMachineInstructionCounts,
             effectiveProgramInstructionCounts);
   }
}

///****************************************************************************/
// void team::getBehaviourSequence(vector<int>&s, int phase) {
//    vector < behaviourType > singleEpisodeBehaviour;
//    map < point *, double >::reverse_iterator rit;
//    for (rit=outcomes_.rbegin(); rit!=outcomes_.rend(); rit++){
//       if ((rit->first)->phase() == phase){
//          (rit->first)->getBehaviour(singleEpisodeBehaviour);
//          if (s.size() + singleEpisodeBehaviour.size() <=
//          MAX_NCD_PROFILE_SIZE)
//             s.insert(s.end(),singleEpisodeBehaviour.begin(),singleEpisodeBehaviour.end());
//             //will cast to int (only discrete values used)
//          else
//             break;
//       }
//    }
// }

/******************************************************************************/
double team::GetMeanOutcome(int phase, int task, int auxDouble) {
   vector<double> outcomes;
   for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
        ouiter1++) {  // task
      if (ouiter1->first != task)
         continue;
      for (auto ouiter2 = ouiter1->second.begin();
           ouiter2 != ouiter1->second.end(); ouiter2++) {  // phase
         if (ouiter2->first != phase)
            continue;
         for (auto ouiter3 = ouiter2->second.begin();
              ouiter3 != ouiter2->second.end(); ouiter3++)  // points
            outcomes.push_back(ouiter3->second->auxDouble(auxDouble));
      }
   }

   if (outcomes.size() == 0)
      die(__FILE__, __FUNCTION__, __LINE__,
          "trying to get meanOutcome with no outcomes");
   return accumulate(outcomes.begin(), outcomes.end(), 0.0) / outcomes.size();
}

/******************************************************************************/
double team::GetMeanOutcome(int phase, int task, int auxDouble, int auxInt,
                            long auxIntMatch) {
   vector<double> outcomes;

   for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
        ouiter1++) {  // task
      if (ouiter1->first != task)
         continue;
      for (auto ouiter2 = ouiter1->second.begin();
           ouiter2 != ouiter1->second.end(); ouiter2++) {  // phase
         if (ouiter2->first != phase)
            continue;
         for (auto ouiter3 = ouiter2->second.begin();
              ouiter3 != ouiter2->second.end(); ouiter3++)  // points
            if (ouiter3->second->auxInt(auxInt) == auxIntMatch)
               outcomes.push_back(ouiter3->second->auxDouble(auxDouble));
      }
   }

   if (outcomes.size() == 0)
      die(__FILE__, __FUNCTION__, __LINE__,
          "trying to get meanOutcome with no outcomes");
   return accumulate(outcomes.begin(), outcomes.end(), 0.0) /
          outcomes
              .size();  //+(int)(topPortion*outcomes.size()),0.0)/(int)(topPortion*outcomes.size());
}

/******************************************************************************/
// TODO(skelly): fix outcomes_ data structure
double team::GetMedianOutcome(int phase, int task, int auxDouble) {
   vector<double> outcomes;
   for (auto o1 : outcomes_) {  //task
      if (o1.first != task)
         continue;
      for (auto o2 : o1.second) {  //phase
         if (o2.first != phase)
            continue;
         for (auto o3 : o2.second) {  //points
            outcomes.push_back(o3.second->auxDouble(auxDouble));
         }
      }
   }
   if (outcomes.size() == 0) {
      die(__FILE__, __FUNCTION__, __LINE__, "no outcomes");
   }
   return VectorMedian<double>(outcomes);
}

///****************************************************************************/
// double team::GetMeanOutcome(int phase, int task, int auxDouble, double
// &rValue1, double &rValue2, double minVal) {
//    vector < double > outcomes;
//    rValue2 = 0;
//    for(auto ouiter = outcomes_[task][phase].begin(); ouiter !=
//    outcomes_[task][phase].end(); ouiter++){
//       if (((phase == -1 || (ouiter->first)->phase() == phase) &&
//                (task == -1 || (ouiter->first)->task() == task) &&
//                (ouiter->first)->auxDouble(auxDouble) > minVal)){
//          outcomes.push_back((ouiter->first)->auxDouble(auxDouble));
//          if((ouiter->first)->auxDouble(auxDouble) >= 50)
//             rValue2++;
//       }
//
//    }
//    if (outcomes.size() == 0)
//       return 0.0;//die(__FILE__, __FUNCTION__, __LINE__, "trying to get
//       meanOutcome with no outcomes");
//    rValue1 = accumulate(outcomes.begin(),outcomes.end(), 0.0) /
//    outcomes.size();
//    //+(int)(topPortion*outcomes.size()),0.0)/(int)(topPortion*outcomes.size());
//    return rValue2 + rValue1/3000;
// }

/******************************************************************************/
bool team::hasPointDesc(string d, int task, int phase) {
   for (auto ouiter = outcomes_[task][phase].begin();
        ouiter != outcomes_[task][phase].end(); ouiter++)
      if (d.compare((ouiter->second)->desc()) == 0)
         return true;
   return false;
}

/******************************************************************************/
double team::getPointDescScore(string d, int task, int phase, int auxDouble) {
   for (auto ouiter = outcomes_[task][phase].begin();
        ouiter != outcomes_[task][phase].end(); ouiter++)
      if (d.compare((ouiter->second)->desc()) == 0)
         return (ouiter->second)->auxDouble(auxDouble);
   die(__FILE__, __FUNCTION__, __LINE__,
       "trying to get score from point that doesn't exist");
   return 0;
}

///****************************************************************************/
// bool team::getOutcome(point *pt, double *out) {
//    map < point *, double, pointLexicalLessThan > :: iterator ouiter;
//
//    if((ouiter = outcomes_.find(pt)) == outcomes_.end())
//       return false;
//
//    *out = ouiter->second;
//
//    return true;
// }

///****************************************************************************/
// double team::getRMSOutcome(int phase, int auxDouble) {
//    double rms = 0;
//    for(auto ouiter = outcomes_.begin(); ouiter != outcomes_.end();
//    ouiter++)
//       if ((ouiter->first)->phase() == phase)
//          rms += (ouiter->first)->auxDouble(auxDouble);
//    return isfinite(rms) ? -(sqrt(rms / outcomes_.size())) :
//    -numeric_limits<double>::max();
// }

///****************************************************************************/
// bool team::hasOutcome(point *pt) {
//    map < point *, double, pointLexicalLessThan > :: iterator ouiter;
//
//    if((ouiter = outcomes_.find(pt)) == outcomes_.end())
//       return false;
//
//    return true;
// }

///****************************************************************************/
///* Calculate normalized compression distance w.r.t another team. */
// double team::ncdBehaviouralDistance(team * t, int phase) {
//    ostringstream oss;
//    vector <int> theirBehaviourSequence;
//    t->getBehaviourSequence(theirBehaviourSequence, phase);
//    vector <int> myBehaviourSequence;
//    getBehaviourSequence(myBehaviourSequence, phase);
//    if (myBehaviourSequence.size() == 0 || theirBehaviourSequence.size()
//    == 0)
//       return -1;
//    return
//    normalizedCompressionDistance(myBehaviourSequence,theirBehaviourSequence);
// }

/******************************************************************************/
int team::numOutcomes(int phase, int task) {
   int numOut = 0;
   // for(auto ouiter = outcomes_.begin(); ouiter != outcomes_.end();
   // ouiter++)
   //    if (ouiter->first->phase() == phase && (task == -1 ||
   //    ouiter->first->task() == task))
   //       numOut++;
   if (task < 0)
      for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
           ouiter1++)
         numOut += ouiter1->second[phase].size();
   else
      numOut = outcomes_[task][phase].size();
   return numOut;
}

std::tuple<int, double, double> team::WrapVectorBidProbability(
    std::vector<RegisterMachine*>& programs,
    std::mt19937& rng,
    const bool homeostatic_root,
    const bool post_connection,
    const bool probabilistic_bid,
    const std::string& homeostatic_TD)
{
    std::vector<double> bid_values;
    std::vector<double> softmax_bid_values;

    bid_values.reserve(programs.size());
    softmax_bid_values.reserve(programs.size());

    std::tuple<int, double, double> winning_program_info;
    std::tuple<long, double, double, double> curr_program_info;

    for (const auto* prog : programs){
         bid_values.push_back(prog->bid_val_);
      }

   auto softmax = [](std::vector<double>& values)
    {
        const double epsilon = 1e-3;
        double factor = 1.0 - epsilon * values.size();
        if (factor < 0.0) factor = 0.0;
        double max_val = *std::max_element(values.begin(), values.end());
        double sum_exps = 0.0;

        for (auto& val : values)
        {
            val = std::exp(val - max_val);
            sum_exps += val;
        }

        double sum_adjusted = 0.0;
        for (auto& val : values)
        {
            val = (val / sum_exps) * factor + epsilon; 
            sum_adjusted += val;
        }

        for (auto& val : values)
            val /= sum_adjusted;
    };

    auto softmax_probs = [](std::vector<double>& values){
        const double epsilon = 1e-3;
        const double T = 0.8;                 // higher temp = more exploration
        const double invT = (T > 0.0) ? 1.0 / T : 1e6;

        double factor = 1.0 - epsilon * values.size();
        if (factor < 0.0) factor = 0.0;

        double max_scaled = -std::numeric_limits<double>::infinity();
        for (const auto& v : values)
            max_scaled = std::max(max_scaled, v * invT);

        double sum_exps = 0.0;
        for (auto& v : values)
        {
            v = std::exp(v * invT - max_scaled);
            sum_exps += v;
        }

        double sum_adjusted = 0.0;
        for (auto& v : values)
        {
            v = (v / sum_exps) * factor + epsilon;
            sum_adjusted += v;
        }

        for (auto& v : values)
            v /= sum_adjusted;
    };

    // Routing-based modulatory signal: higher when the router is confident (low entropy)
   //  auto routing_mod_signal = [](const std::vector<double>& probs) -> double {
   //      if (probs.empty()) return 0.0;
   //      double H = 0.0;
   //      for (double p : probs) {
   //          if (p > 0.0) H -= p * std::log(p);
   //      }
   //      double Hmax = std::log(static_cast<double>(probs.size()));
   //      if (Hmax <= 0.0) return 0.0;
   //      double entropy_norm = H / Hmax;      // in [0,1]
   //      double concentration = 1.0 - entropy_norm;  // 0 = uniform, 1 = peaked
   //      // (Optional) keep away from exact 0 to avoid stalling tiny updates
   //      const double floor = 1e-6;
   //      if (concentration < floor) concentration = floor;
   //      return concentration; // our modulatory term m_t
   //  };

    softmax(bid_values);
    softmax_bid_values = bid_values;
   //  double mod_signal = routing_mod_signal(softmax_bid_values);
   double winning_lr_to_keep = 0.0;
   for (size_t i = 0; i < programs.size(); ++i){
      if (HebbianMap.getEligWeight(HebbianMap.getPrevProgID(), programs[i]->id_) < 1.0){
         winning_lr_to_keep = 0.0;
      }
      else{
         winning_lr_to_keep = programs[i]->learning_rate_;
      }
   }

   if (HebbianMap.getPrevProgID() == -1 && homeostatic_root){ // root team
    for (size_t i = 0; i < programs.size(); ++i){
        const auto* prog = programs[i];
      if (winning_lr_to_keep == 0.0){
         curr_program_info = std::make_tuple(prog->id_, bid_values[i], prog->learning_rate_, prog->initial_heb_noise_);}
      else{
       curr_program_info = std::make_tuple(prog->id_, bid_values[i], winning_lr_to_keep, prog->initial_heb_noise_);}
        HebbianMap.setCurrent(curr_program_info);
        double weight = HebbianMap.getWeight(HebbianMap.getPrevProgID(),prog->id_, bid_values[i]);
        bid_values[i] = weight;
        HebbianMap.decayElig(HebbianMap.getPrevProgID(), prog->id_); //decay eligibility of each connection
        if (homeostatic_TD == "none"){
            HebbianMap.setWeightRoot();
            HebbianMap.divisiveRenormRow(/*Sg=*/3.0);
        }
    }
    HebbianMap.root_step++;
   }
   else if (HebbianMap.getPrevProgID() > -1 && post_connection){ // post connection
        for (size_t i = 0; i < programs.size(); ++i){   
            const auto* prog = programs[i];
            if (winning_lr_to_keep == 0.0){
            curr_program_info = std::make_tuple(prog->id_, bid_values[i], prog->learning_rate_, prog->initial_heb_noise_);}
            else{
            curr_program_info = std::make_tuple(prog->id_, bid_values[i], winning_lr_to_keep, prog->initial_heb_noise_);}
            HebbianMap.setCurrent(curr_program_info);
            double weight = HebbianMap.getWeight(HebbianMap.getPrevProgID(), prog->id_, bid_values[i]);
            bid_values[i] = weight;
            HebbianMap.decayElig(HebbianMap.getPrevProgID(), prog->id_); //decay eligibility of each connection
            if (homeostatic_TD == "none"){
            HebbianMap.setWeight();
            HebbianMap.divisiveRenormRow(/*Sg=*/3.0);
            } 
        }
    }
   

    int index_winner_ = 0;
    if (probabilistic_bid) {
        std::vector<double> probs(programs.size());
        probs = bid_values;
        softmax_probs(probs);
      //   softmax_probs(probs);
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        index_winner_ = dist(rng);
    }
    else {
        index_winner_ = std::distance(
            bid_values.begin(),
            std::max_element(bid_values.begin(), bid_values.end())
        );
    }

   // winner updates
   // **********************
   curr_program_info = std::make_tuple(
            programs[index_winner_]->id_,
            softmax_bid_values[index_winner_],
            programs[index_winner_]->learning_rate_,
            programs[index_winner_]->initial_heb_noise_
        );
        HebbianMap.setCurrent(curr_program_info);
        HebbianMap.boostElig(HebbianMap.getPrevProgID(), programs[index_winner_]->id_, softmax_bid_values[index_winner_]);
        // only modulate reward on teams 
      //   if (programs[index_winner_]->action_ < 0){
      //       HebbianMap.boostElig(HebbianMap.getPrevProgID(), programs[index_winner_]->id_, softmax_bid_values[index_winner_]);
      //   }
      //   HebbianMap.pred_error = mod_signal;
        if (HebbianMap.getPrevProgID() == -1 && homeostatic_TD == "step"){
            HebbianMap.setWeightRootTD();
        } else if (HebbianMap.getPrevProgID() > -1 && homeostatic_TD == "step"){
            HebbianMap.setWeightTD();}
   // **********************

    HebbianMap.divisiveRenormRow(/*Sg=*/3.0);

    winning_program_info = std::make_tuple(
        index_winner_,
        softmax_bid_values[index_winner_],
        programs[index_winner_]->learning_rate_
    );
    
    bid_values.clear();
    softmax_bid_values.clear();
    // Debug: print each program's ID and its softmax bid
    //   std::cout.setf(std::ios::fixed);
    //   std::cout.precision(6);
    //   cout << "------------------------------" <<endl;
    //   for (size_t i = 0; i < programs.size(); ++i) {
    //       std::cout << "prog_id " << programs[i]->id_
    //                 << " softmax_bid " << softmax_bid_values[i]
    //                 << " action " << programs[i]->action_
    //                 << std::endl;
    //   }
    //   cout << "------------------------------" <<endl;
    return winning_program_info;
}


/******************************************************************************/
// Assumes the program is in the team
// Does not maintain team size > 0
// Does not maintain n_atomic_ > 0
void team::RemoveProgram(RegisterMachine* prog) {
   if (prog->action_ < 0)
      n_atomic_--;
   prog->nrefs_--;
   auto it = find(members_.begin(), members_.end(), prog);
   members_.erase(it);
   members_run_.resize(members_.size());  // put in mark introns
}

// Return true if a RegisterMachine was removed, otherwise return false
bool team::RemoveRandomProgram(mt19937& rng) {
   if (members_.size() < 2)
      return false;  // Maintain team size > 0
   uniform_int_distribution<int> dis_RegisterMachines(0, members_.size() - 1);
   auto it = members_.begin();
   advance(it, dis_RegisterMachines(rng));
   // Don't remove the only atomic
   if (!((*it)->action_ < 0 && n_atomic_ < 2)) {
      if ((*it)->action_ < 0)
         n_atomic_--;
      (*it)->nrefs_--;
      members_.erase(it);
      return true;
   }
   return false;
}


/******************************************************************************/
void team::resetOutcomes(int phase) {
   // map < point *, double, pointLexicalLessThan > :: iterator ouiter;
   // for (ouiter = outcomes_.begin(); ouiter != outcomes_.end();)
   //{
   //    if ((ouiter->first)->phase() == phase || phase < 0){
   //       delete ouiter->first;
   //       outcomes_.erase(ouiter++);
   //    }
   //    else
   //       ouiter++;
   // }
   //

   for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
        ouiter1++)  // task
      for (auto ouiter2 = ouiter1->second.begin();
           ouiter2 != ouiter1->second.end(); ouiter2++)  // phase
         if (ouiter2->first == phase || phase == -1) {
            for (auto ouiter3 = ouiter2->second.begin();
                 ouiter3 != ouiter2->second.end();) {
               delete ouiter3->second;
               ouiter2->second.erase(ouiter3++);
            }
         }
   quickSums_.clear();
   quickMeans_.clear();
   runTimeComplexityIns_ = 0;
   runTimeComplexityTms_ = 0;
}

/******************************************************************************/
void team::swapOutcomePhase(int phaseFrom, int phaseTo, int auxInt,
                            long auxIntMatch) {
   resetOutcomes(phaseTo);
   resetOutcomes(_TEST_PHASE);
   for (auto ouiter1 = outcomes_.begin(); ouiter1 != outcomes_.end();
        ouiter1++) {  // task
      for (auto ouiter2 = outcomes_[ouiter1->first][phaseFrom].begin();
           ouiter2 != outcomes_[ouiter1->first][phaseFrom].end();
           ouiter2++)  // points
         if ((ouiter2->second)->auxInt(auxInt) == auxIntMatch) {
            ouiter2->second->phase(phaseTo);
            setOutcome(ouiter2->second);
         }
      outcomes_[ouiter1->first][phaseFrom].clear();
   }
}

/******************************************************************************/
void team::setOutcome(point* pt) {
   // if((outcomes_[pt->task()][pt->phase()].insert(map <
   // pt->auxInt(POINT_AUX_INT_ENVSEED), point
   // *>::value_type(pt->auxInt(POINT_AUX_INT_ENVSEED),pt))).second ==
   // false) die(__FILE__, __FUNCTION__, __LINE__, "could not set outcome,
   // duplicate point?");
   outcomes_[pt->task()][pt->phase()][pt->auxInt(POINT_AUX_INT_ENVSEED)] = pt;
   if (hasQuickSum(pt->task(), pt->key(), pt->phase()))
      quickSums_[pt->task()][pt->key()][pt->phase()] +=
          pt->auxDouble(pt->key());
   else
      quickSums_[pt->task()][pt->key()][pt->phase()] = pt->auxDouble(pt->key());

   quickMeans_[pt->task()][pt->key()][pt->phase()] =
       quickSums_[pt->task()][pt->key()][pt->phase()] /
       numOutcomes(pt->phase(), pt->task());
}

void team::GetAction(EvalData& eval_data, std::mt19937& rng, std::unordered_map<std::string, std::any>& params, std::tuple<long, double, double>& prev_prog_history) {
   eval_data.team_path.push_back(eval_data.team_map[id_]);
   long teamIdToFollow = 0;

   members_run_.clear();
   members_run_.reserve(members_.size());
   for (auto* prog : members_) {
      if (std::any_cast<int>(params["dynamic_lgp"]) == 0){
           prog->Run(eval_data, eval_data.timestep, eval_data.team_path.size(), eval_data.verbose);}
      else{prog->RunCoordination(eval_data, eval_data.timestep, eval_data.team_path.size(), eval_data.verbose);}

   members_run_.push_back(prog);
   eval_data.instruction_count += static_cast<int>(prog->instructions_effective_.size());
   }

   const bool hebb_learn = std::any_cast<int>(params["hebb_learning"]) == 1;
   const bool homeostatic_root = std::any_cast<int>(params["homeostatic_root"]) == 1;
   const bool post_connection = std::any_cast<int>(params["post_connection"]) == 1;
   const bool probabilistic_bid = std::any_cast<int>(params["probabilistic_bid"]) == 1;
   //bool eligibility_trace = std::any_cast<int>(params["eligibility_trace"]) == 1;
   const std::string& homeostatic_TD = std::any_cast<std::string>(params["homeostatic_TD"]);
   

   if(hebb_learn){
     
      HebbianMap.lambda_td = eval_data.tm->lambda_td_;
      HebbianMap.pred_error = eval_data.pred_error;
      HebbianMap.gamma = std::any_cast<double>(params["decay_factor"]);

      HebbianMap.additive_root = (std::any_cast<int>(params["additive_root"]) == 1);
      HebbianMap.additive_sub_team = (std::any_cast<int>(params["additive_sub_team"]) == 1);
 
      if(HebbianMap.gamma == 0.0){
      HebbianMap.gamma = eval_data.tm->decay_factor_;}

      std::tuple<int, double, double> curr_program_info;
      HebbianMap.setPrevious(prev_prog_history);
      curr_program_info = WrapVectorBidProbability(members_run_,rng, homeostatic_root, post_connection, probabilistic_bid, homeostatic_TD); //what returns from here is the index of the winner NOT ID.
      long id = members_run_[std::get<0>(curr_program_info)]->id_;
      // cout << "winner program: " << members_run_[std::get<0>(curr_program_info)]->id_  << " team: " << eval_data.team_path.back()->id_ << endl; // todo(ALI): HEATMAP PROGRAM

      if (members_run_[std::get<0>(curr_program_info)]->action_ < 0) {  // has an atomic action
         if (std::any_cast<int>(params["debug_map"]) == 1)
               HebbianMap.printWeights();
         HebbianMap.clearCurrent();             //clear since this is atomic TODO(ALI) double check logic
         prev_prog_history = std::make_tuple(-1, 1, 1); //id, bid, learning_rate
         
         eval_data.program_out = members_run_[std::get<0>(curr_program_info)];
         // cout << "prog: " << members_run_[std::get<0>(curr_program_info)]->id_ << " bid: " << members_run_[std::get<0>(curr_program_info)]->bid_val_ << endl;
         // members_run_[std::get<0>(curr_program_info)]->PrintInstructions(false);
         return;
      } else { //leads to a new team
         teamIdToFollow = members_run_[std::get<0>(curr_program_info)]->action_;
         prev_prog_history = std::make_tuple(members_run_[std::get<0>(curr_program_info)]->id_,
            std::get<1>(curr_program_info), 
            std::get<2>(curr_program_info));
           //  cout << "prog: " << members_run_[std::get<0>(curr_program_info)]->id_ << " bid: " << members_run_[std::get<0>(curr_program_info)]->bid_val_ << endl;
         // members_run_[std::get<0>(curr_program_info)]->PrintInstructions(false);
      }
   }
   else{
      sort(members_run_.begin(), members_run_.end(), RegisterMachineBidLexicalCompare());
      // if (eval_data.team_path.size() == 1) {
      //    std::cout << "step " << std::endl;
      //    for (size_t i = 0; i < members_run_.size(); i++){
      //       std::cout << "Program " << i << ": Weight = 0 " << " Bid = " << members_run_[i]->bid_val_ << "\n";
      //    }
      // } // todo(ALI): HEATMAP PROGRAM
      for (size_t i = 0; i < members_run_.size(); i++) {
        // cout << "winner program: " << members_run_[i]->id_  << " team: " << eval_data.team_path.back()->id_ << endl; // todo(ALI): HEATMAP PROGRAM
         if (members_run_[i]->action_ < 0) {
            eval_data.program_out = members_run_[i];
            // eval_data.program_out->PrintInstructions(false);
            return;
         } else if (find(eval_data.team_path.begin(), eval_data.team_path.end(),
                         eval_data.team_map[members_run_[i]->action_]) ==
                    eval_data.team_path.end()) {
                     // members_run_[i]->PrintInstructions(false);
            teamIdToFollow = members_run_[i]->action_;
            break;
         }
      }
   }
   return eval_data.team_map[teamIdToFollow]->GetAction(eval_data, rng, params, prev_prog_history);
}
