#include "TPG.h"
#include "core/event_dispatcher.h"
#include "metrics/selection/selection_metrics.h"
#include "metrics/selection/selection_metrics_builder.h"
#include "metrics/replacement/replacement_metrics.h"
#include "metrics/replacement/replacement_metrics_builder.h"
#include "metrics/removal/removal_metrics_builder.h"
#include "metrics/removal/removal_metrics.h"

#include "EvalData.h"
#include <algorithm>
#include <queue>   // for breadth‑first traversal of team pointers
#include <filesystem>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdlib>   // for std::system

/******************************************************************************/
TPG::TPG() {
   instruction::SetupOps();
   real_dist_ = uniform_real_distribution<>(0.0, 1.0);
   _Memids.resize(MemoryEigen::kNumMemoryType_);
   _Memory.resize(MemoryEigen::kNumMemoryType_);
   for (size_t i = 0; i < _NUM_PHASE; i++)
      _numEliteTeamsCurrent.push_back(0);
   _ops.resize(instruction::NUM_OP);
   fill(_ops.begin(), _ops.end(), false);
   rngs_.resize(NUM_RNG);
   seeds_.resize(NUM_RNG);
}

/******************************************************************************/
TPG::~TPG() {}

/******************************************************************************/
void TPG::AddProgram(RegisterMachine* p) {
   program_pop_[p->id_] = p;
}

/******************************************************************************/
void TPG::AddTeam(team* tm) {
   team_pop_.insert(tm);
   team_map_[tm->id_] = tm;
}

/******************************************************************************/
void TPG::RemoveTeam(team* tm) {
   // decrement program refs
   for (auto prog : tm->members_) {
      prog->nrefs_--;
   }
   // TODO(skelly): test cloning
   if (team_map_.find(tm->cloneId_) != team_map_.end())
      team_map_[tm->cloneId_]->clones_--;
   team_map_.erase(tm->id_);
   team_pop_.erase(tm);
   delete tm;
}

/******************************************************************************/
void TPG::AddMemory(long prog_id, MemoryEigen* m) {
   _Memory[m->type_][prog_id] = m;
}

/******************************************************************************/
team* TPG::getTeamByID(long id) {
   for (auto teiter = team_pop_.begin(); teiter != team_pop_.end(); teiter++)
      if ((*teiter)->id_ == id)
         return *teiter;
   return *(team_pop_.begin());
}

/******************************************************************************/
bool TPG::haveEliteTeam(string taskset, int fitMode, int phase) {
   return _eliteTeamPS[taskset][fitMode].find(phase) !=
          _eliteTeamPS[taskset][fitMode].end();
}

/******************************************************************************/
void TPG::Seed(size_t i, uint_fast32_t s) {
   seeds_[i] = s;
   rngs_[i].seed(seeds_[i]);
}

void TPG::InitExperimentTracking(APIClient* apiClient) {
   api_client_ = apiClient;
}

/******************************************************************************/
void TPG::clearMemory() {
   for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
      for (auto meiter = _Memory[mem_t].begin(); meiter != _Memory[mem_t].end();
           meiter++) {
         meiter->second->ClearWorking();
         meiter->second->ClearReadTime();
         meiter->second->ClearWriteTime();
      }
   }
}

void TPG::GetAction(EvalData& eval_data, std::mt19937& rng, std::unordered_map<std::string, std::any>& params, std::tuple<long, double, double>& prev_prog_history){
  eval_data.instruction_count = 0;
  eval_data.team_path.clear();
  eval_data.tm->GetAction(eval_data, rng, params, prev_prog_history);
}



void TPG::ComputeIndReward(EvalData& eval_data,
                           std::unordered_map<std::string, std::any>& params,
                           double reward_per_step)
{
    auto& H = eval_data.tm->HebbianMap;
    double& V   = eval_data.running_mean;  
    // const double gamma = H.gamma;    
    double alpha = 0.02;            

    // double delta = reward_per_step - (1.0 - gamma) * V;
    double delta = reward_per_step - V; // https://dl.acm.org/doi/pdf/10.5555/2074022.2074088

   //  Debug print before update
   //  std::cerr << "[ComputeIndReward] reward=" << reward_per_step 
   //            << " V(before)=" << V 
   //            << " gamma=" << gamma 
   //            << " delta=" << delta << std::endl;

    V += alpha * delta;               // critic update

   //  // Debug print after update
   //  std::cerr << "[ComputeIndReward] V(after)=" << V << std::endl;

    H.pred_error = delta;             // feed to actor/plasticity (which uses λ)
    eval_data.pred_error = delta;
    
}


/******************************************************************************/
void TPG::GetAllNodes(team* tm, set<team*, teamIdComp>& teams,
                      set<RegisterMachine*, RegisterMachineIdComp>& programs) {
   teams.clear();
   programs.clear();
   tm->GetAllNodes(team_map_, teams, programs);
}

/******************************************************************************/
map<long, team*> TPG::GetRootTeamsInMap() const {
   map<long, team*> teams;
   for (auto tm : team_pop_) {
      if (tm->root_) {
         teams[tm->id_] = tm;
      }
   }
   return teams;
}

/******************************************************************************/
set<team*, teamIdComp> TPG::GetRootTeamsInSet() {
   set<team*, teamIdComp> teams;
   for (auto tm : team_pop_) {
      if (tm->root_) {
         teams.insert(tm);
      }
   }
   return teams;
}

/******************************************************************************/
vector<team*> TPG::GetRootTeamsInVec() const {
   vector<team*> teams;
   for (auto tm : team_pop_) {
      if (tm->root_) {
         teams.push_back(tm);
      }
   }
   return teams;
}

/******************************************************************************/
bool TPG::isElitePS(team* tm, int phase) {
   for (auto itr1 = _eliteTeamPS.begin(); itr1 != _eliteTeamPS.end();
        itr1++) {  // taskset
      for (auto itr2 = itr1->second.begin(); itr2 != itr1->second.end();
           itr2++) {  // fitmode
                      // for (auto itr3 = itr2->second.begin(); itr3 !=
                      // itr2->second.end(); itr3++)//phase
         if (itr2->second.find(phase) != itr2->second.end() &&
             itr2->second[phase]->id_ == tm->id_)
            return true;
      }
   }
   return false;
}

/******************************************************************************/
void TPG::MarkEffectiveCode() {
   for (auto prog : program_pop_) {
      prog.second->stateful_ = GetParam<int>("stateful");
      prog.second->MarkIntrons(params_, prog.second->n_memories_);
   }
}

/******************************************************************************/
void TPG::printOss() {
   cout << oss.str();
   oss.str("");
}

/******************************************************************************/
void TPG::printOss(ostringstream& o) {
   oss << o.str();
   o.str("");
}

/******************************************************************************/
void TPG::ReadParameters(std::string file_name, std::unordered_map<std::string, std::any>& params) {
   YAML::Node config = YAML::LoadFile(file_name);

   // Load parameters
   for (const auto& entry : config) {
       string category = entry.first.as<string>();
       if (entry.second.IsMap()) {
           for (const auto& param : entry.second) {
               string key = param.first.as<string>(); // Flatten key

               if (param.second.IsScalar()) {
                  string value = param.second.as<string>();
                  string tag = param.second.Tag();

                  if (tag == "!") {  // Check if value was quoted as a string
                    params[key] = value;
                 } else if (value.find('.') != string::npos)
                     params[key] = stringToDouble(value);  // Store as double if it has a decimal
                  else {
                     params[key] = stringToInt(value);  // Otherwise, store as int
                  }

                  if (category == "operations") { // change _ops value if operations
                     int opCode = instruction::GetOpCodeFromName(key);
                     if (opCode != -1) {
                        _ops[opCode] = param.second.as<int>() ? true : false;
                     }
                  }
               }
           }
       }
   }
}

/******************************************************************************/
void TPG::resetOutcomes(int phase, bool roots) {
   for (auto tm : team_pop_) {
      if (!roots || (roots && tm->root_)) {
         tm->resetOutcomes(phase);
      }
   }
}

/******************************************************************************/
void TPG::setOutcome(team* tm, string behav, vector<double>& rewards,
                     vector<int>& ints, long gtime) {
   point* p = new point(gtime, state_["point_count"]++, behav, rewards, ints);
   p->key(GetParam<int>("fit_mode"));
   tm->setOutcome(p);
}

/******************************************************************************/
void TPG::InitMapElitesArchive() {
    map_elites_archive_.assign(GetParam<int>("n_root"), nullptr);
}

/******************************************************************************/
void TPG::UpdateMapElitesArchive()
{
    if (map_elites_archive_.empty() ||
        map_elites_archive_.size() != static_cast<size_t>(GetParam<int>("n_root")))
    {
        InitMapElitesArchive();
    }
    const size_t bins = map_elites_archive_.size();
    const size_t grid_cols = static_cast<size_t>(std::ceil(std::sqrt(bins)));
    const size_t grid_rows = (bins + grid_cols - 1) / grid_cols;
    const std::string param_mode = GetParam<std::string>("map_elites_param");

    std::vector<long>   best_ids(bins, -1);
    std::vector<double> best_fits(bins, -std::numeric_limits<double>::infinity());
    std::vector<double> best_lambda(bins, 0.0);
    std::vector<double> best_decay(bins, 0.0);
    std::vector<team*>  reps(bins, nullptr);
    std::vector<double> best_plast(bins, 0.0);

    std::filesystem::create_directories("logs/map_elites");
    const std::string filename =
        std::string("logs/map_elites/map_elites_archive_") + std::to_string(seeds_[TPG_SEED]) + ".csv";

    if (std::filesystem::exists(filename)){
        std::ifstream ifs(filename);
        std::string line;
        while (std::getline(ifs, line))
        {
            std::stringstream ss(line);
            std::string token;

            size_t bin_idx = 0;
            long   id      = -1;
            double fit     = -std::numeric_limits<double>::infinity();
            double lambda_val = 0.0;
            double decay_val  = 0.0;
            double plast_val  = 0.0;

            ss >> token;           // "bin"
            ss >> bin_idx;         // bin number
            bin_idx--;             // 0‑index
            while (ss >> token) {
                if (token == "id:" || token == "id")
                    ss >> id;
                else if (token == "fit:" || token == "fit")
                    ss >> fit;
                else if (token == "λ:" || token == "λ")
                    ss >> lambda_val;
                else if (token == "decay:" || token == "decay")
                    ss >> decay_val;
                else if (token == "plast:" || token == "plast")
                    ss >> plast_val;
            }

            if (bin_idx < bins) {
                best_ids[bin_idx]    = id;
                best_fits[bin_idx]   = fit;
                best_lambda[bin_idx] = lambda_val;
                best_decay[bin_idx]  = decay_val;
                best_plast[bin_idx] = plast_val;

                if (team_map_.count(id)) {
                  reps[bin_idx] = team_map_.at(id);
                    if (reps[bin_idx] && !reps[bin_idx]->elite(GetState("phase"))) {
                        reps[bin_idx]->elite(GetState("phase"), true);
                    }
                }
            }
        }
        ifs.close();
    }

    for (auto tm : GetRootTeamsInVec())
    {
        if (param_mode == "plasticity")
        {
            double plast = clamp(tm->team_plasticity_, 0.0, 1.0);
            size_t idx   = std::min(static_cast<size_t>(std::floor(plast * bins)),
                                    bins - 1);

            if (tm->fit_ > best_fits[idx]) {
                best_ids[idx]   = tm->id_;
                best_fits[idx]  = tm->fit_;
                best_plast[idx] = plast;
                reps[idx]       = tm;
                best_lambda[idx] = tm->lambda_td_;
                best_decay[idx]  = tm->decay_factor_;
                if (tm && !tm->elite(GetState("phase")))
                    tm->elite(GetState("phase"), true);
            }
            continue;   // next team
        }
        
        double lam = tm->lambda_td_;
        double dec = tm->decay_factor_;

        size_t lam_idx = std::min(static_cast<size_t>(std::floor(lam * grid_cols)), grid_cols - 1);
        size_t dec_idx = std::min(static_cast<size_t>(std::floor(dec * grid_rows)), grid_rows - 1);
        size_t idx     = lam_idx * grid_cols + dec_idx; 

        if (idx < bins && tm->fit_ > best_fits[idx]) {
            best_ids[idx]    = tm->id_;
            best_fits[idx]   = tm->fit_;
            best_lambda[idx] = lam;
            best_decay[idx]  = dec;
            best_plast[idx]   = clamp(tm->team_plasticity_, 0.0, 1.0);
            reps[idx]  = tm;
            if (tm && !tm->elite(GetState("phase")))
                tm->elite(GetState("phase"), true);
        }
    }

    std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
    if (ofs)
    {
        for (size_t idx = 0; idx < bins; ++idx)
        {
            ofs << "bin " << (idx + 1) << ": ";
            if (best_ids[idx] != -1) {
                ofs << "id: "   << best_ids[idx]
                    << " fit: "  << best_fits[idx]
                    << " λ: "    << best_lambda[idx]
                    << " decay: " << best_decay[idx]
                    << " plast: " << best_plast[idx];
            }
            ofs << '\n';
        }
        ofs.close();
    }
    map_elites_archive_ = std::move(reps);
}

/******************************************************************************/
void TPG::SaveMapElitesArchive() {
    std::filesystem::create_directories("logs");

    bool save_gen = GetParam<int>("save_me_generationally") == 1;
   
   std::string filename = std::string("logs/map_elites/map_elites_archive_");
   if (save_gen) {
      filename = filename + std::to_string(seeds_[TPG_SEED]) +
         "_t" + std::to_string(GetState("t_current")) + ".csv";
   } else {
      filename = filename + std::to_string(seeds_[TPG_SEED]) + ".csv";
   }
    
    const size_t bins = map_elites_archive_.size();
    const std::string param_mode = GetParam<std::string>("map_elites_param");
    std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
    if (!ofs) {
        return;
    }

    for (size_t idx = 0; idx < bins; ++idx) {
        ofs << "bin " << (idx + 1) << ": ";
        if (map_elites_archive_[idx] != nullptr) {
            team* rep = map_elites_archive_[idx];
            ofs << "id: "   << rep->id_
                << " fit: "  << rep->fit_
                << " λ: "    << rep->lambda_td_
                << " decay: " << rep->decay_factor_
                << " plast: " << clamp(rep->team_plasticity_,0.0, 1.0);
        }
        ofs << '\n';
    }
    ofs.close();
}

/******************************************************************************/
const std::vector<team*>& TPG::GetMapElitesArchive() const {
    return map_elites_archive_;
}

/******************************************************************************/
void TPG::finalize() {
   // TODO(skelly): remove clears that are not required
   _allComponentsA.clear();
   _allComponentsAt.clear();
   _eliteTeams.clear();
   _eliteTeamPS.clear();
   team_map_.clear();
   _Memids.clear();  //TODO(skelly): remove
   _Memids.resize(MemoryEigen::kNumMemoryType_);
   state_["memory_count"] = 0;
   _numEliteTeamsCurrent.clear();
   for (size_t i = 0; i < _NUM_PHASE; i++)
      _numEliteTeamsCurrent.push_back(0);
   _persistenceFilterA.clear();
   _persistenceFilterA_t.clear();
   _persistenceFilterAllTime.clear();
   state_["point_count"] = 0;
   state_["program_count"] = 0;
   state_["memory_count"] = 0;
   task_set_map_.clear();

   for (auto tm : team_pop_) {
      tm->resetOutcomes(-1);
      delete tm;
   }
   team_pop_.clear();

   for (auto prog : program_pop_) {
      delete prog.second;
   }
   program_pop_.clear();

   for (auto memory_map : _Memory) {
      for (auto m : memory_map) {
         delete m.second;
      }
   }

   _Memory.clear();
   _Memory.resize(MemoryEigen::kNumMemoryType_);

   phylo_graph_.clear();
   map_elites_archive_.clear();
}

/******************************************************************************/
void TPG::TeamMutator_ProgramOrder(team* team_to_mu) {
   if ((team_to_mu)->size() > 1 &&
       real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pmw")) {
      int i, j;
      uniform_int_distribution<int> dis_team_size(0, (team_to_mu)->size() - 1);
      i = dis_team_size(rngs_[TPG_SEED]);
      do {
         j = dis_team_size(rngs_[TPG_SEED]);
      } while (i == j);
      (team_to_mu)->MuProgramOrder(i, j);
   }
}

/******************************************************************************/
// void TPG::TeamMutator_LambdaVariation(team* team_to_mu) {
//    if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pml")) {
//         const double scaling_factor = 2.0;

//         std::uniform_real_distribution<double> exponent_dist(-1.0, 1.0);

//         double mutation_factor = std::pow(scaling_factor, exponent_dist(rngs_[TPG_SEED]));

//         team_to_mu->lambda_td_ *= mutation_factor;
//         team_to_mu->lambda_td_ = std::clamp(team_to_mu->lambda_td_, 0.0, 1.0);
//     }
// }

void TPG::TeamMutator_LambdaVariation(team* team_to_mu) {
   if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pml")){

   std::normal_distribution<double> gauss_lambda(0.0, 0.5);
   double factor = std::exp(gauss_lambda(rngs_[TPG_SEED])); 
   team_to_mu->lambda_td_ = std::clamp(team_to_mu->lambda_td_ * factor, 0.0, 1.0);}
   }

/******************************************************************************/
void TPG::TeamMutator_DecayVariation(team* team_to_mu) {
   if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pmdf")){
   std::normal_distribution<double> gauss_lambda(0.0, 0.5);
   double factor = std::exp(gauss_lambda(rngs_[TPG_SEED])); 
   team_to_mu->decay_factor_ = std::clamp(team_to_mu->decay_factor_ * factor, 0.0, 1.0);
   }
}

// void TPG::TeamMutator_DecayVariation(team* team_to_mu) {
//    if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pmdf")) {
//         const double scaling_factor = 2.0;

//         std::uniform_real_distribution<double> exponent_dist(-1.0, 1.0);

//         double mutation_factor = std::pow(scaling_factor, exponent_dist(rngs_[TPG_SEED]));

//         team_to_mu->decay_factor_ *= mutation_factor;
//         team_to_mu->decay_factor_ = std::clamp(team_to_mu->decay_factor_, 0.0, 1.0);
//     }
// }


/******************************************************************************/
void TPG::TeamMutator_AddPrograms(team* team_to_mu) {
   double rd = real_dist_(rngs_[TPG_SEED]);
   if ((int)team_to_mu->size() < GetParam<int>("max_team_size") &&
       rd < GetParam<double>("pma")) {
      uniform_int_distribution<int> dis_programs(0, program_pop_.size() - 1);
      uniform_int_distribution<int> dis_team_size(0, team_to_mu->size() - 1);
      int random_prog_index = dis_programs(rngs_[TPG_SEED]);
      auto it = program_pop_.begin();
      std::advance(it, random_prog_index);
      RegisterMachine* p = it->second;
      team_to_mu->AddProgram(p, dis_team_size(rngs_[TPG_SEED]));
   }
}

/******************************************************************************/
void TPG::TeamMutator_RemovePrograms(team* team_to_mu) {
   if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pmd"))
      team_to_mu->RemoveRandomProgram(rngs_[TPG_SEED]);
}


/******************************************************************************/
team* TPG::CloneTeam(team* team_to_clone) {
   team* team_clone = new team(GetState("t_current"), state_["team_count"]++, team_to_clone->obs_index_, team_to_clone->lambda_td_, team_to_clone->decay_factor_);
   for (auto m : team_to_clone->members_) {
      team_clone->AddProgram(m);
   }
   return team_clone;
}

/******************************************************************************/
RegisterMachine* TPG::CloneProgram(RegisterMachine* prog) {
   RegisterMachine* prog_clone = new RegisterMachine(
       *(dynamic_cast<RegisterMachine*>(prog)), params_, state_, rngs_[TPG_SEED]);
   if (prog_clone->action_ >= 0)
      team_map_[prog_clone->action_]->AddIncomingProgram(prog_clone->id_);
   return prog_clone;
}

/******************************************************************************/
void TPG::ProgramMutator_Instructions(RegisterMachine* prog_to_mu) {
   const int mutation_passes =
       HaveParam("n_mutation_passes") ? GetParam<int>("n_mutation_passes") : 1;
   if (mutation_passes < 1) {
      die(__FILE__, __FUNCTION__, __LINE__,
          "n_mutation_passes must be at least one.");
   }
   // Normally mutation consumes the parent's final self-modified S2-S6
   // outputs. This ablation instead makes mutation consume the inherited
   // constants by resetting working memory before the first mutation pass.
   if (HaveParam("reset_self_modifying_before_mutation") &&
       GetParam<int>("reset_self_modifying_before_mutation") != 0) {
      prog_to_mu->SeedSelfModifyingWorkingFromConstants(params_);
   }
   for (int pass = 0; pass < mutation_passes; ++pass) {
      prog_to_mu->Mutate(params_, state_, rngs_[TPG_SEED], _ops);
   }
   // All stacked passes use the selected starting rates (parent outputs by
   // default, inherited constants in the reset-before ablation). Once
   // variation is complete, prepare the child to start evaluation from its
   // inherited constants, which may have evolved if that ablation is enabled.
   prog_to_mu->SeedSelfModifyingWorkingFromConstants(params_);
}

/******************************************************************************/
void TPG::MutateActionToTerminal(RegisterMachine* prog_to_mu, team* new_team) {
   // If program is already terminal (action < 0) and there are no discrete
   // actions there is nothing to change
   if (prog_to_mu->action_ < 0 && GetParam<int>("n_discrete_action") == 0) {
      return;
   } else {
      long new_discrete_action;
      if (GetParam<int>("n_discrete_action") == 0) {
         new_discrete_action = -1;  // Only one terminal option
      } else {
         uniform_int_distribution<int> dis(
             0, GetParam<int>("n_discrete_action") - 1);
         do {
            new_discrete_action = -1 - dis(rngs_[TPG_SEED]);
         } while (prog_to_mu->action_ == new_discrete_action);
      }
      // If changing from team pointer to atomic, update pointee's incoming
      if (prog_to_mu->action_ >= 0) {
         new_team->n_atomic_++;
         team_map_[prog_to_mu->action_]->removeIncomingProgram(prog_to_mu->id_);
      }
      prog_to_mu->action_ = new_discrete_action;
   }
}

/******************************************************************************/
void TPG::MutateActionToTeam(RegisterMachine* prog_to_mu, team* new_team,
                             int& n_new_teams) {
   // All programs remain terminal in the first generation
   if (GetState("t_current") == 1) {
      return;
   } else {
      if (prog_to_mu->action_ < 0) {
        new_team->n_atomic_--;
      }
      uniform_int_distribution<int> disM(0, team_map_.size() - 1);
      team* tm;
      int tries = 0;
      do {
         auto it = team_map_.begin();
         advance(it, disM(rngs_[TPG_SEED]));
         tm = it->second;
      } while (tries++ < 20 &&
               (tm->gtime_ == GetState("t_current") || tm->clones_ > 0 ||
                prog_to_mu->action_ == tm->id_));
      if (prog_to_mu->action_ >= 0)
         team_map_[prog_to_mu->action_]->removeIncomingProgram(prog_to_mu->id_);
      if (!tm->root()) {  // Already subsumed, don't clone
         prog_to_mu->action_ = tm->id_;
         tm->AddIncomingProgram(prog_to_mu->id_);
      } else {  // clone when subsumed
         team* sub = new team(GetState("t_current"), state_["team_count"]++, tm->obs_index_, tm->lambda_td_, tm->decay_factor_);
         tm->clone(phylo_graph_, &sub);
         prog_to_mu->action_ = sub->id_;
         sub->AddIncomingProgram(prog_to_mu->id_);
         // TODO(skelly): put in PhyloGraph functions
         phylo_graph_[tm->id_].adj.push_back(sub->id_);
         phylo_graph_.insert(pair<long, phyloRecord>(sub->id_, phyloRecord()));
         phylo_graph_[sub->id_].gtime = GetState("t_current");
         phylo_graph_[sub->id_].root = false;
         AddTeam(sub);
         n_new_teams++;
      }
   }
}

/******************************************************************************/
void TPG::ProgramMutator_ActionPointer(RegisterMachine* prog_to_mu,
                                       team* new_team, int& n_new_teams) {
   if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pmn")) {
      if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("p_atomic")) {
         MutateActionToTerminal(prog_to_mu, new_team);
      } else if (new_team->n_atomic_ > 1 || prog_to_mu->action_ >= 0){
         MutateActionToTeam(prog_to_mu, new_team, n_new_teams);
      }
   }
}

/******************************************************************************/
void TPG::AddAncestorToPhylogeny(team* parent, team* new_team) {
   phylo_graph_[new_team->id_].ancestorIds.insert(parent->id_);
   new_team->addAncestorId(parent->id_);
   phylo_graph_[parent->id_].adj.push_back(new_team->id_);
}

/******************************************************************************/
void TPG::AddTeamToPhylogeny(team* new_team) {
   phylo_graph_.insert(pair<long, phyloRecord>(new_team->id_, phyloRecord()));
   phylo_graph_[new_team->id_].gtime = GetState("t_current");
   phylo_graph_[new_team->id_].root = new_team->root_;
}

// void TPG::InsertTeamLambdaToCSV(team* new_team) {
//    phylo_graph_[new_team->id_].gtime = GetState("t_current");
//    phylo_graph_[new_team->id_].root = new_team->root_;
// }


/******************************************************************************/
team* TPG::TeamCrossover(team* parent1, team* parent2) {
   double lambda_to_use = 0.0;
   double decay_to_use = 0.0;
   if (real_dist_(rngs_[TPG_SEED]) > 0.5){
      lambda_to_use = parent1->lambda_td_;}
   else{
      lambda_to_use = parent2->lambda_td_;}
   
   if (real_dist_(rngs_[TPG_SEED]) > 0.5){
      decay_to_use = parent1->decay_factor_;}
   else{
      decay_to_use = parent2->decay_factor_;}

   team* child_team = new team(GetState("t_current"), state_["team_count"]++, ((parent1->obs_index_ + parent2->obs_index_)/2), lambda_to_use, decay_to_use);
   
   // TODO(skelly): linear crossover
   if (parent1->size() == 1 && parent2->size() == 1 &&
       parent1->members_.front()->instructions_.size() > 1 &&
       parent2->members_.front()->instructions_.size() > 1 && (real_dist_(rngs_[TPG_SEED]) <  GetParam<double>("p_crossover"))) {
      RegisterMachine* child_1;
      RegisterMachine* child_2;
      //RegisterMachineCrossover(parent1->members_.front(), parent2->members_.front(), &child_1, &child_2);
      
      LinearCrossover(parent1->members_.front(), parent2->members_.front(), &child_1, &child_2);

      RegisterMachine* child_prog = (real_dist_(rngs_[TPG_SEED]) < 0.5) ? child_1 : child_2;
      RegisterMachine* unused_prog = (child_prog == child_1) ? child_2 : child_1;

      AddProgram(child_prog);

      child_team->AddProgram(child_prog);
      delete unused_prog;
   } else {
      std::list<RegisterMachine*> p1_progs = parent1->members_;
      auto p1_it = p1_progs.begin();
      std::list<RegisterMachine*> p2_progs = parent2->members_;
      auto p2_it = p2_progs.begin();
      // TODO(skelly): intertwine crossover
      while (p1_it != p1_progs.end() || p2_it != p2_progs.end()) {
         if (p1_it != p1_progs.end()) {
            if ((*p1_it)->action_ < 0 && child_team->n_atomic_ < 1) {
               child_team->AddProgram(*p1_it);
            } else if ((int)child_team->size() <
                           GetParam<int>("max_team_size") &&
                       real_dist_(rngs_[TPG_SEED]) <
                           GetParam<double>("pmx_p")) {
               child_team->AddProgram(*p1_it);
            }
         }
         if (p2_it != p2_progs.end()) {
            if ((*p2_it)->action_ < 0 && child_team->n_atomic_ < 1) {
               child_team->AddProgram(*p2_it);
            } else if ((int)child_team->size() <
                           GetParam<int>("max_team_size") &&
                       real_dist_(rngs_[TPG_SEED]) <
                           GetParam<double>("pmx_p")) {
               child_team->AddProgram(*p2_it);
            }
         }
         if (p1_it != p1_progs.end())
            p1_it++;
         if (p2_it != p2_progs.end())
            p2_it++;
      }
   }
   return child_team;
}


/******************************************************************************/
// NSGA-II with configurable objectives:
//   (1) fitness (mandatory, maximize)
//   (2+) configurable objectives from pareto_objectives param (all minimize):
//        - flops: weighted instruction cost (FLOPs) across all effective instructions
//        - total_instructions: sum of instructions_.size() across all programs
//        - effective_instructions: sum of instructions_effective_.size() across all programs
//        - total_registers: sum of n_memories_ across all programs
//        - effective_registers: sum of n_effective_registers_ across all programs
// Writes into: team->pareto_rank_ (lower is better), team->crowding_distance_ (higher is better)
void TPG::ComputeParetoFronts(std::vector<team*>& teams) {
  if (teams.empty()) return;
  const size_t n = teams.size();

  std::string objectives_str = "";
  if (HaveParam("pareto_objectives")) {
    objectives_str = GetParam<std::string>("pareto_objectives");
  }
  
  bool use_flops = (objectives_str.find("flops") != std::string::npos);
  bool use_total_instructions = (objectives_str.find("total_instructions") != std::string::npos);
  bool use_effective_instructions = (objectives_str.find("effective_instructions") != std::string::npos);
  bool use_total_registers = (objectives_str.find("total_registers") != std::string::npos);
  bool use_effective_registers = (objectives_str.find("effective_registers") != std::string::npos);

  size_t num_objectives = 1; // fitness is always included
  if (use_flops) num_objectives++;
  if (use_total_instructions) num_objectives++;
  if (use_effective_instructions) num_objectives++;
  if (use_total_registers) num_objectives++;
  if (use_effective_registers) num_objectives++;
  std::vector<std::vector<double>> objectives(n, std::vector<double>(num_objectives, 0.0));

  for (size_t i = 0; i < n; ++i) {
    const team* t = teams[i];
    if (!t) continue;

    size_t obj_idx = 0;
    objectives[i][obj_idx++] = t->fit_;

    double flops_sum = 0.0;
    double total_instr = 0.0;
    double effective_instr = 0.0;
    double total_regs = 0.0;
    double effective_regs = 0.0;

    for (auto* prog : t->members_) {
      if (!prog) continue;
      if (use_flops) {
        double mat_size = prog->private_memory_[MemoryEigen::kMatrixType_]->memory_size_;
        for (auto* instr : prog->instructions_effective_) {
          flops_sum += instruction::GetOperationFLOPs(instr->op_, mat_size);
        }
      }
      total_instr += static_cast<double>(prog->instructions_.size());
      effective_instr += static_cast<double>(prog->instructions_effective_.size());
      total_regs += static_cast<double>(prog->n_memories_);
      effective_regs += static_cast<double>(prog->n_effective_registers_);
    }

    if (use_flops) objectives[i][obj_idx++] = flops_sum;
    if (use_total_instructions) objectives[i][obj_idx++] = total_instr;
    if (use_effective_instructions) objectives[i][obj_idx++] = effective_instr;
    if (use_total_registers) objectives[i][obj_idx++] = total_regs;
    if (use_effective_registers) objectives[i][obj_idx++] = effective_regs;
  }

  // Dominance: A dominates B if A is no worse in all objectives and strictly better in at least one.
  // Objective 0 (fitness): higher is better
  // All other objectives: lower is better
  auto dominates = [&](size_t a, size_t b) -> bool {
    bool no_worse = true;
    bool strictly_better = false;

    for (size_t obj = 0; obj < num_objectives; ++obj) {
      if (obj == 0) {
        // Fitness: maximize (higher is better)
        if (objectives[a][obj] < objectives[b][obj]) no_worse = false;
        if (objectives[a][obj] > objectives[b][obj]) strictly_better = true;
      } else {
        // Other objectives: minimize (lower is better)
        if (objectives[a][obj] > objectives[b][obj]) no_worse = false;
        if (objectives[a][obj] < objectives[b][obj]) strictly_better = true;
      }
    }

    return no_worse && strictly_better;
  };

  // Fast pairwise fill: only compare (i,j) once
  std::vector<int> domination_count(n, 0);
  std::vector<std::vector<size_t>> dominates_list(n);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (dominates(i, j)) {
        dominates_list[i].push_back(j);
        domination_count[j]++;
      } else if (dominates(j, i)) {
        dominates_list[j].push_back(i);
        domination_count[i]++;
      }
    }
  }

  // Build Pareto fronts
  std::vector<std::vector<size_t>> fronts;
  fronts.emplace_back();
  fronts.back().reserve(n);

  for (size_t i = 0; i < n; ++i) {
    if (domination_count[i] == 0) {
      teams[i]->pareto_rank_ = 0;
      fronts[0].push_back(i);
    }
  }

  int front_rank = 0;
  while (front_rank < (int)fronts.size() && !fronts[front_rank].empty()) {
    std::vector<size_t> next_front;
    for (size_t p : fronts[front_rank]) {
      for (size_t q : dominates_list[p]) {
        if (--domination_count[q] == 0) {
          teams[q]->pareto_rank_ = front_rank + 1;
          next_front.push_back(q);
        }
      }
    }
    if (!next_front.empty()) fronts.push_back(std::move(next_front));
    front_rank++;
  }

  // Crowding distance (sum of normalized gaps across all objectives)
  const double INF = std::numeric_limits<double>::max();
  auto safe_range = [](double r) { return (r < 1e-12 ? 1.0 : r); };

  for (const auto& front : fronts) {
    const size_t fsize = front.size();
    if (fsize == 0) continue;

    for (size_t idx : front) teams[idx]->crowding_distance_ = 0.0;

    if (fsize <= 2) {
      for (size_t idx : front) teams[idx]->crowding_distance_ = INF;
      continue;
    }

    std::vector<size_t> sorted = front;

    // Compute crowding distance for each objective
    for (size_t obj = 0; obj < num_objectives; ++obj) {
      // Sort by this objective (ascending)
      std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
        return objectives[a][obj] < objectives[b][obj];
      });

      // Boundary points get infinite crowding distance
      teams[sorted.front()]->crowding_distance_ = INF;
      teams[sorted.back()]->crowding_distance_ = INF;

      double obj_range = safe_range(objectives[sorted.back()][obj] - objectives[sorted.front()][obj]);

      // Interior points: add normalized gap contribution
      for (size_t k = 1; k + 1 < fsize; ++k) {
        if (teams[sorted[k]]->crowding_distance_ >= INF) continue;
        teams[sorted[k]]->crowding_distance_ += 
            (objectives[sorted[k + 1]][obj] - objectives[sorted[k - 1]][obj]) / obj_range;
      }
    }
  }
}

/******************************************************************************/
team* TPG::TeamSelector_Tournament(vector<team*>& candidate_parent_teams) {
   if (candidate_parent_teams.empty()) return nullptr;

   uniform_int_distribution<int> dis(
       0, static_cast<int>(candidate_parent_teams.size()) - 1);
   const int tsize = std::max(1, GetParam<int>("tournament_size"));

   // NSGA-II style comparison: 
   // 1. Lower Pareto rank is better
   // 2. If same rank, higher crowding distance is better (promotes diversity)
   auto better = [&](const team* a, const team* b) -> bool {
      if (!b) return true;
      if (!a) return false;

      // Compare by Pareto front rank first (lower is better)
      if (a->pareto_rank_ < b->pareto_rank_) return true;
      if (a->pareto_rank_ > b->pareto_rank_) return false;

      // Same front: prefer higher crowding distance (more diverse/spread out)
      if (a->crowding_distance_ > b->crowding_distance_) return true;
      if (a->crowding_distance_ < b->crowding_distance_) return false;

      // Tie-breaker: prefer younger (more recent innovation)
      return a->id_ > b->id_;
   };

   /* COMMENTED OUT: Old lexicographic parsimony selection
   auto mean_n_memories = [](const team* t) -> double {
      if (!t || t->members_.empty()) return 0.0;
      double sum = 0.0;
      for (auto* p : t->members_) {
         if (p) sum += static_cast<double>(p->n_memories_);
      }
      return sum / static_cast<double>(t->members_.size());
   };

   auto total_program_instructions_in_policy = [](const team* t) -> double {
      if (!t || t->members_.empty()) return 0.0;
      double sum = 0.0;
      for (auto* prog : t->members_) {
         if (prog) sum += static_cast<double>(prog->instructions_.size());
      }
      return sum;
   };

   auto complexity_score = [&](const team* t) -> double {
      const double I = total_program_instructions_in_policy(t);
      const double M = mean_n_memories(t);
      return M;
   };

   auto better_lexicographic = [&](const team* a, const team* b) -> bool {
      if (!b) return true;
      if (!a) return false;

      const double fa = a->fit_;
      const double fb = b->fit_;

      const double scale = std::max({1.0, std::abs(fa), std::abs(fb)});
      const double eps_fit = 1e-3 * scale;

      if (fa > fb + eps_fit) return true;
      if (fb > fa + eps_fit) return false;

      const double ca = complexity_score(a);
      const double cb = complexity_score(b);
      const double eps_c = 1e-12 * (1.0 + std::abs(ca) + std::abs(cb));

      if (ca < cb - eps_c) return true;
      if (cb < ca - eps_c) return false;

      const double Ia = total_program_instructions_in_policy(a);
      const double Ib = total_program_instructions_in_policy(b);
      const double eps_I = 1e-12 * (1.0 + std::abs(Ia) + std::abs(Ib));

      if (Ia < Ib - eps_I) return true;
      if (Ib < Ia - eps_I) return false;

      const double Ma = mean_n_memories(a);
      const double Mb = mean_n_memories(b);
      const double eps_M = 1e-12 * (1.0 + std::abs(Ma) + std::abs(Mb));

      if (Ma < Mb - eps_M) return true;
      if (Mb < Ma - eps_M) return false;

      return a->id_ < b->id_;
   };
   END COMMENTED OUT */

   team* best = candidate_parent_teams[dis(rngs_[TPG_SEED])];
   for (int k = 1; k < tsize; ++k) {
      team* cand = candidate_parent_teams[dis(rngs_[TPG_SEED])];
      if (better(cand, best)) best = cand;
   }
   return best;
}

/******************************************************************************/
void TPG::GenerateNewTeams() {
   int new_teams_count = 0;
   const bool shadow_run = GetParam<int>("shadow_run") != 0;
   auto task_power_set = PowerSet(GetState("n_task"));
   int n_new_teams_per_set =
       GetParam<int>("n_root_gen") / task_power_set.size();
   vector<team*> candidate_parent_teams;
   if (!GetParam<int>("parent_select_roots_only")) {
      candidate_parent_teams.resize(team_pop_.size());
      std::copy(team_pop_.begin(), team_pop_.end(),
                candidate_parent_teams.begin());
      // Pareto ranks are only needed for fitness-based tournament selection.
      if (!shadow_run) ComputeParetoFronts(candidate_parent_teams);
   }
   for (auto& subset : task_power_set) {
      //TODO(skelly): put selection in a separate function
      if (GetParam<int>("parent_select_roots_only")) {
         if (task_set_map_[VectorToStringNoSpace(subset)].size() == 0)
            continue;
         candidate_parent_teams = task_set_map_[VectorToStringNoSpace(subset)];
         // Pareto ranks are only needed for fitness-based tournament selection.
         if (!shadow_run) ComputeParetoFronts(candidate_parent_teams);
      }
      uniform_int_distribution<int> disP(0, candidate_parent_teams.size() - 1);
      for (int i = 0; i < n_new_teams_per_set; i++) {
         team* parent_team1;
         team* parent_team2;
         if (!shadow_run && GetParam<int>("tournament_size") > 0) {
            // Fitness-based tournament selection.
            parent_team1 = TeamSelector_Tournament(candidate_parent_teams);
            parent_team2 = TeamSelector_Tournament(candidate_parent_teams);
         } else {  // Random selection, including shadow runs.
            parent_team1 = candidate_parent_teams[disP(rngs_[TPG_SEED])];
            parent_team2 = candidate_parent_teams[disP(rngs_[TPG_SEED])];
         }
         team* child_team;
         // Maybe use crossover
         if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("pmx")) {
            child_team = TeamCrossover(parent_team1, parent_team2);
            AddAncestorToPhylogeny(parent_team1, child_team);
            AddAncestorToPhylogeny(parent_team2, child_team);
         } else {
            child_team = CloneTeam(parent_team1);
            AddAncestorToPhylogeny(parent_team1, child_team);
         }
         AddTeamToPhylogeny(child_team);
         // InsertTeamLambdaToCSV(child_team);
         // Mutate child team
         ApplyVariationOps(child_team, new_teams_count);
         AddTeam(child_team);
         new_teams_count++;
      }
   }
   oss << "genTms t " << GetState("t_current") << " Msz " << team_pop_.size()
       << " Lsz " << program_pop_.size();
   oss << _Memory.size() << " eLSz "
       << _numEliteTeamsCurrent[GetState("phase")];
   oss << " nNTms " << new_teams_count << endl;

   ReplacementMetricsBuilder builder;
   builder.with_generation(GetState("t_current"))
      .with_num_teams(team_pop_.size())
      .with_num_programs(program_pop_.size())
      .with_memory_size(_Memory.size())
      .with_num_elite_teams(_numEliteTeamsCurrent[GetState("phase")])
      .with_num_new_teams(new_teams_count);

   ReplacementMetrics metrics = builder.build();
   EventDispatcher<ReplacementMetrics>::instance().notify(EventType::REPLACEMENT, metrics);
}

/******************************************************************************/
void TPG::ApplyVariationOps(team* team_to_modify, int& n_new_teams) {
   uniform_int_distribution<int> disL(0, program_pop_.size() - 1);
   // Mutate team
   TeamMutator_LambdaVariation(team_to_modify);
   TeamMutator_DecayVariation(team_to_modify);
   TeamMutator_RemovePrograms(team_to_modify);
   TeamMutator_AddPrograms(team_to_modify);
   TeamMutator_ProgramOrder(team_to_modify);
   // Mutate programs
   // Clone before modifying ?
   if (real_dist_(rngs_[TPG_SEED]) < GetParam<double>("p_clone_program")) {
      set<RegisterMachine*, RegisterMachineIdComp> new_team_programs =
          team_to_modify->CopyMembers();
      for (auto prog : new_team_programs) {
         if (GetParam<int>("max_team_size") == 1 ||  //Always clone in this mode
             real_dist_(rngs_[TPG_SEED]) < 1.0 / new_team_programs.size()) {
            // TODO(skelly): add/remove changes order and thus behaviour?
            team_to_modify->RemoveProgram(prog);
            RegisterMachine* prog_clone = CloneProgram(prog);
            // ProgramMutator_Memory(prog_clone);
            ProgramMutator_Instructions(prog_clone);
            ProgramMutator_ActionPointer(prog_clone, team_to_modify,
                                         n_new_teams);
            team_to_modify->AddProgram(prog_clone);
            AddProgram(prog_clone);
         }
      }
   } else {  // Modify without cloning
      for (auto prog : team_to_modify->members_) {
         if (real_dist_(rngs_[TPG_SEED]) < 1.0 / team_to_modify->size()) {
            // ProgramMutator_Memory(prog);
            ProgramMutator_Instructions(prog);
         }
      }
   }
}

/******************************************************************************/ 
void TPG::ExtinctionEvent() {
    int keepPercentage = GetParam<int>("extinction_percentage");
    int totalTeams = team_pop_.size();
    const bool shadow_run = GetParam<int>("shadow_run") != 0;
    
    if (totalTeams <= 1)
        return;

    // Normal extinction preserves the best team.  Shadow runs sample every
    // survivor uniformly, including the best team, to avoid fitness selection.
    team* bestTeam = shadow_run ? nullptr : GetBestTeam();
    vector<team*> candidates;
    for (auto tm : GetRootTeamsInVec()) {
        if (!bestTeam || tm->id_ != bestTeam->id_) {
            candidates.push_back(tm);
        }
    }
    
    int candidateCount = candidates.size();
    int keepCandidates = std::max(0, static_cast<int>(std::ceil(candidateCount * (keepPercentage / 100.0))));
    std::shuffle(candidates.begin(), candidates.end(), rngs_[TPG_SEED]);

    unordered_set<team*> teamsToKeep;
    if (bestTeam) teamsToKeep.insert(bestTeam);
    for (int i = 0; i < keepCandidates && i < candidateCount; ++i) {
        teamsToKeep.insert(candidates[i]);
    }

    int n_deleted = 0;
    int n_old_deleted = 0;
    int n_root_remaining = 0;

    for (auto tm : GetRootTeamsInVec()) {
        if (teamsToKeep.find(tm) == teamsToKeep.end()) {
            phylo_graph_[tm->id_].dtime = GetState("t_current");
            n_deleted++;
            n_old_deleted += (tm->gtime_ < GetState("t_current")) ? 1 : 0;
            RemoveTeam(tm);
        }
        n_root_remaining++;
    }

    CleanupProgramsWithNoRefs();

    oss << "ExtinctionEvent: Randomly kept " << teamsToKeep.size() 
        << " teams out of " << totalTeams 
        << ", removed " << n_deleted 
        << " teams (old deleted: " << n_old_deleted << ")" << endl;

    RemovalMetricsBuilder builder;
    builder.with_generation(GetState("t_current"))
           .with_num_teams(team_pop_.size())
           .with_num_programs(program_pop_.size())
           .with_num_root_programs(n_root_remaining)
           .with_num_elite_teams(_numEliteTeamsCurrent[GetState("phase")])
           .with_num_deleted(n_deleted)
           .with_num_old_deleted(n_old_deleted)
           .with_percent_old_deleted(n_deleted > 0 ? (double)n_old_deleted / n_deleted : 0);

    RemovalMetrics metrics = builder.build();
    EventDispatcher<RemovalMetrics>::instance().notify(EventType::REMOVAL, metrics);
}


/******************************************************************************/
team* TPG::GetBestTeam() {
   set<team*, teamFitnessLexicalCompare> teams;
   std::copy_if(team_pop_.begin(), team_pop_.end(),
                std::inserter(teams, teams.end()),
                [](const team* tm) { return tm->root_; });
   return *teams.begin();
}

/******************************************************************************/
void TPG::UpdateTeamPhyloData(team* tm) {
   // MarkEffectiveCode(tm);
   if (tm->runTimeComplexityIns() == 0) {
      // TODO(skelly): clean out magic number - 2
      tm->updateComplexityRecord(team_map_,
                                 GetParam<int>("n_point_aux_double") - 2);
   }
   phylo_graph_[tm->id_].fitness = tm->fit_;
   phylo_graph_[tm->id_].numActiveFeatures = tm->numActiveFeatures_;
   phylo_graph_[tm->id_].numActivePrograms = tm->numActivePrograms_;
   phylo_graph_[tm->id_].numActiveTeams = tm->numActiveTeams_;
   phylo_graph_[tm->id_].numEffectiveInstructions =
       tm->numEffectiveInstructions();
}

/******************************************************************************/
// Find the elite single-task program graphs
void TPG::FindSingleTaskFitnessRange(vector<TaskEnv*>& tasks,
                                     vector<vector<double>>& mins,
                                     vector<vector<double>>& maxs) {
   vector<team*> teamsRankedVec;
   for (int task = 0; task < GetState("n_task"); task++) {
      teamsRankedVec.clear();
      for (auto tm : GetRootTeamsInVec()) {
         tm->elite(GetState("phase"), false);  // mark team as not elite
         if (tm->numOutcomes(GetState("phase"), task) >=
             tasks[task]->GetNumEval(GetState("phase"))) {
            tm->fit_ = tm->GetMeanOutcome(GetState("phase"), task,
                                            GetState("fitMode"));
            teamsRankedVec.push_back(tm);
            if (GetState("phase") == _TEST_PHASE) {
               UpdateTeamPhyloData(tm);
            }
         } else {
            // mark elite to protect until eval in all tasks
            tm->elite(true);
         }
      }
      if (teamsRankedVec.size() > 0) {
         sort(teamsRankedVec.begin(), teamsRankedVec.end(),
              teamFitnessLexicalCompare());
         maxs[GetState("fitMode")][task] = (*(teamsRankedVec.begin()))->fit_;
         mins[GetState("fitMode")][task] = (*(teamsRankedVec.rbegin()))->fit_;
      }
   }
}

void TPG::FindSingleTaskMinMax(vector<TaskEnv*>& tasks,
                                     vector<vector<double>>& mins,
                                     vector<vector<double>>& maxs) {
   vector<team*> teamsRankedVec;
   for (int task = 0; task < GetState("n_task"); task++) {
      teamsRankedVec.clear();
      for (auto tm : GetRootTeamsInVec()) {
         tm->elite(GetState("phase"), false);
         if (tm->numOutcomes(GetState("phase"), task) >=
             tasks[task]->GetNumEval(GetState("phase"))) {
            tm->fit_ = tm->GetMeanOutcome(GetState("phase"), task,
                                            GetState("fitMode"));
            teamsRankedVec.push_back(tm);
            if (GetState("phase") == _TEST_PHASE) {
               UpdateTeamPhyloData(tm);
            }
         }
      }
      if (teamsRankedVec.size() > 0) {
         sort(teamsRankedVec.begin(), teamsRankedVec.end(),
              teamFitnessLexicalCompare());
         maxs[GetState("fitMode")][task] = (*(teamsRankedVec.begin()))->fit_;
         mins[GetState("fitMode")][task] = (*(teamsRankedVec.rbegin()))->fit_;
      }
   }
}

/******************************************************************************/
vector<team*> TPG::NormalizeScoresAndRankTeams(
    vector<TaskEnv*>& tasks, vector<int>& set,
    vector<vector<double>>& min_scores, vector<vector<double>>& max_scores) {
   vector<team*> vec;
   for (auto tm : GetRootTeamsInVec()) {
      if (GetState("phase") == _TEST_PHASE &&
          tm->id_ != (_eliteTeamPS[VectorToStringNoSpace(set)][GetParam<int>(
                          "fit_mode")][_VALIDATION_PHASE])
                         ->id_) {
         continue;
      }
      vector<double> normalizedScores;
      for (size_t task = 0; task < set.size(); task++) {
         if (tm->numOutcomes(GetState("phase"), set[task]) <
             tasks[set[task]]->GetNumEval(GetState("phase"))) {
            die(__FILE__, __FUNCTION__, __LINE__,
                "All root teams should have enough evaluations at this "
                "point.");
         }
         auto raw_mean_score = tm->GetMeanOutcome(
             GetState("phase"), set[task], GetState("fitMode"));
         // guards for same min and max
         if (!isEqual(min_scores[GetState("fitMode")][set[task]],
                      max_scores[GetState("fitMode")][set[task]])) {
            normalizedScores.push_back(
                (raw_mean_score - min_scores[GetState("fitMode")][set[task]]) /
                (max_scores[GetState("fitMode")][set[task]] -
                 min_scores[GetState("fitMode")][set[task]]));
         } else {
            normalizedScores.push_back(
                raw_mean_score / max_scores[GetState("fitMode")][set[task]]);
         }
      }
      if (normalizedScores.size() != set.size()) {
         die(__FILE__, __FUNCTION__, __LINE__,
             "This team should have a score for all tasks.");
      }
      tm->fit_ = *min_element(normalizedScores.begin(), normalizedScores.end());
      // TODO(skelly): debug, test, and cleanup complexity record
      // GetParam<int>("n_point_aux_double") - 2 refers to
      // decision_instructions
      tm->updateComplexityRecord(team_map_,
                                 GetParam<int>("n_point_aux_double") - 1);
      vec.push_back(tm);
   }
   return vec;
}

/******************************************************************************/
void TPG::FindMultiTaskElites(vector<TaskEnv*>& tasks,
                              vector<vector<double>>& min_scores,
                              vector<vector<double>>& max_scores) {
   auto PS = PowerSet(GetState("n_task"));
   size_t n_elite_per_task = GetParam<int>("n_root") / PS.size();
   for (auto& set : PS) {
      if (GetState("phase") == _TRAIN_PHASE)
         task_set_map_[VectorToStringNoSpace(set)]
             .clear();  // TODO(skelly): check this
      auto teams_normed_scores =
          NormalizeScoresAndRankTeams(tasks, set, min_scores, max_scores);

      // A shadow run replaces fitness-based survivor selection with a
      // reproducible random ordering.  Use the evolution RNG (seed_tpg), not
      // the auxiliary/environment RNG, so the selected population is fully
      // determined by the experiment seed.
      const bool shadow_run =
          GetState("phase") == _TRAIN_PHASE && GetParam<int>("shadow_run") != 0;
      if (shadow_run) {
         std::shuffle(teams_normed_scores.begin(), teams_normed_scores.end(),
                      rngs_[TPG_SEED]);
      } else {
         sort(teams_normed_scores.begin(), teams_normed_scores.end(),
              teamFitnessLexicalCompare());
      }
      // sort(teams_normed_scores.begin(), teams_normed_scores.end(),
      // teamFitComplexLexCompare());
      size_t elite_count = 0;
      for (auto tm : teams_normed_scores) {
         if ((elite_count >= n_elite_per_task) || _numEliteTeamsCurrent[GetState("phase")] == static_cast<size_t>(GetParam<int>("n_root")))
            break;
         if (!tm->elite(GetState("phase"))) {
            elite_count++;
            _numEliteTeamsCurrent[GetState("phase")]++;
            tm->elite(GetState("phase"), true);
            if (GetState("phase") == _TRAIN_PHASE) {
               tm->fitnessBin(GetState("t_current"), VectorToStringNoSpace(set));
               phylo_graph_[tm->id_].fitnessBin = tm->fitnessBin();
               phylo_graph_[tm->id_].fitness = tm->fit_;

               phylo_graph_[tm->id_].taskFitnesses.clear();
               for (int task = 0; task < GetState("n_task"); task++) {
                  phylo_graph_[tm->id_].taskFitnesses.push_back(
                      tm->GetMeanOutcome(GetState("phase"), task,
                                           GetState("fitMode")));
               }
            }
            if (GetState("phase") == _TRAIN_PHASE)
               task_set_map_[VectorToStringNoSpace(set)].push_back(tm);
         }
      }
      // Always keep track of single elite team for each task set
      _eliteTeamPS[VectorToStringNoSpace(set)][GetState("fitMode")]
                  [GetState("phase")] = *(teams_normed_scores.begin());
   }
}

/******************************************************************************/
void TPG::SetEliteTeams(vector<TaskEnv*>& tasks) {
   vector<team*> teams_normed_scores;
   _numEliteTeamsCurrent[GetState("phase")] = 0;
   const bool shadow_run =
       GetState("phase") == _TRAIN_PHASE && GetParam<int>("shadow_run") != 0;
   vector<vector<double>> min_scores, max_scores;
   min_scores.resize(GetParam<int>("n_fit_mode"));
   max_scores.resize(GetParam<int>("n_fit_mode"));
   min_scores[GetState("fitMode")].resize(GetState("n_task"));
   max_scores[GetState("fitMode")].resize(GetState("n_task"));

   // MAP ELITES
   // MAP-Elites supplies archive survivors.  Shadow runs bypass the archive so
   // the actual training survivor selection remains random.
   if (GetParam<int>("map_elites") && !shadow_run) {
      FindSingleTaskMinMax(tasks, min_scores, max_scores);
      UpdateMapElitesArchive();

      for (auto rep : map_elites_archive_) {
         if (rep){
            rep->elite(GetState("phase"), true);
            _numEliteTeamsCurrent[GetState("phase")]++;
         }}
      SaveMapElitesArchive();
   }else{
      FindSingleTaskFitnessRange(tasks, min_scores, max_scores);
   }
   FindMultiTaskElites(tasks, min_scores, max_scores);

   auto PS = PowerSet(GetState("n_task"));
   for (auto& set : PS) {
      auto elite_id = _eliteTeamPS[VectorToStringNoSpace(set)][GetState("fitMode")]
                                  [GetState("phase")]
                                      ->id_;
      const bool is_full_task_set =
          set.size() == static_cast<size_t>(GetState("n_task"));
      if (is_full_task_set &&
          haveEliteTeam(VectorToStringNoSpace(set), GetState("fitMode"),
                        GetState("phase"))) {
         oss << "setElTmsMTA eLSz " << _numEliteTeamsCurrent[GetState("phase")]
             << " ss " << VectorToStringNoSpace(set) << " fm " << GetState("fitMode")
             << " minThr "
             << _eliteTeamPS[VectorToStringNoSpace(set)][GetState("fitMode")]
                            [GetState("phase")]
                                ->fit_
             << " ";
         printTeamInfo(GetState("t_current"), GetState("phase"), false,
                       true, elite_id);

         if (GetParam<int>("track_experiments") &&
             GetState("t_current") % GetParam<int>("track_mod") == 0) {
            trackTeamInfo(GetState("t_current"), GetState("phase"), false,
                          elite_id);
         }

         if (HaveParam("save_champ_checkpoints") &&
             GetState("phase") == GetParam<int>("save_champ_checkpoints") &&
             elite_team_id_history_.find(elite_id) ==
                 elite_team_id_history_.end()) {
            elite_team_id_history_.insert(elite_id);
            WriteCheckpoint(false);
         }
      } else if (set.size() == 1 &&
                 haveEliteTeam(VectorToStringNoSpace(set), GetState("fitMode"),
                               GetState("phase"))) {
         oss << "setElTmsST eLSz " << _numEliteTeamsCurrent[GetState("phase")]
             << " ss " << VectorToStringNoSpace(set) << " fm " << GetState("fitMode")
             << " minThr "
             << _eliteTeamPS[VectorToStringNoSpace(set)][GetState("fitMode")]
                            [GetState("phase")]
                                ->fit_
             << " ";
         printTeamInfo(GetState("t_current"), GetState("phase"), false,
                       false, elite_id);

         if (GetParam<int>("track_experiments") &&
             GetState("t_current") % GetParam<int>("track_mod") == 0) {
            trackTeamInfo(GetState("t_current"), GetState("phase"), false,
                          elite_id);
         }
      }
   }
}

/******************************************************************************/
// Helper struct for distance comparisons
struct distanceInstance {
   double distance;
   bool fromArchive;
   distanceInstance(double d, bool a) : distance(d), fromArchive(a) {}
};

/******************************************************************************/
bool compareByDistance(const distanceInstance& a, const distanceInstance& b) {
   return a.distance < b.distance;
}

/******************************************************************************/
void TPG::InitTeams() {
   int max_discrete_action = GetParam<int>("n_discrete_action") > 0
                                 ? GetParam<int>("n_discrete_action") - 1
                                 : 0;
   uniform_int_distribution<int> dis_actions(0, max_discrete_action);
   uniform_int_distribution<int> dis_team_size(
       GetParam<int>("min_initial_team_size"), GetParam<int>("max_initial_team_size"));
   int initial_team_size = dis_team_size(rngs_[TPG_SEED]);
   uniform_int_distribution<int> dis_obs_index(0, 100000);
   int team_obs_size;
   uniform_real_distribution<double> dis_td_values(0.0,1.0);
   // normal_distribution<double> dis_td_values(0.5, 0.5);
   
   double lambda_gen_val = GetParam<double>("lambda_td");
   double decay_gen_val = GetParam<double>("decay_factor");
   for (int t = 0; t < GetParam<int>("n_root"); t++) {
      if (lambda_gen_val == 0.0){
         lambda_gen_val = clamp(dis_td_values(rngs_[TPG_SEED]), 0.0, 1.0);
      }
      if (decay_gen_val == 0.0){
         decay_gen_val = clamp(dis_td_values(rngs_[TPG_SEED]), 0.0, 1.0);
      }
      if(GetParam<int>("team_obs_size"))
         team_obs_size = dis_obs_index(rngs_[TPG_SEED]);
      else 
         team_obs_size = 0;
      auto new_team = new team(GetState("t_current"), state_["team_count"]++, team_obs_size, lambda_gen_val, decay_gen_val);
      for (int p = 0; p < initial_team_size; p++) {
         // Discrete atomic actions are negatives -1 to -numAtomicActions()
         
         long discrete_action = -1 - dis_actions(rngs_[TPG_SEED]);
         auto new_prog = new RegisterMachine(discrete_action, team_obs_size, params_, state_,
                                             rngs_[TPG_SEED], _ops);
         new_team->AddProgram(new_prog);
         AddProgram(new_prog);  // add program to program population
      }
      AddTeam(new_team);  // add team to team population
      phyloRecord p;      // TODO(skelly): make pointer?
      phylo_graph_.insert(pair<long, phyloRecord>(new_team->id_, p));
      phylo_graph_[new_team->id_].gtime = 0;
   }
   oss << "InitTms Msz " << team_pop_.size() << " Lsz " << program_pop_.size()
       << " rSz " << GetRootTeamsInVec().size() << " mSz";
   for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
      oss << " " << _Memory[mem_t].size();
   }
   oss << " eLSz " << _numEliteTeamsCurrent[GetState("phase")] << endl;
}

/******************************************************************************/
// Certain parameters must be processed here
void TPG::ProcessParams() {
   Seed(TPG_SEED, GetParam<int>("seed_tpg"));
   Seed(AUX_SEED, GetParam<int>("seed_aux"));
   // Replaying will require starting from checkpoint
   if (GetParam<int>("replay"))
      params_["start_from_checkpoint"] = 1;
   // Set the starting and current generation (t_start and t_current)
   if (GetParam<int>("start_from_checkpoint")) {
      state_["t_start"] = GetParam<int>("checkpoint_in_t") + 1;
   } else {
      state_["t_start"] = 0;
   }
   state_["t_current"] = GetState("t_start");
}


/******************************************************************************/
// Parameters are set in the parameters.yaml file.
// TPG can also process command line parameters in the form: <name>=<value>
// <name> must be a parameter with a default value in parameters.txt
// Default values are overwritten by command line parameters
void TPG::SetParams(int argc, char** argv) {
   // First read parameters file
   // ReadParameters("parameters.yaml", params_);
   // Parse command line parameters
   params_["pid"] = 0; // Set default param value for PID
   // Keep this opt-in parameter available to every existing configuration and
   // allow it to be overridden on the command line.
   params_["shadow_run"] = 0;
   // Baldwinian inheritance remains the default. Configurations can opt into
   // writing parent-produced S2-S5 outputs into offspring constants.
   params_["lamarkism_evolved_constants"] = 0;
   // By default S2-S6 constants retain their historical ability to evolve.
   // Set this to 0 for the fixed-constant self-modification ablation.
   params_["evolve_self_modifying_constants"] = 1;
   // Preserve parent-produced rates through offspring mutation by default.
   // Set to 1 to reset working S2-S6 from constants before mutation instead.
   params_["reset_self_modifying_before_mutation"] = 0;
   if (argc > 1) {
      for (int i = 1; i < argc; ++i) {
         std::string arg = argv[i];
         size_t pos = arg.find('=');
         if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);
            std::string val = arg.substr(pos + 1);
            if (key == "parameters_file") {
               ReadParameters(val, params_);
            }
            else if (HaveParam(key)) {
               if (params_[key].type() == typeid(std::string)) {
                  params_[key] = val;
               } else if (val.find('.') != std::string::npos) {
                  params_[key] = stringToDouble(val);
               } else {
                  params_[key] = stringToInt(val);
               }
            } else {
               std::string err_message =
                   "Unreconised command line parameter:" + key +
                   ". Command line parameters must have default values in "
                   "parameters.yaml";
               die(__FILE__, __FUNCTION__, __LINE__, err_message.c_str());
            }
         }
      }
   }
   ProcessParams();
}


/******************************************************************************/
void TPG::GetPolicyFeatures(int hostId, set<long>& features, bool active) {
   features.clear();
   set<team*, teamIdComp> visited_teams;
   if (hostId == -1) {
      for (auto tm : team_pop_) {
         if (tm->root_) {
            visited_teams.clear();
            tm->policyFeatures(team_map_, visited_teams, features, active);
         }
      }
   } else {
      team_map_[hostId]->policyFeatures(team_map_, visited_teams, features,
                                       active);
   }
}

// /******************************************************************************/
// // Print graph defined by <rootTeam> in DOT format for GraphViz
// void TPG::printGraphDot(
//     team *rootTeam, size_t frame, int episode, int step, size_t depth,
//     vector<RegisterMachine *> allPrograms, vector<RegisterMachine *> winningPrograms,
//     vector<set<long>> decisionFeatures,
//     vector<set<MemoryEigen *, MemoryEigenIdComp>> decisionMemories,
//     vector<team *> teamPath, bool drawPath,
//     set<team *, teamIdComp> visitedTeamsAllTasks) {
//   // unused arguments
//   (void)decisionFeatures;
//   (void)decisionMemories;
//   (void)allPrograms;

//   // just use winning programs up to a specific graph depth
//   // vector<program*> winningProgramsDepth(winningPrograms.begin(),
//   // winningPrograms.begin()+depth);

//   vector<RegisterMachine *> winningProgramsDepth(winningPrograms.begin(),
//                                          winningPrograms.end());

//   double nodeWidth = 2.0;
//   double edgeWidth_1 = 1;  // 5;
//   double edgeWidth_2 = 30;
//   double arrowSize_1 = 1;  // 0.1;
//   double arrowSize_2 = 1;  // 0.2;

//   char outputFilename[80];
//   ofstream ofs;

//   set<team *, teamIdComp> teams;
//   set<RegisterMachine *, RegisterMachineIdComp> programs;
//   set<MemoryEigen *, MemoryEigenIdComp> memories;

//   //(void)visitedTeamsAllTasks;
//   for (auto it = visitedTeamsAllTasks.begin(); it !=
//   visitedTeamsAllTasks.end();
//        it++) {
//     set<RegisterMachine *, RegisterMachineIdComp> p = (*it)->CopyMembers();
//     programs.insert(p.begin(), p.end());
//   }
//   for (auto it = programs.begin(); it != programs.end(); it++) {
//     for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
//       memories.insert((*it)->MemGet(mem_t));
//     }
//   }

//   sprintf(outputFilename, "replay/graphs/gv_%d_%05d_%03d_%05d_%05d%s",
//           (int)rootTeam->id_, (int)frame, episode, step, (int)depth, ".dot");
//   ofs.open(outputFilename, ios::out);
//   if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

//   ofs << "digraph G {" << endl;
//   ofs << "ratio=1" << endl;
//   ofs << "root=t_" << rootTeam->id_ << endl;

//   ////atomic actions
//   // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++)
//   //    if ((*leiter)->action_ < 0)
//   //       ofs << " a_" << ((*leiter)->action_*-1)-1 << "_" <<
//   (*leiter)->id_
//   //       << " [shape=point, label=\"\", regular=1, width=0.1]" << endl;

//   // programs
//   for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
//     if (step > 0 &&
//         find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//              *leiter) != winningProgramsDepth.end()) {
//       // ofs << " p_" << (*leiter)->id_ << " [shape=box, style=filled,
//       // color=black, label=\"\", fontsize=200, regular=1, width=" <<
//       nodeWidth
//       // << "]" << endl;
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=green, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//     } else if (step > 0 && find(allPrograms.begin(), allPrograms.end(),
//                                 *leiter) != allPrograms.end())
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=black, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//     else if (teamPath.size() > 0)
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=grey90, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//     else
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=grey70, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//   }

//   // teams
//   int label = 0;
//   // for(auto teiter = teams.begin(); teiter != teams.end(); teiter++)
//   for (auto teiter = visitedTeamsAllTasks.begin();
//        teiter != visitedTeamsAllTasks.end(); teiter++)
//     if ((*teiter)->id_ == rootTeam->id_ ||
//         (step > 0 &&
//          find(teamPath.begin(), teamPath.end(), *teiter) != teamPath.end()))
//       ofs << " t_" << (*teiter)->id_
//           << " [shape=circle, style=filled, fillcolor=black, label=\"t"
//           << label++ << "\", fontsize=84, regular=1, width=" << nodeWidth * 2
//           << "]" << endl;
//     else
//       // ofs << " t_" << (*teiter)->id_ << " [shape=circle, style=filled,
//       // fillcolor=grey90, label=\"\", fontsize=84, regular=1, width=" <<
//       // nodeWidth*2 << "]" << endl;
//       ofs << " t_" << (*teiter)->id_
//           << " [shape=circle, style=filled, fillcolor=deepskyblue,
//           label=\"\", "
//              "fontsize=84, regular=1, width="
//           << nodeWidth * 2 << "]" << endl;

//   ////MemoryEigen registers
//   // for(auto meiter = memories.begin(); meiter != memories.end(); meiter++)
//   //    ofs << " m_" << (*meiter)->id_ << " [shape=invhouse, style=filled,
//   //    fillcolor=grey, label=\"\", regular=1, width=" << nodeWidth << "]" <<
//   //    endl;

//   // program -> team edges
//   for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
//     if ((*leiter)->action_ >= 0 &&
//         find(visitedTeamsAllTasks.begin(), visitedTeamsAllTasks.end(),
//              team_map_[(*leiter)->action_]) != visitedTeamsAllTasks.end()) {
//       double w = find(winningProgramsDepth.begin(),
//       winningProgramsDepth.end(),
//                       *leiter) == winningProgramsDepth.end() ||
//                          !drawPath
//                      ? edgeWidth_1
//                      : edgeWidth_2;
//       if (teamPath.size() < 1) w = 5;
//       double as = find(winningProgramsDepth.begin(),
//       winningProgramsDepth.end(),
//                        *leiter) == winningProgramsDepth.end() ||
//                           !drawPath
//                       ? arrowSize_1
//                       : arrowSize_2;
//       string col =
//           find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//                *leiter) == winningProgramsDepth.end()
//               ? "black"
//               : "green";
//       ofs << " p_" << (*leiter)->id_ << "->"
//           << "t_" << (*leiter)->action_ << " [arrowsize=" << as
//           << ", penwidth=" << w << " color=" << col.c_str() << "];" << endl;
//     }
//   }

//   ////program -> MemoryEigen edges
//   // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++){
//   //    double w = find(winningProgramsDepth.begin(),
//   //    winningProgramsDepth.end(), *leiter) == winningProgramsDepth.end() ?
//   //    edgeWidth_1 : edgeWidth_2; double as =
//   //    find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//   *leiter)
//   //    == winningProgramsDepth.end() ? arrowSize_1 : arrowSize_2; ofs << "
//   p_"
//   //    << (*leiter)->id_ << "->" << "m_"<< (*leiter)->MemGet()->id_;
//   //       ofs << " [dir=both, arrowsize=" << as << ", penwidth=" << w <<
//   "];"
//   //       << endl;
//   // }

//   // team -> program edges
//   // for(auto teiter = teams.begin(); teiter != teams.end(); teiter++){
//   for (auto teiter = visitedTeamsAllTasks.begin();
//        teiter != visitedTeamsAllTasks.end(); teiter++) {
//     list<RegisterMachine *> mem;
//     (*teiter)->members(&mem);
//     for (auto leiter = mem.begin(); leiter != mem.end(); leiter++) {
//       double w =
//           find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end()
//           ||
//                   !drawPath
//               ? edgeWidth_1
//               : edgeWidth_2;
//       if (teamPath.size() < 1) w = 5;
//       double as =
//           find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end()
//           ||
//                   !drawPath
//               ? arrowSize_1
//               : arrowSize_2;
//       string col =
//           find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//                *leiter) == winningProgramsDepth.end()
//               ? "black"
//               : "green";
//       ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_
//           << " [arrowsize=" << as << ", penwidth=" << w
//           << " color=" << col.c_str() << "];" << endl;
//     }
//   }
//   ofs << "}" << endl;
//   ofs.close();
// }

// /******************************************************************************/
// void TPG::printGraphDotGPEM(long rootTeamId, map<long, string> &teamColMap,
//                             set<team *, teamIdComp> &visitedTeamsAllTasks,
//                             vector<map<long, double>> &teamUseMapPerTask) {
//   team *rootTeam = team_map_[rootTeamId];

//   (void)teamColMap;
//   vector<string> taskCol;
//   taskCol.push_back("#7fc97f");
//   taskCol.push_back("#beaed4");
//   taskCol.push_back("#fdc086");
//   taskCol.push_back("#ffff99");
//   taskCol.push_back("#386cb0");
//   taskCol.push_back("#f0027f");
//   map<long, string> nodeLabMap;

//   nodeLabMap[1594815] = "1";  //   29000"
//   nodeLabMap[624659] = "2";   //  149"
//   nodeLabMap[1302870] = "3";  //   28373"
//   nodeLabMap[623346] = "4";   //  580"
//   nodeLabMap[493990] = "5";   //  149"
//   nodeLabMap[836830] = "6";   //  28019"
//   nodeLabMap[548151] = "7";   //  832"
//   nodeLabMap[126871] = "8";   //  1373"
//   nodeLabMap[425177] = "9";   //  774"
//   nodeLabMap[602173] = "10";  //   1826"
//   nodeLabMap[42314] = "11";   //  9"
//   nodeLabMap[26879] = "12";   //  7"
//   nodeLabMap[200127] = "13";  //   4"
//   nodeLabMap[470578] = "14";  //   1826"
//   nodeLabMap[5964] = "15";    // 7"
//   nodeLabMap[23266] = "16";   //  2"
//   nodeLabMap[226807] = "17";  //   394"
//   nodeLabMap[180005] = "18";  //   953"

//   double nodeWidth = 2.0;
//   double arrowSize_1 = 3;  // 0.1;

//   char outputFilename[80];
//   ofstream ofs;

//   set<team *, teamIdComp> teams;
//   set<RegisterMachine *, RegisterMachineIdComp> programs;
//   set<MemoryEigen *, MemoryEigenIdComp> memories;

//   for (auto it = visitedTeamsAllTasks.begin(); it !=
//   visitedTeamsAllTasks.end();
//        it++) {
//     set<RegisterMachine *, RegisterMachineIdComp> p = (*it)->CopyMembers();  // no need o
//     copy programs.insert(p.begin(), p.end());
//   }
//   for (auto it = programs.begin(); it != programs.end(); it++) {
//     for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
//       memories.insert((*it)->MemGet(mem_t));
//     }
//   }

//   sprintf(outputFilename,
//           "replay/gv_taskDecomposition_%d_%05d_%03d_%05d_%05d%s",
//           (int)rootTeam->id_, 0, 0, 0, 0, ".dot");
//   ofs.open(outputFilename, ios::out);
//   if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

//   ofs << "strict digraph G {" << endl;
//   ofs << "ratio=0.7" << endl;
//   ofs << "root=t_" << rootTeam->id_ << endl;

//   ////programs
//   // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++){
//   //    ofs << " p_" << (*leiter)->id_ << " [shape=box, style=filled,
//   //    color=grey70, label=\"\", fontsize=200, regular=1, width=" <<
//   nodeWidth
//   //    << "]" << endl;
//   // }

//   // teams
//   for (auto teiter = visitedTeamsAllTasks.begin();
//        teiter != visitedTeamsAllTasks.end(); teiter++) {
//     string col = "";
//     // if (teamColMap.find((*teiter)->id_) != teamColMap.end())
//     //    col = teamColMap[(*teiter)->id_];
//     // else
//     //    col = "white";

//     ofs << " t_" << (*teiter)->id_
//         << " [shape=circle, style=wedged, fillcolor=\"";
//     double sumUse = 0;
//     for (int tsk = 0; tsk < 6; tsk++)
//       sumUse += teamUseMapPerTask[tsk][(*teiter)->id_];
//     for (int tsk = 0; tsk < 6; tsk++) {
//       ofs << taskCol[tsk] << ";"
//           << (teamUseMapPerTask[tsk].find((*teiter)->id_) !=
//                       teamUseMapPerTask[tsk].end()
//                   ? teamUseMapPerTask[tsk][(*teiter)->id_] / sumUse
//                   : 0);
//       if (tsk < 5) ofs << ":";
//     }
//     // ofs << ":" << taskCol[1] << ";"<<
//     // teamUseMapPerTask[0].find((*teiter)->id_) !=
//     teamUseMapPerTask[0].end()
//     // ? teamUseMapPerTask[0][(*teiter)->id_]/6: "0"; ofs << ":" <<
//     taskCol[2]
//     // << ";"<< teamUseMapPerTask[0].find((*teiter)->id_) !=
//     // teamUseMapPerTask[0].end() ? teamUseMapPerTask[0][(*teiter)->id_]/6:
//     // "0"; ofs << ":" << taskCol[3] << ";"<<
//     // teamUseMapPerTask[0].find((*teiter)->id_) !=
//     teamUseMapPerTask[0].end()
//     // ? teamUseMapPerTask[0][(*teiter)->id_]/6: "0"; ofs << ":" <<
//     taskCol[4]
//     // << ";"<< teamUseMapPerTask[0].find((*teiter)->id_) !=
//     // teamUseMapPerTask[0].end() ? teamUseMapPerTask[0][(*teiter)->id_]/6:
//     // "0"; ofs << ":" << taskCol[5] << ";"<<
//     // teamUseMapPerTask[0].find((*teiter)->id_) !=
//     teamUseMapPerTask[0].end()
//     // ? teamUseMapPerTask[0][(*teiter)->id_]/6: "0";
//     ofs << "\"";

//     ofs << ", label=\""
//         << (nodeLabMap.find((*teiter)->id_) == nodeLabMap.end()
//                 ? ""
//                 : nodeLabMap[(*teiter)->id_])
//         << "\", fontsize=84, regular=1, width=" << nodeWidth * 2
//         << ",penwidth=0]" << endl;
//     // ofs << " t_" << (*teiter)->id_ << " [shape=circle, style=wedged,
//     // fillcolor=\"" << col << "\", label=\"\", fontsize=84, regular=1,
//     width="
//     // << nodeWidth*2 << "]" << endl;
//   }

//   ////program -> team edges
//   // for(auto leiter = programs.begin(); leiter != programs.end(); leiter++){
//   //    if ((*leiter)->action_ >= 0 && find(visitedTeamsAllTasks.begin(),
//   //    visitedTeamsAllTasks.end(), team_map_[(*leiter)->action_]) !=
//   //    visitedTeamsAllTasks.end()){
//   //       ofs << " p_" << (*leiter)->id_ << "->" << "t_"<<
//   (*leiter)->action_
//   //       << " [arrowsize=" << arrowSize_1 << ", penwidth=" << "1" << "
//   color="
//   //       << "black" << "];" << endl;
//   //    }
//   // }

//   ////team -> program edges
//   // for(auto teiter = visitedTeamsAllTasks.begin(); teiter !=
//   // visitedTeamsAllTasks.end(); teiter++){
//   //    list < RegisterMachine * > mem;
//   //    (*teiter)->members(&mem);
//   //    for(auto leiter = mem.begin(); leiter != mem.end(); leiter++){
//   //       ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_ << "
//   //       [arrowsize=" << arrowSize_1  << ", penwidth=" << "1" << " color="
//   <<
//   //       "black" << "];" << endl;
//   //    }
//   // }

//   // team -> team edges
//   for (auto teiter = visitedTeamsAllTasks.begin();
//        teiter != visitedTeamsAllTasks.end(); teiter++) {
//     list<RegisterMachine *> mem;
//     (*teiter)->members(&mem);
//     for (auto leiter = mem.begin(); leiter != mem.end(); leiter++) {
//       // ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_ << "
//       // [arrowsize=" << arrowSize_1  << ", penwidth=" << "1" << " color=" <<
//       // "black" << "];" << endl;
//       if ((*leiter)->action_ >= 0 &&
//           find(visitedTeamsAllTasks.begin(), visitedTeamsAllTasks.end(),
//                team_map_[(*leiter)->action_]) != visitedTeamsAllTasks.end())
//                {
//         ofs << " t_" << (*teiter)->id_ << "->t_" << (*leiter)->action_
//             << " [arrowsize=" << arrowSize_1 << ", penwidth="
//             << "1"
//             << " color="
//             << "black"
//             << "];" << endl;
//       }
//     }
//   }

//   ////legend
//   // ofs << "subgraph {" << endl;
//   // ofs << "ratio=1" << endl;
//   // ofs << "rank=sink" << endl;
//   // ofs << "node [shape=plaintext]" << endl;
//   // ofs << "legend [colorscheme=set18," << endl;
//   // ofs << "label=<" << endl;
//   //
//   // ofs << "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\">" <<
//   endl;
//   // ofs << "<tr><td bgcolor=\"" << taskCol[0] << "\">" << "CartPole" <<
//   // "</td></tr>" << endl; ofs << "<tr><td bgcolor=\"" << taskCol[1] << "\">"
//   <<
//   // "Acrobot" << "</td></tr>" << endl; ofs << "<tr><td bgcolor=\"" <<
//   // taskCol[2] << "\">" << "CartCentering" << "</td></tr>" << endl; ofs <<
//   // "<tr><td bgcolor=\"" << taskCol[3] << "\">" << "Pendulum" <<
//   "</td></tr>"
//   // << endl; ofs << "<tr><td bgcolor=\"" << taskCol[4] << "\">" <<
//   // "MountainCar" << "</td></tr>" << endl; ofs << "<tr><td bgcolor=\"" <<
//   // taskCol[5] << "\">" << "MountainCarC." << "</td></tr>" << endl;
//   //
//   // ofs << "</table>>" << endl;
//   // ofs << ", fontsize=84, regular=1];" << endl;
//   // ofs << "}" << endl;
//   ///////////////////////////////////////////////////////////

//   ofs << "}" << endl;
//   ofs.close();
// }

/******************************************************************************/
void TPG::printGraphDotGPTPXXI(long rootTeamId,
                               std::set<team*, teamIdComp>& visitedTeamsAllTasks,
                               std::vector<std::map<long, double>>& teamUseMapPerTask,
                               std::vector<int>& steps_per_task)
{
    std::filesystem::create_directories("graphs");

    // Find root team
    auto it = team_map_.find(rootTeamId);
    if (it == team_map_.end()) {
        std::string msg = "Root team not found: " + std::to_string(rootTeamId);
        die(__FILE__, __FUNCTION__, __LINE__, msg.c_str());
    }
    team* rootTeam = it->second;

    std::string outputFilename = "graphs/team_program_listing_" +
        std::to_string(rootTeam->id_) + ".txt";
    std::ofstream ofs(outputFilename, std::ios::out);
    if (!ofs) {
        std::string msg = "Can't open file: " + outputFilename;
        die(__FILE__, __FUNCTION__, __LINE__, msg.c_str());
    }

    ofs << "==== Tangled Program Graph Listing (root team "
        << rootTeam->id_ << ") ====" << std::endl
        << std::endl;

    for (auto tm : visitedTeamsAllTasks) {
        ofs << "Team " << tm->id_ << std::endl;
        ofs << "----------------------------------------" << std::endl;
        for (auto prog : tm->members_) {
            ofs << "  Program " << prog->id_
                << " | init_noise: " << prog->initial_heb_noise_
                << " | learning_rate: " << prog->learning_rate_
                << " | ptr: " << prog->action_;
            if (prog->action_ >= 0 && team_map_.count(prog->action_))
                ofs << " (Team " << prog->action_ << ")";
            ofs << std::endl;
        }
        ofs << std::endl;
    }
}
// /******************************************************************************/

void TPG::printGraphDotMujoco(long rootTeamId,
                              std::set<team*>& visited)
{
    using std::ostringstream;
    using std::queue;

    std::filesystem::create_directories("graphs");

    std::string outFilename = "graphs/gv_mujoco.dot";
    std::ofstream ofs(outFilename);
    if (!ofs) { die(__FILE__, __func__, __LINE__, ("Can't open file: " + outFilename).c_str()); }

    std::string listFilename = "graphs/program_listing.txt";
    std::ofstream listfs(listFilename);
    if (!listfs) {
        die(__FILE__, __func__, __LINE__, ("Can't open file: " + listFilename).c_str());
    }
    listfs << "Program listing for root team " << rootTeamId << '\n' << std::endl;

    ofs << "digraph Agent {\n"
        << "  rankdir = LR;\n"
        << "  node [fontname = \"Helvetica\", fontsize = 10];\n\n";

    auto writeTeamNode = [&](team* t)
    {
        ostringstream lbl;
        lbl << "{ Team " << t->id_ << " | γ = " << t->decay_factor_ << " }";
        ofs << "  T" << t->id_ << " [shape=Mrecord, style=filled, fillcolor=\"#c6dbef\", label=\""
            << lbl.str() << "\", fontsize=36, width=3.5, height=1.6, margin=\"0.30,0.20\", penwidth=2];\n";
    };

auto writeProgNode = [&](RegisterMachine* p)
{
    long vec_dim = 0;
    long mat_dim = 0;

    // Vector register dimension (per program)
    if (p->private_memory_.size() > static_cast<size_t>(MemoryEigen::kVectorType_) &&
        p->private_memory_[MemoryEigen::kVectorType_] != nullptr) {
        vec_dim = static_cast<long>(
            p->private_memory_[MemoryEigen::kVectorType_]->memory_size_);
    }

    // Matrix register dimension (per program) — matrices are mat_dim x mat_dim
    if (p->private_memory_.size() > static_cast<size_t>(MemoryEigen::kMatrixType_) &&
        p->private_memory_[MemoryEigen::kMatrixType_] != nullptr) {
        mat_dim = static_cast<long>(
            p->private_memory_[MemoryEigen::kMatrixType_]->memory_size_);
    }

    ofs << "  P" << p->id_
        << " [shape=record, style=filled, fillcolor=\"#fee0d2\", "
        << "label=\"{ Program " << p->id_
        << " | idx=" << p->obs_index_
        << " | V=" << vec_dim
        << " | M=" << mat_dim << "x" << mat_dim
        << " }\", fontsize=20];\n";
};

    queue<long> toVisit;
    std::unordered_set<long> printedTeams;
    std::unordered_set<long> printedProgramNodes;
    std::unordered_set<long> printedProgramEdges;
    std::set<std::pair<long,long>> printedTeamProgramEdges;
    toVisit.push(rootTeamId);

    team* root = team_map_.at(rootTeamId);
    writeTeamNode(root);
    printedTeams.insert(rootTeamId);
    visited.insert(root);

    while (!toVisit.empty()) {
        long tid = toVisit.front(); toVisit.pop();
        team* t   = team_map_.at(tid);

        for (auto p : t->members_) {
            if (printedTeamProgramEdges.insert({tid, p->id_}).second) {
                ofs << "  T" << tid << " -> P" << p->id_ << " [penwidth=1.8];\n";
            }
            if (printedProgramNodes.insert(p->id_).second) {
                writeProgNode(p);
            }

            listfs << "Team " << tid
                   << " | Program " << p->id_
                   << " | idx=" << p->obs_index_
                   << " | vec/mat size=" << p->observation_buff_size_
                   << " -> ";
            if (p->action_ >= 0) {
                listfs << "Team " << p->action_;
            } else {
                listfs << "Atomic action " << p->action_;
            }
            listfs << std::endl;

            if (p->action_ >= 0) {
                long childId = p->action_;
                if (printedProgramEdges.insert(p->id_).second) {
                    ofs << "  P" << p->id_ << " -> T" << childId << " [arrowhead=normal];\n";

                    if (!printedTeams.count(childId)) {
                        team* child = team_map_.at(childId);
                        writeTeamNode(child);
                        printedTeams.insert(childId);
                    }
                }

                if (visited.emplace(team_map_.at(childId)).second) {
                    toVisit.push(childId);
                }
            }
        }
    }

    ofs << "}\n";
    ofs.close();
    listfs.close();
    {
        std::string pngFilename = outFilename.substr(0, outFilename.find_last_of('.')) + ".png";
        std::string cmd = "dot -Tpng " + outFilename + " -o " + pngFilename;
        std::system(cmd.c_str());
    }
}

void TPG::printGraphDotMujocoVisited(long rootTeamId,
                                     std::set<team*>& visitedTeams)
{
    using std::ostringstream;

    std::filesystem::create_directories("graphs");

    std::string outFilename = "graphs/gv_mujoco_visited.dot";
    std::ofstream ofs(outFilename);
    if (!ofs)
        die(__FILE__, __func__, __LINE__,
            ("Can't open file: " + outFilename).c_str());

    std::string listFilename = "graphs/program_listing_visited.txt";
    std::ofstream listfs(listFilename);
    if (!listfs)
        die(__FILE__, __func__, __LINE__,
            ("Can't open file: " + listFilename).c_str());

    listfs << "Program listing (visited only) for root team "
           << rootTeamId << '\n' << std::endl;

    ofs << "digraph Agent {\n"
        << "  rankdir = LR;\n"
        << "  node [fontname = \"Helvetica\", fontsize = 10];\n\n";

    auto writeTeamNode = [&](team* t)
    {
        ostringstream lbl;
        lbl << "{ Team " << t->id_
           << " | γ = " << t->decay_factor_ <<" }";
        ofs << "  T" << t->id_
            << " [shape=Mrecord, style=filled, fillcolor=\"#c6dbef\", "
            << "label=\"" << lbl.str() << "\", fontsize=36, width=3.5, height=1.6, margin=\"0.30,0.20\", penwidth=2];\n";
    };

    auto writeProgNode = [&](RegisterMachine* p)
    {
        ofs << "  P" << p->id_
            << " [shape=record, style=filled, fillcolor=\"#fee0d2\", "
            << "label=\"{ Program " << p->id_
            << " | LR=" << std::fixed << std::setprecision(2)
            << p->learning_rate_ << " }\", fontsize=20];\n";
    };

    std::set<team*> vis = visitedTeams;
    team* root = team_map_.at(rootTeamId);
    vis.insert(root);

    std::unordered_set<long> printedTeams;
    std::unordered_set<long> printedProgNodes;
    std::unordered_set<long> printedProgEdges;
    std::set<std::pair<long,long>> printedTeamProgramEdges;

    /* Emit team nodes first */
    for (auto tm : vis) {
        writeTeamNode(tm);
        printedTeams.insert(tm->id_);
    }

    /* Emit program nodes and edges, restricted to the visited subset */
    for (auto tm : vis) {
        for (auto p : tm->members_) {
            if (printedTeamProgramEdges.insert({tm->id_, p->id_}).second) {
                ofs << "  T" << tm->id_ << " -> P" << p->id_
                    << " [penwidth=1.8];\n";
            }

            if (printedProgNodes.insert(p->id_).second)
                writeProgNode(p);

            listfs << "Team " << tm->id_
                   << " | Program " << p->id_ << " -> ";
            if (p->action_ >= 0)
                listfs << "Team " << p->action_;
            else
                listfs << "Atomic action " << p->action_;
            listfs << std::endl;

            if (p->action_ >= 0 &&
                printedTeams.count(p->action_) &&
                printedProgEdges.insert(p->id_).second) {
                ofs << "  P" << p->id_ << " -> T" << p->action_
                    << " [arrowhead=normal];\n";
            }
        }
    }

    ofs << "}\n";
    ofs.close();
    listfs.close();
    {
        std::string pngFilename = outFilename.substr(0, outFilename.find_last_of('.')) + ".png";
        std::string cmd = "dot -Tpng " + outFilename + " -o " + pngFilename;
        std::system(cmd.c_str());
    }
}
// void TPG::printGraphDotGPEMAnimate(
//     long rootTeamId, size_t frame, int episode, int step, size_t depth,
//     vector<RegisterMachine *> allPrograms, vector<RegisterMachine *> winningPrograms,
//     set<team *, teamIdComp> &visitedTeamsAllTasks,
//     vector<map<long, double>> &teamUseMapPerTask, vector<team *> teamPath) {
//   team *rootTeam = team_map_[rootTeamId];

//   vector<string> taskCol;
//   taskCol.push_back("#7fc97f");
//   taskCol.push_back("#beaed4");
//   taskCol.push_back("#fdc086");
//   taskCol.push_back("#ffff99");
//   taskCol.push_back("#386cb0");
//   taskCol.push_back("#f0027f");
//   map<long, string> nodeLabMap;

//   nodeLabMap[1594815] = "1";  //   29000"
//   nodeLabMap[624659] = "2";   //  149"
//   nodeLabMap[1302870] = "3";  //   28373"
//   nodeLabMap[623346] = "4";   //  580"
//   nodeLabMap[493990] = "5";   //  149"
//   nodeLabMap[836830] = "6";   //  28019"
//   nodeLabMap[548151] = "7";   //  832"
//   nodeLabMap[126871] = "8";   //  1373"
//   nodeLabMap[425177] = "9";   //  774"
//   nodeLabMap[602173] = "10";  //   1826"
//   nodeLabMap[42314] = "11";   //  9"
//   nodeLabMap[26879] = "12";   //  7"
//   nodeLabMap[200127] = "13";  //   4"
//   nodeLabMap[470578] = "14";  //   1826"
//   nodeLabMap[5964] = "15";    // 7"
//   nodeLabMap[23266] = "16";   //  2"
//   nodeLabMap[226807] = "17";  //   394"
//   nodeLabMap[180005] = "18";  //   953"

//   bool drawPath = true;
//   double nodeWidth = 2.0;
//   double arrowSize_1 = 3;
//   double arrowSize_2 = 3;
//   double edgeWidth_2 = 10;
//   double edgeWidth_1 = 1;

//   char outputFilename[80];
//   ofstream ofs;

//   set<team *, teamIdComp> teams;
//   set<RegisterMachine *, RegisterMachineIdComp> programs;
//   set<MemoryEigen *, MemoryEigenIdComp> memories;
//   vector<RegisterMachine *> winningProgramsDepth(winningPrograms.begin(),
//                                          winningPrograms.end());

//   for (auto it = visitedTeamsAllTasks.begin(); it !=
//   visitedTeamsAllTasks.end();
//        it++) {
//     set<RegisterMachine *, RegisterMachineIdComp> p = (*it)->CopyMembers();  // no need to
//     copy programs.insert(p.begin(), p.end());
//   }
//   for (auto it = programs.begin(); it != programs.end(); it++) {
//     for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
//       memories.insert((*it)->MemGet(mem_t));
//     }
//   }

//   sprintf(outputFilename, "replay/graphs/gv_%d_%05d_%03d_%05d_%05d%s",
//           (int)rootTeam->id_, (int)frame, episode, step, (int)depth, ".dot");
//   ofs.open(outputFilename, ios::out);
//   if (!ofs) die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

//   ofs << "strict digraph G {" << endl;
//   ofs << "ratio=0.7" << endl;
//   ofs << "root=t_" << rootTeam->id_ << endl;

//   // programs
//   for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
//     if (step > 0 &&
//         find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//              *leiter) != winningProgramsDepth.end()) {
//       // ofs << " p_" << (*leiter)->id_ << " [shape=box, style=filled,
//       // color=black, label=\"\", fontsize=200, regular=1, width=" <<
//       nodeWidth
//       // << "]" << endl;
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=green, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//     } else if (step > 0 && find(allPrograms.begin(), allPrograms.end(),
//                                 *leiter) != allPrograms.end())
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=black, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//     else if (teamPath.size() > 0)
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=grey90, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//     else
//       ofs << " p_" << (*leiter)->id_
//           << " [shape=box, style=filled, color=grey70, label=\"\", "
//              "fontsize=200, regular=1, width="
//           << nodeWidth << "]" << endl;
//   }

//   // teams
//   for (auto teiter = visitedTeamsAllTasks.begin();
//        teiter != visitedTeamsAllTasks.end(); teiter++) {
//     string col = "";

//     ofs << " t_" << (*teiter)->id_
//         << " [shape=circle, style=wedged, fillcolor=\"";
//     double sumUse = 0;
//     for (int tsk = 0; tsk < 6; tsk++) {
//       if (teamUseMapPerTask[tsk].find((*teiter)->id_) !=
//           teamUseMapPerTask[tsk].end())
//         sumUse += teamUseMapPerTask[tsk][(*teiter)->id_];
//     }
//     for (int tsk = 0; tsk < 6; tsk++) {
//       ofs << taskCol[tsk] << ";"
//           << (teamUseMapPerTask[tsk].find((*teiter)->id_) !=
//                       teamUseMapPerTask[tsk].end()
//                   ? teamUseMapPerTask[tsk][(*teiter)->id_] / sumUse
//                   : 0);
//       if (tsk < 5) ofs << ":";
//     }
//     ofs << "\"";

//     ofs << ", label=\""
//         << (nodeLabMap.find((*teiter)->id_) == nodeLabMap.end()
//                 ? ""
//                 : nodeLabMap[(*teiter)->id_])
//         << "\", fontsize=84, regular=1, width=" << nodeWidth * 2
//         << ",penwidth=0]" << endl;
//   }

//   ////team -> team edges
//   // for(auto teiter = visitedTeamsAllTasks.begin(); teiter !=
//   // visitedTeamsAllTasks.end(); teiter++){
//   //    list < RegisterMachine * > mem;
//   //    (*teiter)->members(&mem);
//   //    for(auto leiter = mem.begin(); leiter != mem.end(); leiter++){
//   //       if ((*leiter)->action_ >= 0 && find(visitedTeamsAllTasks.begin(),
//   //       visitedTeamsAllTasks.end(), team_map_[(*leiter)->action_]) !=
//   //       visitedTeamsAllTasks.end()){
//   //          ofs << " t_" << (*teiter)->id_ << "->t_" << (*leiter)->action_
//   //          << " [arrowsize=" << arrowSize_1  << ", penwidth=" << "1" << "
//   //          color=" << "black" << "];" << endl;
//   //       }
//   //    }
//   // }

//   ////team -> team edges path
//   // if (step > 0){
//   //    for(size_t t = 0; t < teamPath.size()-1; t++){
//   //       list < RegisterMachine * > mem;
//   //       teamPath[t]->members(&mem);
//   //       for(auto leiter = mem.begin(); leiter != mem.end(); leiter++){
//   //          if ((*leiter)->action_ >= 0 && teamPath[t+1]->id_ ==
//   //          team_map_[(*leiter)->action_]->id_){
//   //             ofs << " t_" << teamPath[t]->id_ << "->t_" <<
//   //             (*leiter)->action_ << " [arrowsize=" << arrowSize_2  << ",
//   //             penwidth=" << edgeWidth_2 << " color=" << "black" << "];" <<
//   //             endl;
//   //          }
//   //       }
//   //    }
//   // }

//   // program -> team edges
//   for (auto leiter = programs.begin(); leiter != programs.end(); leiter++) {
//     if ((*leiter)->action_ >= 0 &&
//         find(visitedTeamsAllTasks.begin(), visitedTeamsAllTasks.end(),
//              team_map_[(*leiter)->action_]) != visitedTeamsAllTasks.end()) {
//       double w = find(winningProgramsDepth.begin(),
//       winningProgramsDepth.end(),
//                       *leiter) == winningProgramsDepth.end() ||
//                          !drawPath
//                      ? edgeWidth_1
//                      : edgeWidth_2;
//       if (teamPath.size() < 1) w = 5;
//       double as = find(winningProgramsDepth.begin(),
//       winningProgramsDepth.end(),
//                        *leiter) == winningProgramsDepth.end() ||
//                           !drawPath
//                       ? arrowSize_1
//                       : arrowSize_2;
//       string col =
//           find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//                *leiter) == winningProgramsDepth.end()
//               ? "black"
//               : "green";
//       ofs << " p_" << (*leiter)->id_ << "->"
//           << "t_" << (*leiter)->action_ << " [arrowsize=" << as
//           << ", penwidth=" << w << " color=" << col.c_str() << "];" << endl;
//     }
//   }

//   // team -> program edges
//   for (auto teiter = visitedTeamsAllTasks.begin();
//        teiter != visitedTeamsAllTasks.end(); teiter++) {
//     list<RegisterMachine *> mem;
//     (*teiter)->members(&mem);
//     for (auto leiter = mem.begin(); leiter != mem.end(); leiter++) {
//       double w =
//           find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end()
//           ||
//                   !drawPath
//               ? edgeWidth_1
//               : edgeWidth_2;
//       if (teamPath.size() < 1) w = 5;
//       double as =
//           find(teamPath.begin(), teamPath.end(), *teiter) == teamPath.end()
//           ||
//                   !drawPath
//               ? arrowSize_1
//               : arrowSize_2;
//       string col =
//           find(winningProgramsDepth.begin(), winningProgramsDepth.end(),
//                *leiter) == winningProgramsDepth.end()
//               ? "black"
//               : "green";
//       ofs << " t_" << (*teiter)->id_ << "->p_" << (*leiter)->id_
//           << " [arrowsize=" << as << ", penwidth=" << w
//           << " color=" << col.c_str() << "];" << endl;
//     }
//   }

//   ofs << "}" << endl;
//   ofs.close();
// }

/******************************************************************************/
void TPG::printPhyloGraphDot(team* tm) {
   char outputFilename[80];
   ofstream ofs;

   sprintf(outputFilename, "replay/graphs/phylo-t%05d-s%lu%s",
           (int)GetState("t_current"), seeds_[TPG_SEED], ".dot");
   ofs.open(outputFilename, ios::out);
   if (!ofs)
      die(__FILE__, __FUNCTION__, __LINE__, "Can't open file.");

   ofs << "digraph G {" << endl;

   // Basic breadth-first search

   std::vector<long> visited = {tm->id_};
   list<long> queue = {tm->id_};

   while (!queue.empty()) {
      long currId = queue.front();
      queue.pop_front();

      for (long ancId : phylo_graph_[currId].ancestorIds) {
         ofs << ancId << " -> " << currId << endl;
         if (std::find(visited.begin(), visited.end(), ancId) ==
             visited.end()) {
            visited.push_back(ancId);
            queue.push_back(ancId);
         }
      }
   }

   // Color nodes based on fitness
   for (long id : visited) {
      double fitness = phylo_graph_[id].fitness;
      double hue = std::clamp(fitness, 0.0, 1.0) / 3;

      ofs << id << " [label=\"id: " << id << "\\nfit: " << std::fixed
          << std::setprecision(4) << fitness << "\" style=filled, fillcolor=\""
          << hue << " 1.000 1.000\"]" << endl;
   }

   ofs << "}" << endl;
   ofs.close();
}

double TPG::ComputeTeamFLOPs(const team* t) {
    if (!t) return 0.0;
    double flops_sum = 0.0;

    for (auto* prog : t->members_) {
      if (!prog) continue;
      double mat_size = prog->private_memory_[MemoryEigen::kMatrixType_]->memory_size_;
      for (auto* instr : prog->instructions_effective_) {
         flops_sum += instruction::GetOperationFLOPs(instr->op_, mat_size);
      }
    }
    return flops_sum;
}

/******************************************************************************/
void TPG::printTeamInfo(long t, int phase, bool singleBest, bool multitask, long teamId) {
   (void)multitask;
   team* bestTeam = *(team_pop_.begin());
   if (singleBest && teamId == -1)
      bestTeam = GetBestTeam();
   ostringstream tmposs;
   map<point*, double, pointLexicalLessThan> allOutcomes;
   // map < point *, double > :: iterator myoiter;
   vector<int> behaviourSequence;
   set<team*, teamIdComp> visitedTeams;
   for (auto teiter = team_pop_.begin(); teiter != team_pop_.end(); teiter++) {
      if ((!singleBest && (*teiter)->root() &&
           teamId == -1) ||                             // all root teams
          (!singleBest && (*teiter)->id_ == teamId) ||  // specific team
          (singleBest &&
           (*teiter)->id_ == bestTeam->id_))  // singleBest root team
      {      
         oss << "tminfo t " << t << " id " << (*teiter)->id_ << " gtm "
             << (*teiter)->gtime_ << " phs " << phase;
         oss << " root " << ((*teiter)->root() ? 1 : 0);
         oss << " sz " << (*teiter)->size();
         oss << " age " << t - (*teiter)->gtime_;
         // oss << " compl";
         // oss << " " << (*teiter)->numActiveTeams_;
         // oss << " " << (*teiter)->numActivePrograms_;
         // oss << " " << (*teiter)->numEffectiveInstructions();
         // oss << " " << (*teiter)->numActiveFeatures_;
         oss << " nOut " << (*teiter)->numOutcomes(_TRAIN_PHASE, -1) << " "
             << (*teiter)->numOutcomes(_TEST_PHASE, -1);

         oss << setprecision(5) << fixed;

         oss << " mnOut";            

         for (int phs : {0, 1, 2}) {
            // bool allPhase = false;
            // bool allTask = false;  // for genomic, set to true

            // multi-task
            // statistics stored in aux doubles
            for (int task = 0; task < GetState("n_task"); task++) {
               for (int i = 0; i < GetParam<int>("n_point_aux_double"); i++) {
                  if ((*teiter)->numOutcomes(phs, task) > 0) {
                     oss << " p" << phs << "t" << task << "a" << i << " ";
                     oss << (*teiter)->GetMeanOutcome(phs, task, i);
                  } else
                     oss << " p" << phs << "t" << task << "a" << i << " x";
               }
            }
         }

         oss << " fit " << (*teiter)->fit_;

         oss << setprecision(2) << fixed;

         visitedTeams.clear();
         vector<int> programInstructionCounts,
             effectiveProgramInstructionCounts;
         (*teiter)->policyInstructions(team_map_, visitedTeams,
                                       programInstructionCounts,
                                       effectiveProgramInstructionCounts);

         oss << " pIns "
             << accumulate(programInstructionCounts.begin(),
                           programInstructionCounts.end(), 0);
         oss << " mnProgIns " << VectorMean<int>(programInstructionCounts);

         oss << " ePIns "
             << accumulate(effectiveProgramInstructionCounts.begin(),
                           effectiveProgramInstructionCounts.end(), 0);
         oss << " mnEProgIns "
             << VectorMean<int>(effectiveProgramInstructionCounts);
         set<RegisterMachine*, RegisterMachineIdComp> programs;
         // set<MemoryEigen *, MemoryEigenIdComp> memories;
         set<team*, teamIdComp> visitedTeams2;
         (*teiter)->GetAllNodes(team_map_, visitedTeams2, programs);
         oss << " nP " << programs.size();
         oss << " nT " << visitedTeams2.size();
         // oss << " nM " << memories.size();
         // visitedTeams.clear();
         // set<long> pF;
         // (*teiter)->policyFeatures(team_map_, visitedTeams, pF, true);
         // oss << " pF " << (double)pF.size() / n_input_[];
         // //GetParam<int>("n_input");

         vector<int> op_countsSingle;
         vector<int> op_countsTally;
         op_countsTally.resize(instruction::NUM_OP);

         fill(op_countsTally.begin(), op_countsTally.end(), 0);
         for (auto it = programs.begin(); it != programs.end(); it++) {
            op_countsSingle = (*it)->op_counts_;
            for (size_t i = 0; i < op_countsSingle.size(); i++)
               op_countsTally[i] += op_countsSingle[i];
         }
         oss << " nOp " << VectorToString(op_countsTally);
            // Validation-only, within-generation statistics:
            // validation fitness vs (effective_registers / total_registers)
            // across all root teams that were actually validated.
            double validation_register_efficiency_pearson =
                std::numeric_limits<double>::quiet_NaN();
            double validation_register_efficiency_spearman =
                std::numeric_limits<double>::quiet_NaN();
            int validation_register_efficiency_n = 0;
            if (GetState("validation_ran_this_gen") == 1) {
               const int task = GetState("active_task");
               const int fitMode = GetState("fitMode");

               std::vector<double> fits, props;
               fits.reserve(GetRootTeamsInVec().size());
               props.reserve(GetRootTeamsInVec().size());
               for (auto* tm : GetRootTeamsInVec()) {
                  if (!tm || tm->numOutcomes(_VALIDATION_PHASE, task) <= 0) continue;

                  const int total_reg = tm->TotalScalarRegisters();
                  if (total_reg <= 0) continue;

                  const int eff_reg = tm->TotalEffectiveScalarRegisters();
                  const double validation_fit =
                      tm->GetMeanOutcome(_VALIDATION_PHASE, task, fitMode);

                  fits.push_back(validation_fit);
                  props.push_back(static_cast<double>(eff_reg) /
                                  static_cast<double>(total_reg));
               }

               const size_t n = fits.size();
               validation_register_efficiency_n = static_cast<int>(n);
               if (n >= 2) {
                  auto pearson = [](const std::vector<double>& xs,
                                    const std::vector<double>& ys) -> double {
                     const size_t n = xs.size();
                     double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_y2 = 0.0, sum_xy = 0.0;
                     for (size_t i = 0; i < n; i++) {
                        sum_x += xs[i];
                        sum_y += ys[i];
                        sum_x2 += xs[i] * xs[i];
                        sum_y2 += ys[i] * ys[i];
                        sum_xy += xs[i] * ys[i];
                     }
                     const double nf = static_cast<double>(n);
                     const double num = nf * sum_xy - sum_x * sum_y;
                     const double den_x = nf * sum_x2 - sum_x * sum_x;
                     const double den_y = nf * sum_y2 - sum_y * sum_y;
                     if (den_x <= 1e-20 || den_y <= 1e-20) {
                        return std::numeric_limits<double>::quiet_NaN();
                     }
                     return num / std::sqrt(den_x * den_y);
                  };

                  auto average_ranks = [](const std::vector<double>& values) {
                     std::vector<std::pair<double, size_t>> order;
                     order.reserve(values.size());
                     for (size_t i = 0; i < values.size(); i++) {
                        order.push_back({values[i], i});
                     }
                     std::sort(order.begin(), order.end(),
                               [](const auto& a, const auto& b) { return a.first < b.first; });

                     std::vector<double> ranks(values.size(), 0.0);
                     size_t i = 0;
                     while (i < order.size()) {
                        size_t j = i + 1;
                        while (j < order.size() && order[j].first == order[i].first) {
                           j++;
                        }
                        const double avg_rank =
                            (static_cast<double>(i) + static_cast<double>(j - 1)) / 2.0 + 1.0;
                        for (size_t k = i; k < j; k++) {
                           ranks[order[k].second] = avg_rank;
                        }
                        i = j;
                     }
                     return ranks;
                  };

                  validation_register_efficiency_pearson = pearson(fits, props);
                  validation_register_efficiency_spearman =
                      pearson(average_ranks(fits), average_ranks(props));
               }
            }

            // Get best agent's register size (total and effective)
            // This is the same full-task champion whose ID and fitness are
            // written to the selection row below.
            team* best_agent = *teiter;
            const bool self_modifying =
                std::any_cast<int>(params_["self_modifying"]) != 0;
            double best_agent_register_size = best_agent ? best_agent->AvgScalarRegsPerProgram() : 0.0;
            double best_agent_effective_register_size = 0.0;
            double best_agent_mean_register_similarity = 0.0;
            double best_agent_max_register_similarity = 0.0;
            if (best_agent) {
               int total_effective_regs = 0;
               double sim_sum = 0.0;
               double max_sim = 0.0;
               int prog_count = 0;
               for (auto* prog : best_agent->members_) {
                  if (prog) {
                     total_effective_regs += prog->n_effective_registers_;
                     double mean_s = prog->ComputeMeanPairwiseRegisterSimilarity();
                     double max_s = prog->ComputeMaxPairwiseRegisterSimilarity();
                     sim_sum += mean_s;
                     if (max_s > max_sim) max_sim = max_s;
                     prog_count++;
                  }
               }
               best_agent_effective_register_size = static_cast<double>(total_effective_regs);
               best_agent_mean_register_similarity = prog_count > 0 ? sim_sum / prog_count : 0.0;
               best_agent_max_register_similarity = max_sim;
            }

            // Average self-modifying mutation probabilities across the current
            // training elites. Start rates come from inherited constants;
            // output rates come from working S2-S6; S6 is the neutral decoy.
            double elite_avg_start_rate_swap = 0.0;
            double elite_avg_start_rate_delete = 0.0;
            double elite_avg_start_rate_add = 0.0;
            double elite_avg_start_rate_mutate = 0.0;
            double elite_avg_start_rate_decoy = 0.0;
            double elite_avg_output_rate_swap = 0.0;
            double elite_avg_output_rate_delete = 0.0;
            double elite_avg_output_rate_add = 0.0;
            double elite_avg_output_rate_mutate = 0.0;
            double elite_avg_output_rate_decoy = 0.0;
            int elite_rate_team_count = 0;
            int elite_decoy_team_count = 0;
            if (self_modifying) {
               for (const auto* elite_team : GetRootTeamsInVec()) {
                  if (!elite_team || !elite_team->elite(_TRAIN_PHASE)) {
                     continue;
                  }

                  int team_program_count = 0;
                  double team_output_swap_sum = 0.0;
                  double team_output_delete_sum = 0.0;
                  double team_output_add_sum = 0.0;
                  double team_output_mutate_sum = 0.0;
                  double team_output_decoy_sum = 0.0;
                  double team_start_swap_sum = 0.0;
                  double team_start_delete_sum = 0.0;
                  double team_start_add_sum = 0.0;
                  double team_start_mutate_sum = 0.0;
                  double team_start_decoy_sum = 0.0;
                  int team_decoy_program_count = 0;
                  for (const auto* prog : elite_team->members_) {
                     if (!prog ||
                         prog->private_memory_.size() <= MemoryEigen::kScalarType_) {
                        continue;
                     }
                     const auto* scalar_memory =
                         prog->private_memory_[MemoryEigen::kScalarType_];
                     if (!scalar_memory ||
                         scalar_memory->working_memory_.size() <
                             kSelfModifyingMinScalarRegisters ||
                         scalar_memory->const_memory_.size() <
                             kSelfModifyingMinScalarRegisters) {
                        continue;
                     }
                     const double output_swap =
                         scalar_memory->working_memory_[kSelfModifyingFirstRegister](0, 0);
                     const double output_delete =
                         scalar_memory->working_memory_[kSelfModifyingFirstRegister + 1](0, 0);
                     const double output_add =
                         scalar_memory->working_memory_[kSelfModifyingFirstRegister + 2](0, 0);
                     const double output_mutate =
                         scalar_memory->working_memory_[kSelfModifyingFirstRegister + 3](0, 0);
                     const double start_swap =
                         scalar_memory->const_memory_[kSelfModifyingFirstRegister](0, 0);
                     const double start_delete =
                         scalar_memory->const_memory_[kSelfModifyingFirstRegister + 1](0, 0);
                     const double start_add =
                         scalar_memory->const_memory_[kSelfModifyingFirstRegister + 2](0, 0);
                     const double start_mutate =
                         scalar_memory->const_memory_[kSelfModifyingFirstRegister + 3](0, 0);
                     team_output_swap_sum += SelfModifyingRawTendencyToProbability(output_swap);
                     team_output_delete_sum += SelfModifyingRawTendencyToProbability(output_delete);
                     team_output_add_sum += SelfModifyingRawTendencyToProbability(output_add);
                     team_output_mutate_sum += SelfModifyingRawTendencyToProbability(output_mutate);
                     team_start_swap_sum += SelfModifyingRawTendencyToProbability(start_swap);
                     team_start_delete_sum += SelfModifyingRawTendencyToProbability(start_delete);
                     team_start_add_sum += SelfModifyingRawTendencyToProbability(start_add);
                     team_start_mutate_sum += SelfModifyingRawTendencyToProbability(start_mutate);
                     if (scalar_memory->working_memory_.size() >
                             kSelfModifyingDecoyRegister &&
                         scalar_memory->const_memory_.size() >
                             kSelfModifyingDecoyRegister) {
                        team_output_decoy_sum += SelfModifyingRawTendencyToProbability(
                            scalar_memory->working_memory_[kSelfModifyingDecoyRegister](0, 0));
                        team_start_decoy_sum += SelfModifyingRawTendencyToProbability(
                            scalar_memory->const_memory_[kSelfModifyingDecoyRegister](0, 0));
                        team_decoy_program_count++;
                     }
                     team_program_count++;
                  }
                  if (team_program_count > 0) {
                     elite_avg_output_rate_swap +=
                         team_output_swap_sum / team_program_count;
                     elite_avg_output_rate_delete +=
                         team_output_delete_sum / team_program_count;
                     elite_avg_output_rate_add +=
                         team_output_add_sum / team_program_count;
                     elite_avg_output_rate_mutate +=
                         team_output_mutate_sum / team_program_count;
                     elite_avg_start_rate_swap +=
                         team_start_swap_sum / team_program_count;
                     elite_avg_start_rate_delete +=
                         team_start_delete_sum / team_program_count;
                     elite_avg_start_rate_add +=
                         team_start_add_sum / team_program_count;
                     elite_avg_start_rate_mutate +=
                         team_start_mutate_sum / team_program_count;
                     elite_rate_team_count++;
                     if (team_decoy_program_count > 0) {
                        elite_avg_output_rate_decoy +=
                            team_output_decoy_sum / team_decoy_program_count;
                        elite_avg_start_rate_decoy +=
                            team_start_decoy_sum / team_decoy_program_count;
                        elite_decoy_team_count++;
                     }
                  }
               }
            }
            if (elite_rate_team_count > 0) {
               elite_avg_start_rate_swap /= elite_rate_team_count;
               elite_avg_start_rate_delete /= elite_rate_team_count;
               elite_avg_start_rate_add /= elite_rate_team_count;
               elite_avg_start_rate_mutate /= elite_rate_team_count;
               elite_avg_output_rate_swap /= elite_rate_team_count;
               elite_avg_output_rate_delete /= elite_rate_team_count;
               elite_avg_output_rate_add /= elite_rate_team_count;
               elite_avg_output_rate_mutate /= elite_rate_team_count;
            }
            if (elite_decoy_team_count > 0) {
               elite_avg_start_rate_decoy /= elite_decoy_team_count;
               elite_avg_output_rate_decoy /= elite_decoy_team_count;
            }

            // Compute NSGA-II Pareto front statistics
            int pareto_front_0_size = 0;
            int pareto_front_1_size = 0;
            double avg_complexity_front_0 = 0.0;
            int best_agent_pareto_rank = best_agent ? best_agent->pareto_rank_ : -1;
            double best_agent_crowding_dist = best_agent ? best_agent->crowding_distance_ : 0.0;
            
            // Complexity function (same as in ComputeParetoFronts)
            auto complexity_for_logging = [](const team* t) -> double {
               if (!t || t->members_.empty()) return 0.0;
               double sum = 0.0;
               for (auto* prog : t->members_) {
                  if (prog) sum += static_cast<double>(prog->instructions_.size());
               }
               double mem_sum = 0.0;
               for (auto* prog : t->members_) {
                  if (prog) mem_sum += static_cast<double>(prog->n_memories_);
               }
               return sum + 0.3 * mem_sum;
            };
            
            double complexity_sum_front_0 = 0.0;
            for (auto tm : GetRootTeamsInVec()) {
               if (tm->pareto_rank_ == 0) {
                  pareto_front_0_size++;
                  complexity_sum_front_0 += complexity_for_logging(tm);
               } else if (tm->pareto_rank_ == 1) {
                  pareto_front_1_size++;
               }
            }
            if (pareto_front_0_size > 0) {
               avg_complexity_front_0 = complexity_sum_front_0 / static_cast<double>(pareto_front_0_size);
            }
            if (GetState("phase") == _TRAIN_PHASE) {
               const int task = GetState("active_task");
               const int fitMode = GetState("fitMode");

               const double train_best_fitness =
                   (*teiter)->GetMeanOutcome(_TRAIN_PHASE, task, fitMode);

               double validation_fitness = 0.0;
               if (GetState("validation_ran_this_gen") == 1 &&
                   (*teiter)->numOutcomes(_VALIDATION_PHASE, task) > 0) {
                  validation_fitness =
                      (*teiter)->GetMeanOutcome(_VALIDATION_PHASE, task, fitMode);
               }

               SelectionMetricsBuilder builder;
               builder.with_generation(t)
                   .with_best_fitness(train_best_fitness)
                   .with_validation_fitness(validation_fitness)
                   .with_team_id((*teiter)->id_)
                   .with_team_size((*teiter)->size())
                   .with_age(t - (*teiter)->gtime_)
                   .with_fitness_value_for_selection((*teiter)->fit_)
                   .with_total_program_instructions(accumulate(programInstructionCounts.begin(),
                                                              programInstructionCounts.end(), 0))
                   .with_total_effective_program_instructions(accumulate(effectiveProgramInstructionCounts.begin(),
                                                                        effectiveProgramInstructionCounts.end(), 0))
                   .with_best_agent_register_size(best_agent_register_size)
                   .with_best_agent_effective_register_size(best_agent_effective_register_size)
                   .with_best_agent_flops(best_agent ? TPG::ComputeTeamFLOPs(best_agent) : 0.0)
                   .with_elite_avg_start_rates(elite_avg_start_rate_swap,
                                               elite_avg_start_rate_delete,
                                               elite_avg_start_rate_add,
                                               elite_avg_start_rate_mutate,
                                               elite_avg_start_rate_decoy)
                   .with_elite_avg_output_rates(elite_avg_output_rate_swap,
                                                elite_avg_output_rate_delete,
                                                elite_avg_output_rate_add,
                                                elite_avg_output_rate_mutate,
                                                elite_avg_output_rate_decoy)
                   .with_operations_use(op_countsTally)
                   // NSGA-II Pareto front metrics
                   .with_pareto_front_0_size(pareto_front_0_size)
                   .with_pareto_front_1_size(pareto_front_1_size)
                   .with_best_agent_pareto_rank(best_agent_pareto_rank)
                   .with_best_agent_crowding_dist(best_agent_crowding_dist)
                   .with_avg_complexity_front_0(avg_complexity_front_0)
                   .with_best_agent_mean_register_similarity(best_agent_mean_register_similarity)
                   .with_best_agent_max_register_similarity(best_agent_max_register_similarity)
                   .with_validation_register_efficiency_pearson(validation_register_efficiency_pearson)
                   .with_validation_register_efficiency_spearman(validation_register_efficiency_spearman)
                   .with_validation_register_efficiency_n(validation_register_efficiency_n);

               SelectionMetrics metrics = builder.build();
               EventDispatcher<SelectionMetrics>::instance().notify(EventType::SELECTION, metrics);
            }

         vector<int> tmSizesRoot, tmSizesSub;
         tmSizesRoot.push_back((*teiter)->size());
         for (auto teiter2 = visitedTeams2.begin();
              teiter2 != visitedTeams2.end(); teiter2++)
            if ((*teiter2)->id_ !=
                (*teiter)->id_)  // not the root of this policy
               tmSizesSub.push_back((*teiter2)->size());
         oss << " mnTmSzR " << VectorMean<int>(tmSizesRoot) << " mnTmSzS "
             << (tmSizesSub.size() > 0 ? VectorMean<int>(tmSizesSub) : 0);

         // Log NSGA-II Pareto front statistics
         {
            int front0_count = 0, front1_count = 0;
            for (auto tm : GetRootTeamsInVec()) {
               if (tm->pareto_rank_ == 0) front0_count++;
               else if (tm->pareto_rank_ == 1) front1_count++;
            }
            team* best = GetBestTeam();
            oss << " pF0 " << front0_count << " pF1 " << front1_count;
            if (best) {
               oss << " bestRank " << best->pareto_rank_ 
                   << " bestCrowd " << std::fixed << std::setprecision(2) << best->crowding_distance_;
            }
         }

         ////map < int, map< int, map <point *, double, pointLexicalLessThan
         ///> > >
         /// outs;
         // map < int, map< int, map <int, point *> > > outs;
         //(*teiter)->outcomes(outs);

         // double maxVisitedTeams = 0;
         // for (auto ouiter = outs.begin(); ouiter != outs.end();
         // ouiter++){//tasks
         //   for (auto ouiter2 = ouiter->second[2].begin(); ouiter2 !=
         //   ouiter->second[2].end(); ouiter2++){//test outcomes, loop over
         //   envSeed
         //      maxVisitedTeams = max(maxVisitedTeams,
         //      ouiter2->second->auxDouble(2));
         //   }
         // }
         // oss << " maxVisitedTeams " << maxVisitedTeams << endl;

         // oss << " outcomes";
         // int nSolved = 0;
         // for(auto ouiter = outs[0][0].begin(); ouiter != outs[0][0].end();
         // ouiter++){
         //    if ((ouiter->first)->auxDouble(0) > 1)
         //       oss << " " << (ouiter->first)->auxDouble(0);
         //    if ((ouiter->first)->auxDouble(0) >= 50)
         //       nSolved++;
         // }
         // oss << " outcomesDesc";
         // for(auto ouiter = outs[0][0].begin(); ouiter != outs[0][0].end();
         // ouiter++){
         //    if ((ouiter->first)->auxDouble(0) >= 50)
         //       oss<< " " << (ouiter->first)->desc().c_str();
         // }
         // oss << " nSol " << nSolved;

         // oss << " fBins";
         // map <long, int> bins = (*teiter)->fitnessBins();
         // for (auto iter = bins.begin(); iter != bins.end(); iter++)
         //    oss << ":" << (*iter).first << "-" << (*iter).second;
         oss << endl;
      }
   }
}

void TPG::trackTeamInfo(long t, int phase, bool singleBest, long teamId) {
   string gen = std::to_string(t);

   team* bestTeam = *(team_pop_.begin());
   if (singleBest && teamId == -1)
      bestTeam = GetBestTeam();
   ostringstream tmposs;
   map<point*, double, pointLexicalLessThan> allOutcomes;
   // map < point *, double > :: iterator myoiter;
   vector<int> behaviourSequence;
   set<team*, teamIdComp> visitedTeams;
   for (auto teiter = team_pop_.begin(); teiter != team_pop_.end(); teiter++) {
      if ((!singleBest && (*teiter)->root() &&
           teamId == -1) ||                             // all root teams
          (!singleBest && (*teiter)->id_ == teamId) ||  // specific team
          (singleBest &&
           (*teiter)->id_ == bestTeam->id_))  // singleBest root team
      {
         api_client_->LogMetric("teamInfo/root",
                                ((*teiter)->root() ? "1" : "0"), "", gen);
         api_client_->LogMetric("teamInfo/sz",
                                std::to_string((*teiter)->size()), "", gen);
         api_client_->LogMetric("teamInfo/age",
                                std::to_string(t - (*teiter)->gtime_), "", gen);
         api_client_->LogMetric(
             "teamInfo/nOutTrain",
             std::to_string((*teiter)->numOutcomes(_TRAIN_PHASE, -1)), "", gen);
         api_client_->LogMetric(
             "teamInfo/nOutTest",
             std::to_string((*teiter)->numOutcomes(_TEST_PHASE, -1)), "", gen);
         api_client_->LogMetric("teamInfo/fit", std::to_string((*teiter)->fit_),
                                "", gen);

         visitedTeams.clear();
         vector<int> programInstructionCounts,
             effectiveProgramInstructionCounts;
         (*teiter)->policyInstructions(team_map_, visitedTeams,
                                       programInstructionCounts,
                                       effectiveProgramInstructionCounts);

         int pIns = accumulate(programInstructionCounts.begin(),
                               programInstructionCounts.end(), 0);
         double mnProgIns = VectorMean<int>(programInstructionCounts);
         int ePIns = accumulate(effectiveProgramInstructionCounts.begin(),
                                effectiveProgramInstructionCounts.end(), 0);
         double mnEProgIns = VectorMean<int>(effectiveProgramInstructionCounts);
         api_client_->LogMetric("teamInfo/pIns", std::to_string(pIns), "", gen);
         api_client_->LogMetric("teamInfo/mnProgIns", std::to_string(mnProgIns),
                                "", gen);
         api_client_->LogMetric("teamInfo/ePIns", std::to_string(ePIns), "",
                                gen);
         api_client_->LogMetric("teamInfo/mnEProgIns",
                                std::to_string(mnEProgIns), "", gen);

         set<RegisterMachine*, RegisterMachineIdComp> programs;
         // set<MemoryEigen *, MemoryEigenIdComp> memories;
         set<team*, teamIdComp> visitedTeams2;
         (*teiter)->GetAllNodes(team_map_, visitedTeams2, programs);
         api_client_->LogMetric("teamInfo/nP", std::to_string(programs.size()),
                                "", gen);
         api_client_->LogMetric("teamInfo/nT",
                                std::to_string(visitedTeams2.size()), "", gen);
         // api_client_->LogMetric("teamInfo/nM",
         // std::to_string(memories.size()), "",
         //                        gen);

         vector<int> op_countsSingle;
         vector<int> op_countsTally;
         op_countsTally.resize(instruction::NUM_OP);

         fill(op_countsTally.begin(), op_countsTally.end(), 0);
         for (auto it = programs.begin(); it != programs.end(); it++) {
            op_countsSingle = (*it)->op_counts_;
            for (size_t i = 0; i < op_countsSingle.size(); i++)
               op_countsTally[i] += op_countsSingle[i];
         }

         api_client_->LogMetric("teamInfo/nOp", VectorToString(op_countsTally), "",
                                gen);

         vector<int> tmSizesRoot, tmSizesSub;
         tmSizesRoot.push_back((*teiter)->size());
         for (auto teiter2 = visitedTeams2.begin();
              teiter2 != visitedTeams2.end(); teiter2++)
            if ((*teiter2)->id_ !=
                (*teiter)->id_)  // not the root of this policy
               tmSizesSub.push_back((*teiter2)->size());

         double mnTmSzR = VectorMean<int>(tmSizesRoot);
         int mnTmSzS =
             (tmSizesSub.size() > 0 ? VectorMean<int>(tmSizesSub) : 0);
         api_client_->LogMetric("teamInfo/mnTmSzR", std::to_string(mnTmSzR), "",
                                gen);
         api_client_->LogMetric("teamInfo/mnTmSzS", std::to_string(mnTmSzS), "",
                                gen);
      }
   }
}

/******************************************************************************/
void TPG::RegisterMachineCrossover(RegisterMachine* p1, RegisterMachine* p2,
                                   RegisterMachine** c1, RegisterMachine** c2) {
   int split1, split2;

   // Split parent 1 (p1) into 3 random chunks
   std::vector<std::vector<instruction*>> p1_chunks(3);
   std::uniform_int_distribution<> dis1(0, p1->instructions_.size() - 1);
   split1 = dis1(rngs_[TPG_SEED]);
   split2 = dis1(rngs_[TPG_SEED]);
   while (split2 == split1) {
      split2 = dis1(rngs_[TPG_SEED]);
   }
   if (split1 > split2) {
      std::swap(split1, split2);
   }
   p1_chunks[0] = std::vector<instruction*>(p1->instructions_.begin(),
                                            p1->instructions_.begin() + split1);
   p1_chunks[1] = std::vector<instruction*>(p1->instructions_.begin() + split1,
                                            p1->instructions_.begin() + split2);
   p1_chunks[2] = std::vector<instruction*>(p1->instructions_.begin() + split2,
                                            p1->instructions_.end());

   // Split parent 2 (p2) into 3 random chunks
   std::vector<std::vector<instruction*>> p2_chunks(3);
   std::uniform_int_distribution<> dis2(0, p2->instructions_.size() - 1);
   split1 = dis2(rngs_[TPG_SEED]);
   split2 = dis2(rngs_[TPG_SEED]);
   while (split2 == split1) {
      split2 = dis2(rngs_[TPG_SEED]);
   }
   if (split1 > split2) {
      std::swap(split1, split2);
   }
   p2_chunks[0] = std::vector<instruction*>(p2->instructions_.begin(),
                                            p2->instructions_.begin() + split1);
   p2_chunks[1] = std::vector<instruction*>(p2->instructions_.begin() + split1,
                                            p2->instructions_.begin() + split2);
   p2_chunks[2] = std::vector<instruction*>(p2->instructions_.begin() + split2,
                                            p2->instructions_.end());

   // Cretae child 1 (c1) from parent chunks {p1-0, p2-1, p1-2}
   std::vector<instruction*> c1_instructions;
   c1_instructions.insert(c1_instructions.end(), p1_chunks[0].begin(),
                          p1_chunks[0].end());
   c1_instructions.insert(c1_instructions.end(), p2_chunks[1].begin(),
                          p2_chunks[1].end());
   c1_instructions.insert(c1_instructions.end(), p1_chunks[2].begin(),
                          p1_chunks[2].end());
   *c1 = new RegisterMachine(p1->action_, c1_instructions, params_, state_,
                             rngs_[TPG_SEED], _ops, p1->n_memories_);

   // Cretae child 2 (c2) from parent chunks {p2-0, p1-1, p2-2}
   std::vector<instruction*> c2_instructions;
   c2_instructions.insert(c2_instructions.end(), p2_chunks[0].begin(),
                          p2_chunks[0].end());
   c2_instructions.insert(c2_instructions.end(), p1_chunks[1].begin(),
                          p1_chunks[1].end());
   c2_instructions.insert(c2_instructions.end(), p2_chunks[2].begin(),
                          p2_chunks[2].end());
   *c2 = new RegisterMachine(p2->action_, c2_instructions, params_, state_,
                             rngs_[TPG_SEED], _ops, p2->n_memories_);
}

void TPG::LinearCrossover(RegisterMachine* gp1,
                          RegisterMachine* gp2,
                          RegisterMachine** c1,
                          RegisterMachine** c2) {
    // =========================================================
    // Linear Register Crossover (Sorted Index Method)
    // =========================================================
    // 1. Identify all unique destination registers in both parents.
    // 2. Sort them by Index (e.g., R0, R1, R2...).
    // 3. Cut both parents at a proportional point in their sorted lists.
    // 4. Merge: Child 1 takes Lower registers from P1 + Upper from P2.
    //           Child 2 takes Lower registers from P2 + Upper from P1.

    // 1. Define Module Key (Type + Index)
    struct ModuleKey {
        int out_type;
        int out_idx;

        // Strict weak ordering: Sort by Type, then by Index
        bool operator<(const ModuleKey& o) const {
            if (out_type != o.out_type) return out_type < o.out_type;
            return out_idx < o.out_idx;
        }
        // Equality check for map/set
        bool operator==(const ModuleKey& o) const {
            return out_type == o.out_type && out_idx == o.out_idx;
        }
    };

    // Helper: Extract key from instruction
    auto get_key = [](instruction* instr, ModuleKey& k_out) -> bool {
        if (!instr || instr->outIdx_ < 0) return false;
        k_out = { static_cast<int>(instr->GetOutType()), instr->outIdx_ };
        return true;
    };

    // Helper: Get unique, naturally sorted list of registers (R0, R1, R2...)
    auto get_sorted_modules = [&](RegisterMachine* p) -> std::vector<ModuleKey> {
        std::set<ModuleKey> unique_mods;
        ModuleKey k;
        for (auto* instr : p->instructions_) {
            if (get_key(instr, k)) {
                unique_mods.insert(k);
            }
        }
        // Copy to vector (already sorted by std::set)
        return std::vector<ModuleKey>(unique_mods.begin(), unique_mods.end());
    };

    const std::vector<ModuleKey> mods1 = get_sorted_modules(gp1);
    const std::vector<ModuleKey> mods2 = get_sorted_modules(gp2);

    // Fallback: Clone if too small to cut
    if (mods1.size() < 2 || mods2.size() < 2) {
        *c1 = new RegisterMachine(gp1->action_, gp1->instructions_, params_, state_, rngs_[TPG_SEED], _ops, gp1->n_memories_);
        *c2 = new RegisterMachine(gp2->action_, gp2->instructions_, params_, state_, rngs_[TPG_SEED], _ops, gp2->n_memories_);
        return;
    }

    // 2. Calculate Cut Points
    // We pick a cut relative to the smallest parent to ensure validity.
    const std::size_t min_sz = std::min(mods1.size(), mods2.size());
    std::uniform_int_distribution<int> disCut(1, static_cast<int>(min_sz) - 1);
    
    const int raw_cut = disCut(rngs_[TPG_SEED]); 
    const double cut_fraction = static_cast<double>(raw_cut) / static_cast<double>(min_sz);

    // Project cut fraction onto both parents
    auto get_cut_idx = [cut_fraction](size_t size) -> int {
        int cut = static_cast<int>(std::lround(cut_fraction * static_cast<double>(size)));
        return std::clamp(cut, 1, static_cast<int>(size) - 1);
    };

    const int cut1 = get_cut_idx(mods1.size());
    const int cut2 = get_cut_idx(mods2.size());

    // 3. Build Ownership Maps (Map Module -> Parent ID 0 or 1)
    using OwnerMap = std::map<ModuleKey, int>;

    auto build_ownership = [&](const std::vector<ModuleKey>& lower_list,
                               const std::vector<ModuleKey>& upper_list,
                               int lower_pid, int upper_pid) -> OwnerMap {
        OwnerMap owner;
        // Lower block (e.g., R0...Rk) comes from lower_pid
        for (const auto& k : lower_list) owner[k] = lower_pid;
        
        // Upper block (e.g., Rk+1...Rn) comes from upper_pid
        // Conflict resolution: If a register exists in both lists (unlikely in sorted split, 
        // but safe to check), prefer the Lower parent.
        for (const auto& k : upper_list) {
            if (owner.find(k) == owner.end()) {
                owner[k] = upper_pid;
            }
        }
        return owner;
    };

    // Split lists
    std::vector<ModuleKey> p1_lower(mods1.begin(), mods1.begin() + cut1);
    std::vector<ModuleKey> p1_upper(mods1.begin() + cut1, mods1.end());
    std::vector<ModuleKey> p2_lower(mods2.begin(), mods2.begin() + cut2);
    std::vector<ModuleKey> p2_upper(mods2.begin() + cut2, mods2.end());

    // Child 1: P1 Lower + P2 Upper
    OwnerMap owner_c1 = build_ownership(p1_lower, p2_upper, 0, 1);
    // Child 2: P2 Lower + P1 Upper
    OwnerMap owner_c2 = build_ownership(p2_lower, p1_upper, 1, 0);

    // 4. Construct Child Instructions (Zipper Merge)
    auto build_prog = [&](RegisterMachine* p1, RegisterMachine* p2,
                          const OwnerMap& ownership,
                          std::vector<instruction*>& out_prog) {
        out_prog.clear();
        size_t n1 = p1->instructions_.size();
        size_t n2 = p2->instructions_.size();
        size_t i = 0, j = 0;

        auto norm = [](size_t idx, size_t total) {
            return (total <= 1) ? 0.0 : static_cast<double>(idx) / (total - 1);
        };

        ModuleKey k;
        while (i < n1 || j < n2) {
            // Normalized positions for merge sort
            double t1 = (i < n1) ? norm(i, n1) : 1e100;
            double t2 = (j < n2) ? norm(j, n2) : 1e100;

            bool take_p1 = (t1 <= t2);
            instruction* instr = take_p1 ? p1->instructions_[i++] : p2->instructions_[j++];
            int origin_pid = take_p1 ? 0 : 1;

            if (get_key(instr, k)) {
                auto it = ownership.find(k);
                // Keep instruction ONLY if its destination is owned by the parent it came from
                if (it != ownership.end() && it->second == origin_pid) {
                    out_prog.push_back(instr);
                }
            }
        }
    };

    std::vector<instruction*> c1_instrs, c2_instrs;
    build_prog(gp1, gp2, owner_c1, c1_instrs);
    build_prog(gp1, gp2, owner_c2, c2_instrs);

    *c1 = new RegisterMachine(gp1->action_, c1_instrs, params_, state_, rngs_[TPG_SEED], _ops, gp1->n_memories_);
    *c2 = new RegisterMachine(gp2->action_, c2_instrs, params_, state_, rngs_[TPG_SEED], _ops, gp2->n_memories_);
}

// void TPG::LinearCrossover(RegisterMachine* gp1,
//                           RegisterMachine* gp2,
//                           RegisterMachine** c1,
//                           RegisterMachine** c2) {
//     // =========================================================
//     // Linear Register Crossover (Per-Register 50/50)
//     // =========================================================
//     // For each destination register index up to the maximum index used
//     // across BOTH parents (per register type), randomly decide which
//     // parent "owns" that register in each child (50/50).
//     //
//     // Edge case: if a register is only present in one parent (extra
//     // register in the longer parent), keep it in a child with a
//     // "biological" retention probability (default 0.5), otherwise drop it.

//     // 1) Define Module Key (Type + Index)
//     struct ModuleKey {
//         int out_type;
//         int out_idx;

//         bool operator<(const ModuleKey& o) const {
//             if (out_type != o.out_type) return out_type < o.out_type;
//             return out_idx < o.out_idx;
//         }
//         bool operator==(const ModuleKey& o) const {
//             return out_type == o.out_type && out_idx == o.out_idx;
//         }
//     };

//     // Helper: Extract key from instruction
//     auto get_key = [](instruction* instr, ModuleKey& k_out) -> bool {
//         if (!instr || instr->outIdx_ < 0) return false;
//         k_out = { static_cast<int>(instr->GetOutType()), instr->outIdx_ };
//         return true;
//     };

//     // 2) Gather which destination registers are actually used in each parent
//     auto get_used_set = [&](RegisterMachine* p) -> std::set<ModuleKey> {
//         std::set<ModuleKey> used;
//         ModuleKey k;
//         for (auto* instr : p->instructions_) {
//             if (get_key(instr, k)) used.insert(k);
//         }
//         return used;
//     };

//     const std::set<ModuleKey> used1 = get_used_set(gp1);
//     const std::set<ModuleKey> used2 = get_used_set(gp2);

//     // If neither parent uses any destination register (unlikely), clone.
//     if (used1.empty() && used2.empty()) {
//         *c1 = new RegisterMachine(gp1->action_, gp1->instructions_, params_, state_, rngs_[TPG_SEED], _ops);
//         *c2 = new RegisterMachine(gp2->action_, gp2->instructions_, params_, state_, rngs_[TPG_SEED], _ops);
//         return;
//     }

//     // 3) Compute max index per type across both parents
//     std::map<int, int> max_idx_per_type;
//     for (const auto& k : used1) {
//         auto it = max_idx_per_type.find(k.out_type);
//         if (it == max_idx_per_type.end()) max_idx_per_type[k.out_type] = k.out_idx;
//         else it->second = std::max(it->second, k.out_idx);
//     }
//     for (const auto& k : used2) {
//         auto it = max_idx_per_type.find(k.out_type);
//         if (it == max_idx_per_type.end()) max_idx_per_type[k.out_type] = k.out_idx;
//         else it->second = std::max(it->second, k.out_idx);
//     }

//     // 4) Build ownership maps (Module -> parent id) for each child
//     //    ownership[k] == 0 => keep only if instruction comes from gp1
//     //    ownership[k] == 1 => keep only if instruction comes from gp2
//     //    ownership[k] == -1 => dropped (no parent owns it in that child)
//     using OwnerMap = std::map<ModuleKey, int>;
//     OwnerMap owner_c1;
//     OwnerMap owner_c2;

//     std::uniform_real_distribution<double> dis01(0.0, 1.0);

//     // "Biological" retention probability for registers that only exist
//     // in one parent (extra registers from the longer parent).
//     // If you want to tune later, wire this to params_.
//     const double p_keep_extra = 0.5;

//     for (const auto& pr : max_idx_per_type) {
//         const int out_type = pr.first;
//         const int max_idx  = pr.second;

//         for (int idx = 0; idx <= max_idx; ++idx) {
//             ModuleKey k{out_type, idx};
//             const bool in1 = (used1.find(k) != used1.end());
//             const bool in2 = (used2.find(k) != used2.end());

//             if (in1 && in2) {
//                 // 50/50 assignment, complementary across children
//                 const bool pick_p1 = (dis01(rngs_[TPG_SEED]) < 0.5);
//                 owner_c1[k] = pick_p1 ? 0 : 1;
//                 owner_c2[k] = pick_p1 ? 1 : 0;
//             } else if (in1 || in2) {
//                 // Only present in one parent: keep with retention probability
//                 const int pid = in1 ? 0 : 1;

//                 owner_c1[k] = (dis01(rngs_[TPG_SEED]) < p_keep_extra) ? pid : -1;
//                 owner_c2[k] = (dis01(rngs_[TPG_SEED]) < p_keep_extra) ? pid : -1;
//             }
//             // else: present in neither parent (within range) => ignore
//         }
//     }

//     // 5) Construct child instructions using a normalized zipper-merge.
//     //    Keep an instruction only if its destination register is owned by
//     //    the parent it came from in that child.
//     auto build_prog = [&](RegisterMachine* p1, RegisterMachine* p2,
//                           const OwnerMap& ownership,
//                           std::vector<instruction*>& out_prog) {
//         out_prog.clear();
//         const size_t n1 = p1->instructions_.size();
//         const size_t n2 = p2->instructions_.size();
//         size_t i = 0, j = 0;

//         auto norm = [](size_t idx, size_t total) {
//             return (total <= 1) ? 0.0 : static_cast<double>(idx) / (total - 1);
//         };

//         ModuleKey k;
//         while (i < n1 || j < n2) {
//             const double t1 = (i < n1) ? norm(i, n1) : 1e100;
//             const double t2 = (j < n2) ? norm(j, n2) : 1e100;

//             const bool take_p1 = (t1 <= t2);
//             instruction* instr = take_p1 ? p1->instructions_[i++] : p2->instructions_[j++];
//             const int origin_pid = take_p1 ? 0 : 1;

//             if (get_key(instr, k)) {
//                 auto it = ownership.find(k);
//                 if (it != ownership.end() && it->second >= 0 && it->second == origin_pid) {
//                     out_prog.push_back(instr);
//                 }
//             }
//         }
//     };

//     std::vector<instruction*> c1_instrs, c2_instrs;
//     build_prog(gp1, gp2, owner_c1, c1_instrs);
//     build_prog(gp1, gp2, owner_c2, c2_instrs);

//     *c1 = new RegisterMachine(gp1->action_, c1_instrs, params_, state_, rngs_[TPG_SEED], _ops);
//     *c2 = new RegisterMachine(gp2->action_, c2_instrs, params_, state_, rngs_[TPG_SEED], _ops);
// }


/******************************************************************************/
// Read in populations from a checkpoint file.
// TODO(skelly): move reading logic to "deserialize" constructors and cleanup
void TPG::ReadCheckpoint(long t, int phase, bool fromString,
                         const string& inString) {
   finalize();  // clear populations
   string str;

   if (fromString) {
      str = inString;
   } else {
      char filename[80];
      sprintf(filename, "%s.%ld.%lu.%d.rslt", "checkpoints/cp",
           t, seeds_[TPG_SEED], phase);
      ifstream t(filename);
      if (!t.is_open() || !t.good()) {
         std::cerr << "ReadCheckpoint: cannot open file: " << filename
                   << " (check that checkpoint exists and path is correct)"
                   << std::endl;
         die(__FILE__, __FUNCTION__, __LINE__, "Checkpoint file not found.");
      }
      t.seekg(0, ios::end);
      auto size = t.tellg();
      t.seekg(0, ios::beg);
      if (size <= 0) {
         std::cerr << "ReadCheckpoint: file is empty or invalid: " << filename
                   << std::endl;
         die(__FILE__, __FUNCTION__, __LINE__, "Checkpoint file is empty.");
      }
      str.reserve(static_cast<size_t>(size));
      str.assign((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
   }

   istringstream iss(str);
   string oneline;
   long memberId = 0;
   long max_teamCount = -1;
   long max_programCount = -1;
   int f;

   while (getline(iss, oneline)) {
      if (oneline.size() == 0)
         continue;
      auto outcome_fields = SplitString(oneline, ':');
      // if (outcome_fields[0] == "weights") {
      //    long teamId = atol(outcome_fields[1].c_str());
      //    long prevProgID = atol(outcome_fields[2].c_str());
      //    long currProgID = atol(outcome_fields[3].c_str());
      //    double weightValue = atof(outcome_fields[4].c_str());
         
      //    team* targetTeam = _teamMap[teamId];
      //    if (targetTeam) {
      //       if (targetTeam->HebbianMap.weights.find(prevProgID) == targetTeam->HebbianMap.weights.end()) {
      //             targetTeam->HebbianMap.weights[prevProgID] = std::map<long, double>();
      //       }
      //       targetTeam->HebbianMap.weights[prevProgID][currProgID] = weightValue;
      //    }
      // }
      if (outcome_fields[0].compare("t") == 0)
         state_["t_current"] = atoi(outcome_fields[1].c_str());
      else if (outcome_fields[0].compare("active_task") == 0)
         state_["active_task"] = atoi(outcome_fields[1].c_str());
      else if (outcome_fields[0].compare("fitMode") == 0)
         params_["fit_mode"] = atoi(outcome_fields[1].c_str());
      else if (outcome_fields[0].compare("phase") == 0)
         state_["phase"] = atoi(outcome_fields[1].c_str());
      else if (outcome_fields[0].compare("MemoryEigen") == 0) {
         // max_memoryCount =
         //  std::max(max_memoryCount, atol(outcome_fields[1].c_str()));
         long prog_id = atol(outcome_fields[1].c_str());
         AddMemory(prog_id, new MemoryEigen(outcome_fields));
      } else if (outcome_fields[0].compare("RegisterMachine") == 0) {
         max_programCount =
             std::max(max_programCount, atol(outcome_fields[1].c_str()));
         AddProgram(new RegisterMachine(outcome_fields, _Memory, params_,
                                        rngs_[TPG_SEED]));
      } else if (outcome_fields[0].compare("team") == 0) {
         team* m;
         f = 1;
         long id = atoi(outcome_fields[f++].c_str());
         if (id > max_teamCount)
            max_teamCount = id;
         long gtime = atoi(outcome_fields[f++].c_str());
         int obs_index = atoi(outcome_fields[f++].c_str());
         double lambda_td = atof(outcome_fields[f++].c_str());
         double decay_factor = atof(outcome_fields[f++].c_str());
         m = new team(gtime, id, obs_index, lambda_td, decay_factor);
         m->_n_eval = atoi(outcome_fields[f++].c_str());
         // add programs in order
         for (size_t ii = f; ii < outcome_fields.size(); ii++) {
            memberId = atoi(outcome_fields[ii].c_str());
            m->AddProgram(program_pop_[memberId]);
            // nrefs_ is loaded from checkpoint, so dec here TODO(skelly):fix
            program_pop_[memberId]->nrefs_--;
         }
         AddTeam(m);
      } else if (outcome_fields[0].compare("incoming_progs") == 0) {
         f = 1;
         long id = atoi(outcome_fields[f++].c_str());
         for (size_t ii = f; ii < outcome_fields.size(); ii++) {
            long incomingId = atoi(outcome_fields[ii].c_str());
            team_map_[id]->AddIncomingProgram(incomingId);
         }
      } else if (!fromString && outcome_fields[0].compare("phyloNode") == 0 &&
                 find(outcome_fields.begin(), outcome_fields.end(), "gtime") ==
                     outcome_fields.end()) {
         f = 1;
         long id = atoi(outcome_fields[f++].c_str());
         phylo_graph_.insert(pair<long, phyloRecord>(id, phyloRecord()));
         phylo_graph_[id].gtime = atoi(outcome_fields[f++].c_str());
         phylo_graph_[id].dtime = atoi(outcome_fields[f++].c_str());
         phylo_graph_[id].fitnessBin = atoi(outcome_fields[f++].c_str());
         phylo_graph_[id].fitness = atof(outcome_fields[f++].c_str());
         phylo_graph_[id].root =
             atoi(outcome_fields[f++].c_str()) > 0 ? true : false;
      } else if (!fromString && outcome_fields[0].compare("phyloLink") == 0 &&
                 find(outcome_fields.begin(), outcome_fields.end(), "from,to") ==
                     outcome_fields.end()) {
         phylo_graph_[atoi(outcome_fields[1].c_str())].adj.push_back(
             atoi(outcome_fields[2].c_str()));
                     }
      else if (!fromString && outcome_fields[0].compare("ancestorIds") == 0) {
         f = 1;
         long id = atoi(outcome_fields[f++].c_str());
         for (size_t ii = f; ii < outcome_fields.size(); ii++) {
            long aid = atoi(outcome_fields[ii].c_str());
            phylo_graph_[id].ancestorIds.insert(aid);
         }
      }
   }
   state_["program_count"] = max_programCount + 1;
   state_["team_count"] = max_teamCount + 1;
   // state_["memory_count"] = max_memoryCount + 1;
}

/******************************************************************************/
void TPG::recalculateProgramRefs() {
   for (auto p : program_pop_)
      p.second->nrefs_ = 0;
   for (auto tm : team_pop_)
      for (auto p : tm->members_)
         p->nrefs_++;
}


/******************************************************************************/
void TPG::SanityCheck() {
   TeamSizesMatchProgRefs();
}

/******************************************************************************/
void TPG::TeamSizesMatchProgRefs() {
   int sum_team_sizes = 0;
   int sum_prog_refs = 0;
   for (auto tm : team_pop_)
      sum_team_sizes += tm->size();
   for (auto prog : program_pop_)
      sum_prog_refs += prog.second->nrefs_;
   if (sum_prog_refs != sum_team_sizes) {
      std::string error_message =
          "Program reference mismatch. sum_team_sizes " +
          to_string(sum_team_sizes) + " sum_prog_refs " +
          to_string(sum_prog_refs);
      die(__FILE__, __FUNCTION__, __LINE__, error_message.c_str());
   }
}

/******************************************************************************/
// TODO(skelly): survivor selection
void TPG::SelectTeams() {

   int n_old_deleted = 0;
   int n_deleted = 0;
   int n_root_remaining = 0;

   for (auto tm : GetRootTeamsInVec()) {
      if (!tm->elite(GetState("phase")) && !isElitePS(tm, GetState("phase"))) {
         phylo_graph_[tm->id_].dtime = GetState("t_current");
         n_deleted++;
         n_old_deleted += tm->gtime_ < GetState("t_current") ? 1 : 0;
         RemoveTeam(tm);
      }
      n_root_remaining++;
   }

   CleanupProgramsWithNoRefs();

   oss << "selTms t " << GetState("t_current") << " Msz " << team_pop_.size()
       << " Lsz " << program_pop_.size() << " mrSz " << n_root_remaining
       << " mSz";
   for (size_t mem_t = 0; mem_t < MemoryEigen::kNumMemoryType_; mem_t++) {
      oss << " " << _Memory[mem_t].size();
   }
   oss << " eLSz " << _numEliteTeamsCurrent[GetState("phase")] << " nDel "
       << n_deleted << " nOldDel " << n_old_deleted << " nOldDelPr "
       << (double)n_old_deleted / n_deleted;
   oss << endl;

   RemovalMetricsBuilder builder;
   builder.with_generation(GetState("t_current"))
      .with_num_teams(team_pop_.size())
      .with_num_programs(program_pop_.size())
      .with_num_root_programs(n_root_remaining)
      .with_num_elite_teams(_numEliteTeamsCurrent[GetState("phase")])
      .with_num_deleted(n_deleted)
      .with_num_old_deleted(n_old_deleted)
      .with_percent_old_deleted((double) n_old_deleted / n_deleted);

   RemovalMetrics metrics = builder.build();
   EventDispatcher<RemovalMetrics>::instance().notify(EventType::REMOVAL, metrics);

}

/******************************************************************************/
void TPG::CleanupProgramsWithNoRefs() {
   vector<long> deletedIds;
   for (auto it = program_pop_.begin(); it != program_pop_.end();) {
      auto prog = it->second;
      if (prog->nrefs_ < 1) {
         if (prog->action_ >= 0) {
            if (team_map_[prog->action_]->inDeg() == 1) {
               phylo_graph_[team_map_[prog->action_]->id_].root = true;
            }
            team_map_[prog->action_]->removeIncomingProgram(prog->id_);
            // If team was a subsumed root clone that has now become a root
            // itself, just delete it
            if (team_map_[prog->action_]->root() &&
                team_map_[prog->action_]->cloneId_ != -1) {
               auto it = team_map_.find(team_map_[prog->action_]->cloneId_);
               if (it != team_map_.end())
                  it->second->clones_ = it->second->clones_ - 1;
               team* tm = team_map_[prog->action_];
               phylo_graph_[tm->id_].dtime = GetState("t_current");
               RemoveTeam(tm);
            }
         }
         it = program_pop_.erase(it);
         delete prog;
      } else {
         it++;
      }
   }
}

// /******************************************************************************/
// void TPG::teamTaskRank(int phase, const vector<int> &objectives) {
//    oss << "TPG::teamTaskRank <team:avgRank>";
//    for (auto teiterA = team_pop_.begin(); teiterA != team_pop_.end(); teiterA++) {
//       if (!(*teiterA)->root()) continue;
//       vector<int> ranks;
//       for (size_t i = 0; i < objectives.size(); i++) ranks.push_back(1);
//       for (size_t o = 0; o < objectives.size(); o++) {
//          for (auto teiterB = team_pop_.begin(); teiterB != team_pop_.end(); teiterB++) {
//             if (!(*teiterB)->root()) continue;
//             if ((*teiterB)->getMeanOutcome(phase, objectives[o],
//                                            GetParam<int>("fit_mode"), false,
//                                            false) >
//                 (*teiterA)->getMeanOutcome(phase, objectives[o],
//                                            GetParam<int>("fit_mode"), false,
//                                            false))
//                ranks[o]++;
//          }
//       }

//       double rankSum = 0;
//       for (size_t i = 0; i < ranks.size(); i++) rankSum += ranks[i];
//       (*teiterA)->fit_ = 1 / (rankSum / ranks.size());
//       oss << " " << (*teiterA)->id_ << ":" << (*teiterA)->fit_;
//    }
//    oss << endl;
// }

/******************************************************************************/
void TPG::updateMODESFilters(bool roots) {
   (void)roots;
   vector<long> symbiontIntersection;
   symbiontIntersection.reserve(100);
   vector<long> symbiontUnion;
   symbiontUnion.reserve(100);

   if (GetState("t_current") != GetState("t_start")) {
      _persistenceFilterA.clear();
      for (auto tm : GetRootTeamsInVec()) {
         vector<long> ancestorIds;
         tm->getAncestorIds(ancestorIds);
         for (auto pr = _allComponentsA.begin(); pr != _allComponentsA.end();
              pr++)
            if (find(ancestorIds.begin(), ancestorIds.end(), (*pr).first) !=
                    ancestorIds.end() &&
                tm->hasOutcome(0, _TRAIN_PHASE, 0)) {
               // found an ancestor of *teiter in _allComponentsA, add
               // *teiter to _persistenceFilterA
               _persistenceFilterA.insert(
                   pair<long, modesRecord>(tm->id_, modesRecord()));
               // store active programs for novelty metric
               set<team*, teamIdComp> teams;
               set<RegisterMachine*, RegisterMachineIdComp> programs;
               tm->GetAllNodes(team_map_, teams, programs);
               for (auto leiter = programs.begin(); leiter != programs.end();
                    leiter++) {
                  _persistenceFilterA[tm->id_].activeProgramIds.insert(
                      (*leiter)->id_);
                  _persistenceFilterA[tm->id_].effectiveInstructionsTotal +=
                      static_cast<int>(
                          (*leiter)->instructions_effective_.size());
               }
               for (auto tm2 : teams)
                  _persistenceFilterA[tm->id_].activeTeamIds.insert(tm2->id_);
               _persistenceFilterA[tm->id_].runTimeComplexityIns =
                   tm->runTimeComplexityIns();
               _persistenceFilterA[tm->id_].behaviourString =
                   tm->getBehaviourString(0, _TRAIN_PHASE);
               break;
            }
      }

      // do analysis here
      int change = 0;
      // double minNCD = 0;
      double novelty = 0;
      int complexityRTC = 0;
      int complexityTeams = 0;
      int complexityPrograms = 0;
      int complexityInstructions = 0;
      double ecology = 0;
      for (auto prA = _persistenceFilterA.begin();
           prA != _persistenceFilterA.end(); prA++) {
         // change
         if (_persistenceFilterA_t.find((*prA).first) ==
             _persistenceFilterA_t.end())
            change++;
         // novelty
         bool found = false;
         for (auto prAll = _persistenceFilterAllTime.begin();
              prAll != _persistenceFilterAllTime.end(); prAll++) {
            // double ncd =
            // normalizedCompressionDistance((*prAll).second.behaviourString,
            // (*prA).second.behaviourString);
            if ((*prAll).second.behaviourString.compare(
                    (*prA).second.behaviourString) == 0) {
               found = true;
               break;
            }
         }
         if (!found)
            novelty++;

         // complexity
         complexityRTC =
             max(complexityRTC, (int)(*prA).second.runTimeComplexityIns);
         complexityTeams =
             max(complexityTeams,
                 (int)(*prA).second.activeTeamIds.size());  // transitions ?
         complexityPrograms = max(complexityPrograms,
                                  (int)(*prA).second.activeProgramIds.size());
         complexityInstructions =
             max(complexityInstructions,
                 (int)(*prA).second.effectiveInstructionsTotal);

         // ecology
         double Pc = 0;
         for (auto prA2 = _persistenceFilterA.begin();
              prA2 != _persistenceFilterA.end(); prA2++)
            if ((*prA2).first != (*prA).first &&
                (*prA2).second.behaviourString.compare(
                    (*prA).second.behaviourString) == 0)
               Pc++;
         if (Pc > 0) {
            Pc = Pc / _persistenceFilterA.size();
            ecology += Pc * log2(Pc);
         }
      }
      ecology *= -1;

      oss << "TPG::MODES t " << GetState("t_current") << " pfASize "
          << _persistenceFilterA.size() << " pfA_tSize "
          << _persistenceFilterA_t.size();
      oss << " change " << change << " novelty " << novelty;
      oss << " complexityRTC " << complexityRTC << " complexityTeams "
          << complexityTeams << " complexityPrograms " << complexityPrograms;
      oss << " complexityInstructions " << complexityInstructions << " ecology "
          << ecology << endl;

      _persistenceFilterA_t.clear();
      _persistenceFilterA_t.insert(_persistenceFilterA.begin(),
                                   _persistenceFilterA.end());
      _persistenceFilterAllTime.insert(_persistenceFilterA.begin(),
                                       _persistenceFilterA.end());
   }

   // for next t
   _allComponentsA.clear();
   for (auto tm : GetRootTeamsInVec()) {
      _allComponentsA.insert(pair<long, modesRecord>(tm->id_, modesRecord()));
   }
}

/******************************************************************************/
void TPG::WriteCheckpoint(bool elite) {
   ofstream ofs;
   auto filename = "checkpoints/cp." + to_string(GetState("t_current")) + "." +
                   to_string(seeds_[TPG_SEED]) + "." +
                   to_string(GetState("phase")) + ".rslt";
   ofs.open(filename, ios::out);
   if (!ofs.is_open() || ofs.fail()) {
      std::cerr << "open failed for file: " << filename
           << " error:" << strerror(errno) << '\n';
      die(__FILE__, __FUNCTION__, __LINE__, "Can't open checkpoint file.");
   }
   ofs << "seed_tpg:" << seeds_[TPG_SEED] << endl;
   ofs << "seed_aux:" << seeds_[AUX_SEED] << endl;
   ofs << "t:" << GetState("t_current") << endl;
   ofs << "active_task:" << GetState("active_task") << endl;
   ofs << "fitMode:" << GetParam<int>("fit_mode") << endl;

   if (elite) {  // Include data for elite teams only
      set<MemoryEigen*> memories;
      set<RegisterMachine*, RegisterMachineIdComp> programs;
      set<team*, teamIdComp> teams, teamsAll;
      // Collect memories, progrms, teams in champions...
      for (auto ps : _eliteTeamPS) {  // ...for each task set
         for (auto fm : ps.second) {  // ...for each fitness mode
            auto tm = fm.second[GetState("phase")];
            teams.clear();
            tm->GetAllNodes(team_map_, teams, programs, memories);
            teamsAll.insert(teams.begin(), teams.end());
         }
      }
      for (auto prog : programs) {
         ofs << prog->ToStringMemory();
      }
      for (auto prog : programs) {
         ofs << prog->ToString(GetParam<int>("skip_introns"));
      }
      for (auto tm : teamsAll) {
         ofs << tm->ToString();
      }
   } else {  // Include all memories, teams, and programs
      for (auto key : program_pop_) {
         ofs << key.second->ToStringMemory();
      }
      for (auto key : program_pop_) {
         ofs << key.second->ToString(false);  // Write all instructions
      }
      for (auto tm : team_pop_) {
         ofs << tm->ToString();
      }
      ofs << PhylogenyToString();
   }
   ofs << "end" << endl;
   ofs.close();

   // Remove old checkpoints.
   std::regex reg("checkpoints/cp.*." + to_string(seeds_[TPG_SEED]) + "." +
                  to_string(GetState("phase")) + ".rslt");
   for (const auto& entry :
        std::filesystem::directory_iterator("checkpoints")) {
      if (std::regex_match(entry.path().string(), reg) &&
          entry.path().string() != filename) {
         remove(entry.path().string().c_str());
      }
   }
}

/******************************************************************************/
std::string TPG::PhylogenyToString() {
   std::stringstream ss;
   ss << "phyloNode:id:gtime:dtime:fitness:root" << endl;
   ss << "phyloLink:from,to" << endl;
   for (auto it = phylo_graph_.begin(); it != phylo_graph_.end(); it++) {
      ss << "phyloNode:" << (*it).first << ":" << (*it).second.gtime << ":"
         << (*it).second.dtime << ":" << (*it).second.fitnessBin << ":"
         << (*it).second.fitness << ":" << (*it).second.root << endl;
      if ((*it).second.adj.size() > 0)
         for (size_t i = 0; i < (*it).second.adj.size(); i++)
            ss << "phyloLink:" << (*it).first << ":" << (*it).second.adj[i]
               << endl;
      if ((*it).second.ancestorIds.size() > 0) {
         ss << "ancestorIds:" << (*it).first;
         for (auto it2 = (*it).second.ancestorIds.begin();
              it2 != (*it).second.ancestorIds.end(); it2++)
            ss << ":" << *it2;
         ss << endl;
      }
   }
   return ss.str();
}

/******************************************************************************/
std::string TPG::WriteMPICheckpoint(vector<team*>& root_teams) {
   set<MemoryEigen*> memories;
   set<RegisterMachine*, RegisterMachineIdComp> programs;
   set<team*, teamIdComp> teams;
   for (auto tm : root_teams) {
      tm->GetAllNodes(team_map_, teams, programs, memories);
   }
   stringstream ss;
   ss << "seed_tpg:" << seeds_[TPG_SEED] << endl;
   ss << "seed_aux:" << seeds_[AUX_SEED] << endl;
   ss << "t:" << GetState("t_current") << endl;
   ss << "active_task:" << GetState("active_task") << endl;
   ss << "fitMode:" << GetParam<int>("fit_mode") << endl;
   ss << "phase:" << GetState("phase") << endl;
   // for (auto mem : memories) {
   //    ss << mem->ToString();
   // }
   for (auto prog : programs) {
      ss << prog->ToStringMemory();
   }
   for (auto prog : programs) {
      ss << prog->ToString(GetParam<int>("skip_introns"));
   }
   for (auto tm : teams) {
      ss << tm->ToString();
   }
   ss << endl;
   return ss.str();
}

/******************************************************************************/
string TPG::AgentOpUseToString(team* agent) {
   set<MemoryEigen*> memories;
   set<RegisterMachine*, RegisterMachineIdComp> programs;
   set<team*, teamIdComp> teams;
   agent->GetAllNodes(team_map_, teams, programs, memories);
   vector<double>op_use(instruction::NUM_OP, 0);
   for (auto prog : programs) {
      for (size_t i = 0; i < op_use.size(); i++) {
         op_use[i] += prog->op_counts_[i];
      }
   }
   stringstream ss;
   for (size_t i = 0; i < op_use.size(); i++) {
      ss << instruction::op_names_[i] << "," << op_use[i] << endl;
   }
   return ss.str();//VectorToString(op_use);
}

/******************************************************************************/
void TPG::EncodeEvalResultString(EvalData& eval_data) {
   eval_data.eval_result += to_string(static_cast<long>(eval_data.tm->id_));
   eval_data.eval_result += ":" + VectorToStringNoSpace(eval_data.fingerprint);
   eval_data.eval_result += ":" + to_string(GetState("active_task"));
   for (size_t r = 0; r < eval_data.stats_double.size(); r++) {
      eval_data.eval_result += ":" + to_string(eval_data.stats_double[r]);
   }
   for (size_t r = 0; r < eval_data.stats_int.size(); r++) {
      eval_data.eval_result += ":" + to_string(eval_data.stats_int[r]);
   }
   eval_data.eval_result += "\n";
}

void TPG::AppendSelfModifyingRates(std::string& result,
                                   std::vector<team*>& teams) {
   if (std::any_cast<int>(params_["self_modifying"]) == 0) return;

   set<team*, teamIdComp> visited_teams;
   set<RegisterMachine*, RegisterMachineIdComp> programs;
   for (auto* tm : teams) {
      if (tm) tm->GetAllNodes(team_map_, visited_teams, programs);
   }

   for (auto* prog : programs) {
      if (!prog) continue;
      auto* scalar_memory = prog->private_memory_[MemoryEigen::kScalarType_];
      if (!scalar_memory ||
          scalar_memory->working_memory_.size() < kSelfModifyingMinScalarRegisters) {
         continue;
      }

      result += "R:" + to_string(static_cast<long>(prog->id_));
      const size_t last_reg = kSelfModifyingDecoyRegister;
      for (size_t reg = kSelfModifyingFirstRegister; reg <= last_reg; ++reg) {
         double v = scalar_memory->working_memory_[reg](0, 0);
         v = std::isfinite(v) ? v : 0.0;
         result += ":" + to_string(v);
      }
      result += "\n";
   }
}

void TPG::LogReplaySelfModifyingRates(const EvalData& eval_data) {
   if (GetParam<int>("replay") == 0 ||
       std::any_cast<int>(params_["self_modifying"]) == 0 ||
       eval_data.tm == nullptr) {
      return;
   }

   set<team*, teamIdComp> visited_teams;
   set<RegisterMachine*, RegisterMachineIdComp> programs;
   eval_data.tm->GetAllNodes(team_map_, visited_teams, programs);

   double start[4] = {0.0, 0.0, 0.0, 0.0};
   double output[4] = {0.0, 0.0, 0.0, 0.0};
   size_t count = 0;
   for (auto* prog : programs) {
      if (!prog) continue;
      auto* memory = prog->private_memory_[MemoryEigen::kScalarType_];
      if (!memory ||
          memory->const_memory_.size() < kSelfModifyingMinScalarRegisters ||
          memory->working_memory_.size() < kSelfModifyingMinScalarRegisters) {
         continue;
      }
      for (size_t rate = 0; rate < kSelfModifyingRegisterCount; ++rate) {
         const size_t reg = kSelfModifyingFirstRegister + rate;
         start[rate] += SanitizeSelfModifyingRawTendency(
             memory->const_memory_[reg](0, 0));
         output[rate] += SanitizeSelfModifyingRawTendency(
             memory->working_memory_[reg](0, 0));
      }
      ++count;
   }
   if (count == 0) return;

   std::filesystem::create_directories("logs/misc");
   std::ostringstream filename;
   filename << "logs/misc/mutation_rates." << seeds_[TPG_SEED] << "."
            << static_cast<long>(eval_data.tm->id_) << ".csv";
   const bool first_row = eval_data.episode == 0 && eval_data.timestep == 1;
   std::ofstream out(filename.str(), first_row ? std::ios::trunc : std::ios::app);
   if (!out) return;
   if (first_row) {
      out << "episode,timestep,program_count,start_swap,output_swap,"
             "start_delete,output_delete,start_add,output_add,"
             "start_mutate,output_mutate\n";
   }
   out << eval_data.episode << ',' << eval_data.timestep << ',' << count;
   for (size_t rate = 0; rate < 4; ++rate) {
      out << ',' << start[rate] / count << ',' << output[rate] / count;
   }
   out << '\n';
}

/******************************************************************************/
void TPG::DecodeEvalResultString(std::string& s) {
   string line;
   vector<string> split_str;
   istringstream f(s);
   while (getline(f, line)) {
      vector<long> active;
      vector<double> r_stats_double;
      vector<int> r_stats_int;
      SplitString(line, ':', split_str);
      if (!split_str.empty() && split_str[0] == "R" && split_str.size() > 2) {
         long prog_id = atol(split_str[1].c_str());
         auto prog_it = program_pop_.find(prog_id);
         if (prog_it != program_pop_.end() && prog_it->second) {
            auto* scalar_memory =
                prog_it->second->private_memory_[MemoryEigen::kScalarType_];
            if (scalar_memory) {
               for (size_t k = 2; k < split_str.size(); ++k) {
                  const size_t reg = k - 1;  // split_str[2] -> working S1
                  if (reg < scalar_memory->working_memory_.size()) {
                     scalar_memory->working_memory_[reg](0, 0) =
                         atof(split_str[k].c_str());
                  }
               }
            }
         }
         continue;
      }
      size_t s = 0;
      long rslt_id = atol(split_str[s++].c_str());
      string fingerprint = split_str[s++].c_str();
      params_["active_task"] = atoi(split_str[s++].c_str());
      for (int i = 0; i < GetParam<int>("n_point_aux_double"); i++)
         r_stats_double.push_back(atof(split_str[s++].c_str()));
      for (int i = 0; i < GetParam<int>("n_point_aux_int"); i++)
         r_stats_int.push_back(atoi(split_str[s++].c_str()));
      setOutcome(team_map_[rslt_id], fingerprint, r_stats_double,
                 r_stats_int, GetState("t_current"));
   }
}

/******************************************************************************/
void TPG::FinalizeStepData(EvalData& eval_data) {
   // if (eval_data.task->eval_type_ == "RecursiveForecast") {
   //    if (GetParam<string>("forecast_fitness") == "mse") {
   //       auto err =
   //           MeanSquaredError(eval_data.sequence_targ, eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = -err;
   //    } else if (GetParam<string>("forecast_fitness") == "correlation") {
   //       auto corr =
   //           Correlation(eval_data.sequence_targ, eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = corr;
   //    } else if (GetParam<string>("forecast_fitness") == "pearson") {
   //       auto corr = PearsonCorrelation(eval_data.sequence_targ,
   //                                      eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = corr;
   //    } else if (GetParam<string>("forecast_fitness") == "theils") {
   //       auto err =
   //           TheilsStatistic(eval_data.sequence_targ, eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = -err;
   //    } else if (GetParam<string>("forecast_fitness") == "mse_multivar") {
   //       auto err = calculateMSE_Multi(eval_data.sequence_targ,
   //                                     eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = -err;
   //    } else if (GetParam<string>("forecast_fitness") == "theils_multivar") {
   //       auto err = calculateTheils_Multi(eval_data.sequence_targ,
   //                                        eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = -err;
   //    } else if (GetParam<string>("forecast_fitness") == "pearson_multivar") {
   //       auto corr = calculatePearson_Multi(eval_data.sequence_targ,
   //                                          eval_data.sequence_pred);
   //       eval_data.stats_double[REWARD1_IDX] = corr;
   //       if (!isfinite(eval_data.stats_double[REWARD1_IDX]))
   //          eval_data.stats_double[REWARD1_IDX] = 0;
   //    } else {
   //       die(__FILE__, __FUNCTION__, __LINE__,
   //           "Unsupported forecast fitness function");
   //    }
   // }
   eval_data.stats_double[VISITED_TEAMS_IDX] /= eval_data.n_prediction;
   eval_data.stats_double[INSTRUCTIONS_IDX] /= eval_data.n_prediction;
   eval_data.stats_int[POINT_AUX_INT_TASK] = GetState("active_task");
   eval_data.stats_int[POINT_AUX_INT_PHASE] = GetState("phase");
   eval_data.stats_int[POINT_AUX_INT_ENVSEED] = eval_data.episode;
   eval_data.stats_int[POINT_AUX_INT_internalTestNodeId] =
       GetState("internal_test_node_id");           
   EncodeEvalResultString(eval_data);
}

/******************************************************************************/
EvalData TPG::InitEvalData() {
   EvalData eval_data;
   eval_data.tpg_seed = seeds_[TPG_SEED];
   eval_data.stats_double.resize(GetParam<int>("n_point_aux_double"));
   eval_data.stats_int.resize(GetParam<int>("n_point_aux_int"));
   eval_data.animate = GetParam<int>("animate") == 1;
   eval_data.partially_observable = GetParam<int>("partially_observable") == 1;
   eval_data.n_prediction = 0;
   // IMPORTANT: timestep is used by RegisterMachine execution (e.g., temporal indexing).
   // It must be initialized deterministically.
   eval_data.timestep = 0;
   eval_data.sample = 0;
   eval_data.verbose = false;
   return eval_data;
}
