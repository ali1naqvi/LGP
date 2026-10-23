// Offline, single-program classic-control mutation assay. See the Python runner.
#include "TPG.h"
#include "ActionWrappers.h"
#include "MountainCarContinuous.h"
#include "Pendulum.h"
#include "Acrobot.h"
#include <filesystem>
#include <memory>
#include <iomanip>

namespace fs = std::filesystem;
using Rates = std::array<double, 4>;
void require(bool ok, const std::string& message) {
   if (!ok) throw std::runtime_error(message);
}
std::string read(const fs::path& path) {
   std::ifstream in(path); require(bool(in), "Cannot read " + path.string());
   return {std::istreambuf_iterator<char>(in), {}};
}
std::string json(const std::string& value) {
   std::string out = "\"";
   for (char c : value) {
      if (c == '\n') out += "\\n";
      else if (c == '\r') out += "\\r";
      else if (c == '\t') out += "\\t";
      else { if (c == '"' || c == '\\') out += '\\'; out += c; }
   }
   return out + '"';
}
// Normalize incidental metadata and exclude runtime phenotype from genotype identity.
std::string genotype(RegisterMachine& p) {
   std::istringstream input(p.ToString(false));
   std::string line, result;
   while (std::getline(input, line)) {
      if (line.starts_with("self_modifying_state:")) continue;
      std::vector<std::string> fields; std::istringstream row(line); std::string field;
      while (std::getline(row, field, ':')) fields.push_back(field);
      if (fields.empty()) continue;
      if (fields[0] == "RegisterMachine") {
         fields[1] = fields[2] = fields[5] = "0";
         std::ostringstream lr, noise;
         lr << std::setprecision(17) << p.learning_rate_;
         noise << std::setprecision(17) << p.initial_heb_noise_;
         fields[7] = lr.str(); fields[8] = noise.str();
      } else if (fields[0] == "MemoryEigen") fields[1] = "0";
      for (size_t i = 0; i < fields.size(); ++i) result += (i ? ":" : "") + fields[i];
      result += '\n';
   }
   return result;
}
void configure(TPG& tpg, const std::string& config, int seed) {
   std::vector<std::string> args = {"neutrality", "parameters_file=" + config,
       "seed_tpg=" + std::to_string(seed), "animate=0"};
   std::vector<char*> pointers; for (auto& a : args) pointers.push_back(a.data());
   tpg.SetParams(pointers.size(), pointers.data());
   std::cout << "Configured " << config << std::endl;
   tpg.state_["n_task"] = 1; tpg.state_["active_task"] = 0;
   tpg.state_["phase"] = _TRAIN_PHASE;
   tpg.params_["n_point_aux_double"] = 3;
   tpg.params_["n_point_aux_int"] = 4;
   tpg.n_input_ = {std::stoi(tpg.GetParam<std::string>("n_input"))};
   require(tpg.GetParam<int>("self_modifying") == 1, "Runtime-modified config required");
   require(tpg.GetParam<int>("continuous_output") == 1 &&
           tpg.GetParam<int>("dynamic_lgp") == 0 &&
           tpg.GetParam<int>("hebb_learning") == 0, "Only atomic scalar control supported");
}
std::unique_ptr<ClassicControlEnv> environment(TPG& tpg) {
   const auto name = tpg.GetParam<std::string>("active_tasks");
   if (name == "MountainCarContinuous")
      return std::make_unique<MountainCarContinuous>(tpg.GetParam<int>("mountaincar_continuous_max_timesteps"));
   if (name == "Pendulum") return std::make_unique<Pendulum>(tpg.GetParam<int>("pendulum_max_timesteps"));
   if (name == "Acrobot") return std::make_unique<Acrobot>(tpg.GetParam<int>("acrobot_max_timesteps"));
   throw std::runtime_error("Unsupported environment: " + name);
}
struct Trace {
   double reward = 0;
   Rates mean{}, last{};
   std::vector<std::vector<double>> observations;
   std::vector<double> actions;
   std::vector<Rates> rates;
};
// Each episode is independent: initialize all memory from inherited constants.
// This intentionally differs from training's cross-episode state carryover.
Trace evaluate(TPG& tpg, RegisterMachine& original, unsigned seed,
               const Trace* probe = nullptr) {
   auto env = environment(tpg);
   std::mt19937 cloning(713), rng(seed);
   auto clone_state = tpg.state_;
   RegisterMachine p(original, tpg.params_, clone_state, cloning);
   for (auto* memory : p.private_memory_)
      memory->working_memory_.assign(memory->const_memory_.begin(), memory->const_memory_.end());
   p.SeedSelfModifyingWorkingFromConstants(tpg.params_);
   p.MarkIntrons(tpg.params_, p.n_memories_);
   auto eval = tpg.InitEvalData();
   eval.episode = 0;
   state observation(tpg.n_input_[0]); eval.obs = &observation;
   eval.task = env.get(); eval.program_out = &p;
   env->Reset(rng);
   Trace result;
   for (int step = 0; probe ? step < int(probe->observations.size()) : !env->Terminal(); ++step) {
      auto obs = probe ? probe->observations[step] : env->GetObsVec(eval.partially_observable);
      observation.Set(obs); eval.timestep = step; eval.n_prediction = step;
      eval.instruction_count = 0;
      p.Run(eval, eval.timestep, 1, eval.verbose);
      const double raw_action = WrapContinuousAction(eval);
      require(std::isfinite(raw_action), "Non-finite action; job retained for investigation");
      const double action = std::clamp(raw_action, env->MinActionContinuous(), env->MaxActionContinuous());
      const auto rates = p.MutationProbabilities(tpg.params_);
      result.observations.push_back(obs); result.actions.push_back(action); result.rates.push_back(rates);
      for (int k = 0; k < 4; ++k) { require(std::isfinite(rates[k]), "Non-finite rate"); result.mean[k] += rates[k]; }
      result.last = rates;
      if (!probe) result.reward += env->Update(WrapDiscreteAction(eval), raw_action, rng).r1;
   }
   require(!result.actions.empty() && std::isfinite(result.reward), "Invalid evaluation");
   for (auto& rate : result.mean) rate /= result.actions.size();
   return result;
}
void write_rates(std::ostream& out, const Rates& rates) { for (auto r : rates) out << ',' << r; }

int main(int argc, char** argv) try {
   require(argc == 12, "config checkpoint target_config_or_dash seed generation parents offspring episodes sampling_seed evaluation_seed mutation_seed");
   const int seed = std::stoi(argv[4]), generation = std::stoi(argv[5]);
   const int nparents = std::stoi(argv[6]), noffspring = std::stoi(argv[7]), nepisodes = std::stoi(argv[8]);
   require(nparents > 0 && noffspring > 0 && nepisodes > 0, "Counts must be positive");
   TPG source, target; configure(source, argv[1], seed);
   const std::string checkpoint = read(argv[2]);
   source.ReadCheckpoint(generation, 0, true, checkpoint);
   std::cout << "Restored checkpoint" << std::endl;
   source.state_["t_current"] = generation;
   std::vector<TPG*> contexts = {&source};
   if (std::string(argv[3]) != "-") { configure(target, argv[3], seed); contexts.push_back(&target); }
   auto teams = source.GetRootTeamsInVec();
   require(teams.size() == size_t(source.GetParam<int>("n_root")), "Checkpoint is not the full surviving population");
   std::sort(teams.begin(), teams.end(), [](team* a, team* b){ return a->id_ < b->id_; });
   require(teams.size() >= size_t(nparents), "Not enough elites");
   for (auto* tm : teams) require(tm->members_.size() == 1 && (*tm->members_.begin())->action_ < 0,
                                 "Expected one atomic program per elite");
   std::mt19937 sample(std::stoul(argv[9])); std::shuffle(teams.begin(), teams.end(), sample);
   teams.resize(nparents);
   std::mt19937 mutation_seeds(std::stoul(argv[11]));
   std::ofstream individuals("individuals.csv"), episodes("episodes.csv"), genomes("genomes.jsonl");
   individuals << std::setprecision(17); episodes << std::setprecision(17);
   individuals << "parent_index,team_id,parent_program_id,offspring,mutation_seed,genotype_changed,total,action_effective,mutation_only,introns,used_swap,used_delete,used_add,used_mutate\n";
   episodes << "parent_index,offspring,environment,episode,episode_seed,return,parent_return,delta_return,steps,mean_swap,mean_delete,mean_add,mean_mutate,final_swap,final_delete,final_add,final_mutate,probe_action_mae,probe_swap_mae,probe_delete_mae,probe_add_mae,probe_mutate_mae\n";
   for (int parent_index = 0; parent_index < nparents; ++parent_index) {
      auto* tm = teams[parent_index]; auto* parent = *tm->members_.begin();
      require(checkpoint.find("self_modifying_state:" + std::to_string(parent->id_) + ":") != std::string::npos,
              "Missing saved runtime rates for sampled parent");
      const auto parent_genome = genotype(*parent);
      const auto saved_rates = parent->MutationProbabilities(source.params_);
      require(parent->mutation_rate_genome_.genes.empty(), "Expected runtime rates, not strategy genes");
      std::cout << "Evaluating parent " << parent->id_ << std::endl;
      const auto initial_state = source.state_;
      std::vector<std::vector<Trace>> baseline(contexts.size());
      for (int offspring = -1; offspring < noffspring; ++offspring) {
         const unsigned mutation_seed = mutation_seeds();
         source.state_ = initial_state; source.rngs_[TPG_SEED].seed(mutation_seed);
         std::unique_ptr<RegisterMachine> child;
         RegisterMachine* individual = parent;
         Rates used = parent->MutationProbabilities(source.params_);
         if (offspring >= 0) {
            child.reset(source.CloneProgram(parent)); individual = child.get();
            if (source.HaveParam("reset_self_modifying_before_mutation") &&
                source.GetParam<int>("reset_self_modifying_before_mutation"))
               individual->SeedSelfModifyingWorkingFromConstants(source.params_);
            used = individual->MutationProbabilities(source.params_);
            source.ProgramMutator_Instructions(individual, POPULATION_ROLE_DEFAULT, tm->id_, parent->id_);
         }
         individual->MarkIntrons(source.params_, individual->n_memories_);
         const auto full = individual->instructions_effective_.size();
         const auto mutation_only = individual->n_self_modification_only_instructions_;
         const auto total = individual->instructions_.size();
         const auto canonical = genotype(*individual);
         individuals << parent_index << ',' << tm->id_ << ',' << parent->id_ << ',' << offspring << ','
                     << mutation_seed << ',' << (canonical != parent_genome) << ',' << total << ','
                     << full - mutation_only << ',' << mutation_only << ',' << total - full;
         write_rates(individuals, used); individuals << '\n';
         genomes << "{\"parent_index\":" << parent_index << ",\"offspring\":" << offspring
                 << ",\"genotype\":" << json(canonical) << "}\n";
         for (size_t c = 0; c < contexts.size(); ++c) {
            auto& context = *contexts[c];
            for (int episode = 0; episode < nepisodes; ++episode) {
               const unsigned episode_seed = std::stoul(argv[10]) + episode;
               auto trace = evaluate(context, *individual, episode_seed);
               if (offspring == -1) baseline[c].push_back(trace);
               const auto& reference = baseline[c][episode];
               double action_mae = 0; Rates rate_mae{};
               if (offspring >= 0) {
                  const auto probe = evaluate(context, *individual, episode_seed, &reference);
                  for (size_t j = 0; j < probe.actions.size(); ++j) {
                     action_mae += std::abs(probe.actions[j] - reference.actions[j]);
                     for (int k = 0; k < 4; ++k) rate_mae[k] += std::abs(probe.rates[j][k] - reference.rates[j][k]);
                  }
                  action_mae /= probe.actions.size();
                  for (auto& x : rate_mae) x /= probe.actions.size();
               }
               episodes << parent_index << ',' << offspring << ',' << context.GetParam<std::string>("active_tasks")
                        << ',' << episode << ',' << episode_seed << ',' << trace.reward << ',' << reference.reward
                        << ',' << trace.reward - reference.reward << ',' << trace.actions.size();
               write_rates(episodes, trace.mean); write_rates(episodes, trace.last);
               episodes << ',' << action_mae; write_rates(episodes, rate_mae); episodes << '\n';
            }
         }
      }
      individuals.flush(); episodes.flush(); genomes.flush();
      require(bool(individuals) && bool(episodes) && bool(genomes), "Output write failed");
      require(genotype(*parent) == parent_genome, "Parent genotype was modified");
      require(parent->MutationProbabilities(source.params_) == saved_rates, "Parent runtime state was modified");
      std::cout << "Completed parent " << parent_index + 1 << '/' << nparents << std::endl;
   }
   source.finalize();
   return 0;
} catch (const std::exception& error) {
   std::cerr << "Neutrality worker: " << error.what() << '\n'; return 1;
}
