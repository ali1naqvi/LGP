#ifndef TPG_H
#define TPG_H
#include <TaskEnv.h>

#include <any>
#include <filesystem>
#include <regex>
#include <iomanip>
#include <random>
#include <yaml-cpp/yaml.h>

#include "MemoryEigen.h"
#include "RegisterMachine.h"
#include "api_client.h"
#include "instruction.h"
#include "point.h"
#include "state.h"
#include "team.h"
#include "EvalData.h"
#include "sequence_comparisons.h"

#define NUM_RNG 2
#define TPG_SEED 0
#define AUX_SEED 1

class TPG {
  public:
   TPG();
   TPG(const TPG&);
   ~TPG();

    void AddProgram(RegisterMachine *p);
    void AddTeam(team *tm);
    void RemoveTeam(team *tm);
    void AddMemory(long prog_id, MemoryEigen *m);
    team *getTeamByID(long id);
    bool haveEliteTeam(string taskset, int fitMode, int phase);
    void Seed(size_t i, uint_fast32_t s);
    void InitExperimentTracking(APIClient *apiClient);
    void ExtinctionEvent();

    /* Map Elites*/
    void InitMapElitesArchive();
    void UpdateMapElitesArchive();
    void SaveMapElitesArchive();
    const std::vector<team*>& GetMapElitesArchive() const;

   /***************************************************************************
     * Methods to implement the TPG algorithm.
     **************************************************************************/
    string AgentOpUseToString(team* agent);
    void checkRefCounts(const char *);
    void CleanupProgramsWithNoRefs();
    void clearMemory();
    void countRefs();
    void finalize();
    void genSampleSets(size_t);
    team* TeamSelector_Tournament(vector<team*> &candidate_parent_teams);
    void ComputeParetoFronts(vector<team*>& teams);  // NSGA-II non-dominated sorting
    void GenerateNewTeams();
    void TeamMutator_ProgramOrder(team *team_to_mu);
    void TeamMutator_LambdaVariation(team *team_to_mu);
    void TeamMutator_DecayVariation(team *team_to_mu);
    void TeamMutator_AddPrograms(team *team_to_mu);
    void TeamMutator_RemovePrograms(team *team_to_mu);
    team *CloneTeam(team *team_to_clone);
    RegisterMachine *CloneProgram(RegisterMachine *prog);
    // void ProgramMutator_Memory(RegisterMachine *&prog_to_mu);
    void ProgramMutator_Instructions(RegisterMachine *prog_to_mu);
    void ProgramMutator_ActionPointer(RegisterMachine *prog_to_mu, team *new_team,
                                      int &n_new_teams);
    void AddAncestorToPhylogeny(team *parent, team *new_team);
    void AddTeamToPhylogeny(team *new_team);
    // void InsertTeamLambdaToCSV(team* new_team);
    void ApplyVariationOps(team *team_to_modify, int &n_new_teams);
    team *genTeamsInternal(long, mt19937 &, set<team *, teamIdComp> &,
                           map<long, team *> &);
    int genUniqueProgram(RegisterMachine *, set<RegisterMachine *, RegisterMachineIdComp>);
    void GetAction(EvalData& eval_data, std::mt19937& rng, std::unordered_map<std::string, std::any>& params, std::tuple<long, double, double>& prev_prog_history);
    void LogReplaySelfModifyingRates(const EvalData& eval_data);
    //TD ERROR
    void ComputeIndReward(EvalData& eval_data, std::unordered_map<string, std::any> &params, double reward_per_step);

    void GetAllNodes(team *tm, set<team *, teamIdComp> &teams,
                     set<RegisterMachine *, RegisterMachineIdComp> &RegisterMachines);
    // void GetAllNodes(team *tm, set<team *, teamIdComp> &teams,
    //                  set<RegisterMachine *, RegisterMachineIdComp> &RegisterMachines,
    //                  set<MemoryEigen *, MemoryEigenIdComp> &memories);


    team *GetBestTeam();
    // set<team*, teamFitnessLexicalCompare> GetRootTeamsInSet();
    set<team*, teamIdComp> GetRootTeamsInSet();
    vector<team *> GetRootTeamsInVec() const;
    map<long, team *> GetRootTeamsInMap() const;
    void InitTeams();
    void internalReplacementPareto(int, int, int, team *, map<long, team *> &,
                                   set<team *, teamIdComp> &, mt19937 &);
    bool isElitePS(team *tm, int phase);
    void MarkEffectiveCode();
    void GetPolicyFeatures(int, set<long> &, bool);
    // void printGraphDot(
    //     team *, size_t frame, int episode, int step, size_t depth,
    //     vector<RegisterMachine *> allPrograms, vector<RegisterMachine *> winningPrograms,
    //     vector<set<long>> decisionFeatures,
    //     vector<set<MemoryEigen *, MemoryEigenIdComp>> decisionMemories,
    //     vector<team *> teamPath, bool drawPath,
    //     set<team *, teamIdComp> visitedTeamsAllTasks);
    // void printGraphDotGPEM(long rootTeamId, map<long, string> &teamColMap,
    //                        set<team *, teamIdComp> &visitedTeamsAllTasks,
    //                        vector<map<long, double>> &teamUseMapPerTask);
    void printGraphDotGPTPXXI(long rootTeamID,
                              set<team *, teamIdComp> &visitedTeamsAllTasks,
                              vector<map<long, double>> &teamUseMapPerTask,
                              vector<int> &steps_per_task);
    
    void printGraphDotMujoco(long rootTeamID,
                              set<team *> &visitedTeamsAllTasks);
                              
    void printGraphDotMujocoVisited(long rootTeamID,
                              set<team *> &visitedTeamsAllTasks);
    // void printGraphDotGPEMAnimate(long rootTeamId, size_t frame, int episode,
    //                               int step, size_t depth,
    //                               vector<RegisterMachine *> allPrograms,
    //                               vector<RegisterMachine *> winningPrograms,
    //                               set<team *, teamIdComp>
    //                               &visitedTeamsAllTasks, vector<map<long,
    //                               double>> &teamUseMapPerTask, vector<team *>
    //                               teamPath);
    void printHostGraphsDFS(long, long);
    void printHostGraphsHostsOnly(long, long);
    void printPhyloGraphDot(team *);
    void printProgramInfo();
    void printOss();
    void printOss(ostringstream &o);
    void printTeamInfo(long, int, bool, bool, long teamId = -1);
    void trackTeamInfo(long, int, bool, long teamId = -1);
    static double ComputeTeamFLOPs(const team* t);
    void RegisterMachineCrossover(RegisterMachine *p1, RegisterMachine *p2,
                          RegisterMachine **c1, RegisterMachine **c2);


    void ReadCheckpoint(long, int, bool, const string &);
    
    void LinearCrossover(RegisterMachine* gp1, RegisterMachine* gp2,
                         RegisterMachine** c1, RegisterMachine** c2);

    void ReadParameters(string file_name,
                        std::unordered_map<string, std::any> &params);
    void recalculateProgramRefs();
    void SanityCheck();
    void TeamSizesMatchProgRefs();  // Sanity check.
    inline void resetOutcomes(int phase, bool roots);
    void SelectTeams();
    team *TeamCrossover(team* parent1, team* parent2);
    void UpdateTeamPhyloData(team *tm);
    void FindSingleTaskFitnessRange(vector<TaskEnv *> &tasks,
                                    vector<vector<double>> &mins,
                                    vector<vector<double>> &maxs);
    void FindSingleTaskMinMax(vector<TaskEnv *> &tasks,
                                    vector<vector<double>> &mins,
                                    vector<vector<double>> &maxs);
    vector<team *> NormalizeScoresAndRankTeams(
        vector<TaskEnv *> &tasks, vector<int> &set,
        vector<vector<double>> &min_scores, vector<vector<double>> &max_scores);
    void FindMultiTaskElites(vector<TaskEnv *> &tasks,
                             vector<vector<double>> &min_scores,
                             vector<vector<double>> &max_scores);
    void SetEliteTeams(vector<TaskEnv *> &tasks);
    void setOutcome(team *tm, string behav, vector<double> &rewards,
                    vector<int> &ints, long gtime);
    std::string PhylogenyToString();
    inline void teamMap(map<long, team *> &team_map) const {
        team_map = team_map_;
    }
    // void teamTaskRank(int, const vector<int> &);
    void updateMODESFilters(bool);
    void WriteCheckpoint(bool);
    std::string WriteMPICheckpoint(vector<team *> &);

   /*****************************************************************************
     *  TPG member variables and data structures.
     ****************************************************************************/
    //  Populations
    std::tuple<long, double, double> prev_prog_history_ = std::make_tuple(-1, 1, 1); //id, bid, learning_rate
    set<team *, teamIdComp> team_pop_;      // Teams
    // Map team id -> team* for RegisterMachine graph traversal
    map<long, team *> team_map_;
    // keep track of which teams are elites wrt each taskSet
    map<string, vector<team *>> task_set_map_;
    map<long, RegisterMachine *> program_pop_;
    vector<vector<long>> _Memids;
    // one map for each memory type: id->memory*
    vector<map<long, MemoryEigen *>> _Memory;

    /* Archive of best teams */
    std::vector<team*> map_elites_archive_;

    // Phylogeny data
    map<long, phyloRecord> phylo_graph_;
    map<long, modesRecord> _allComponentsA;
    map<long, modesRecord> _allComponentsAt;
    map<long, modesRecord> _persistenceFilterA;
    map<long, modesRecord> _persistenceFilterA_t;
    map<long, modesRecord> _persistenceFilterAllTime;

    vector<bool> _ops;  // legel RegisterMachine operations
    set<team *, teamIdComp> _eliteTeams;
    // keys: taskSet, fitMode, phase
    map<string, map<int, map<int, team *>>> _eliteTeamPS;
    set<long> elite_team_id_history_;
    vector<mt19937> rngs_;
    vector<uint_fast32_t> seeds_;
    vector<int> n_input_;                // Number of inputs per task.
    vector<int> observation_buff_size_;  // Observaiton buff size per task.
    ostringstream oss;                   // logging, reporting
    vector<size_t> _numEliteTeamsCurrent;

    uniform_real_distribution<> real_dist_;  // random reals in [0,1]

    // API client for tracking experiments
    APIClient *api_client_;

   public:
    std::unordered_map<std::string, std::any> params_;
    template <typename T>
    T GetParam(string p) {
        return std::any_cast<T>(params_[p]);
    }

    bool HaveParam(string p) { return params_.find(p) != params_.end(); }
    std::unordered_map<std::string, int> state_;
    int GetState(string p) { return state_[p]; }
    void ProcessParams();
    void SetParams(int argc, char **argv);
    void MutateActionToTerminal(RegisterMachine *prog_to_mu, team *new_team);
    void MutateActionToTeam(RegisterMachine *prog_to_mu, team *new_team,
                            int &n_new_teams);

    // Method used by TPG processes to share evaluation data.
    void EncodeEvalResultString(EvalData& eval_data);
    void DecodeEvalResultString(std::string& s);
    void AppendSelfModifyingRates(std::string& result,
                                  std::vector<team*>& teams);
    void FinalizeStepData(EvalData& eval_data);  
    EvalData InitEvalData();
};

#endif
