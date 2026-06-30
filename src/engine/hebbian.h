#ifndef hebbian_h
#define hebbian_h

#include <functional>
#include <map>
#include <string>
#include <stdexcept>

using HebbianRuleFunc = std::function<double(double, double, double, double, double)>;

class ProgramWeights {
public:
    struct WeightData {
        double weight = 0.0;
        double elig = 0.0;
    };

    double pred_error = 0.0;
    double gamma = 0.0; //discount factor
    double lambda_td = 0.0;
    double plasticity = 0.0;
    long root_step = 0;

    bool additive_root = true;
    bool additive_sub_team = true;
    
    std::map<long, std::map<long, ProgramWeights::WeightData>> weights;
    explicit ProgramWeights(const std::string& ruleName);
    void setWeight();
    void setWeightTD();
    void resetElig();

    void calculatePlasticity(double weightChange, int progs_size);
    void divisiveRenormRow(const double norm);

    void setWeightRoot();
    void setWeightRootTD();
    //void setWeight(const long prev_id, const long curr_id, const bool ltd);
    void decayElig(const long prev_id,const long curr_id);
    void boostElig(const long prev_id,const long curr_id, const double bid_value);

    double getWeight(const long prev_id, const long curr_id, const double curr_bid);
    double getEligWeight(const long prev_id, const long curr_id) const;

    void initializeHebbianRule(const std::string& ruleName);

    void setPrevious(const std::tuple<long, double, double>& prevData);
    void setCurrent(const std::tuple<long, double, double,double>& currentData);

    void clearPrevious();
    void clearCurrent();

    const std::map<long, std::map<long, WeightData>>& getWeights() const;

    //GETS 
    long getPrevProgID() const;
    long getCurrentProgID() const;
    double getPreviousLearningRate() const;
    double getCurrentLearningRate() const;
    double getPreviousBid() const;
    double getCurrentBid() const;

    void resetWeights();
    void printWeights() const;

    // std::map<long, std::map<long, double, double>> weights; 
    HebbianRuleFunc hebbianRuleFunc;

    bool use_initial_noise = false;


    
    long PrevProgID_ = -1;
    long CurrProgID_ = -1;

    double PrevBid_ =  1.0;
    double CurrBid_ = 1.0;
    
    double CurrHebbNoise_ = 0.0;

    double PrevLearningRate_ = 1.0;
    double CurrLearningRate_ = 1.0;

    // eligibility trace data structure. If a trace is > 0, (a winner), we add it here. 
    // This should reset per generation
    //structure should be something like: 
    // {[team, winning program], [team, winning program] ... }
    // the traces of this gets larger for every program visited
    //1. add this in getaction for winning program
    //2. if the winning program is an action, update all values in this array so that the eligibility is traced.
    // using a set over vector would be better, however, there could be an error here so would have to DOUBLE CHECK
    // double running_variance = 0.0;
};

// Hebbian rule functions
double hetero_rule(double bidA, double bidB, double learningRateA, double learningRateB);
double ojas_rule(double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight);
double ojas_rule_no_lr(double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight);

#endif // HEBBIAN_H
