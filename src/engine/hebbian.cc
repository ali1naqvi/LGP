// In hebbian.cpp
#include "hebbian.h"

#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <unordered_map>
#include <cstdlib>       
#include <ctime> 
#include <iomanip>

double hetero_rule(double bidA, double bidB, double learningRateA, double learningRateB) {return (1 - bidA) * (1- bidB) * learningRateA * learningRateB;}
double ojas_rule(double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight) { return (learningRateB * ((bidB * bidA) - (bidB*bidB * CurrWeight)));}
double ojas_rule_no_lr(double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight) { return ((bidB * bidA) - (bidB*bidB * CurrWeight));}
double homeostatic(double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight) { return (learningRateB * (bidB - CurrWeight));}

ProgramWeights::ProgramWeights(const std::string& ruleName) {
    initializeHebbianRule(ruleName);
}

const std::map<long, std::map<long, ProgramWeights::WeightData>>& ProgramWeights::getWeights() const {
    return weights;
}

void ProgramWeights::printWeights() const
{
    if (weights.empty()) {
        return;
    }
    std::cerr << "----------- Weights Matrix: ---------------" << std::endl;
    for (const auto& [prev_id, curr_map] : weights) {
        std::cerr << "Prev ID " << prev_id << ": with bid: " << PrevBid_ << "\n";
        if (curr_map.empty()) {
            std::cerr << "  -> No current IDs found for this prev_id.\n";
        } else {
            for (const auto& [curr_id, data] : curr_map) {
                std::cerr << "  -> Prev ID " << prev_id << "  -> Curr ID " << curr_id << " weight: " << std::fixed << std::setprecision(3) << data.weight << " and Elig: " << data.elig;
                if (std::isnan(data.weight)) {
                    std::cerr << "  [Detected NaN!]";
                    std::cerr << "\n       Details:";
                    std::cerr << "\n         PrevProgID: " << prev_id;
                    std::cerr << "\n         CurrProgID: " << curr_id;
                    std::cerr << "\n         PrevBid: " << getPreviousBid();
                    std::cerr << "\n         CurrBid: " << CurrBid_;
                    std::cerr << "\n         PrevLearningRate: " << PrevLearningRate_;
                    std::cerr << "\n         CurrLearningRate: " << CurrLearningRate_;
                }
                std::cerr << std::endl;
            }
        }
    }
}
void ProgramWeights::initializeHebbianRule(const std::string& ruleName) {
    
    if (ruleName == "hetero") {
        hebbianRuleFunc = [](double bidA, double bidB, double learningRateA, double learningRateB, double) {
            return hetero_rule(bidA, bidB, learningRateA, learningRateB);
        };
    } 
    else if (ruleName == "ojas") {
        hebbianRuleFunc = [](double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight) {
            return ojas_rule(bidA, bidB, learningRateA, learningRateB, CurrWeight);
        };
    } 
    else if (ruleName == "ojas_no_lr") {
        hebbianRuleFunc = [](double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight) {
            return ojas_rule_no_lr(bidA, bidB, learningRateA, learningRateB, CurrWeight);
        };
    } 
    else if (ruleName == "homeostatic") {
        hebbianRuleFunc = [](double bidA, double bidB, double learningRateA, double learningRateB, double CurrWeight) {
            return homeostatic(bidA, bidB, learningRateA, learningRateB, CurrWeight);
        };
    } 
    else {
        throw std::invalid_argument("Invalid Hebbian rule: " + ruleName);
    }
}

double ProgramWeights::getWeight(const long prev_id, const long curr_id, const double curr_bid){
    auto prev_it = weights.find(prev_id);
    if (prev_it != weights.end()) {
        auto curr_it = prev_it->second.find(curr_id);
        if (curr_it != prev_it->second.end()) {
            return curr_it->second.weight;  
        }
    }
        weights[prev_id][curr_id].weight = curr_bid;
        return curr_bid;
}

// void ProgramWeights::boostElig(const long prev_id, const long curr_id, const double bid_value) {
//     weights[prev_id][curr_id].elig += PrevBid_ * bid_value;
//     if (weights[prev_id][curr_id].elig > 1.0)  weights[prev_id][curr_id].elig = 1.0; 
// }

void ProgramWeights::boostElig(const long prev_id, const long curr_id, const double bid_value) {
   // weights[prev_id][curr_id].elig = (gamma * lambda_td) * weights[prev_id][curr_id].elig + PrevBid_ * bid_value;
    // weights[prev_id][curr_id].elig += (PrevBid_ * bid_value);
    weights[prev_id][curr_id].elig = 1;
    // if (weights[prev_id][curr_id].elig > 1.0)  weights[prev_id][curr_id].elig = 1.0; 
}


void ProgramWeights::decayElig(const long prev_id, const long curr_id) {
    // weights[prev_id][curr_id].elig *= (gamma * lambda_td);
    weights[prev_id][curr_id].elig *= gamma;
}

void ProgramWeights::calculatePlasticity(double WeightChange, int progs_size){
    if (progs_size == 0){
        plasticity += pow(WeightChange, 2);
    }else{
        plasticity = sqrt((plasticity / progs_size * root_step));
    }
}

void ProgramWeights::divisiveRenormRow(const double norm){
        auto prev_it = weights.find(PrevProgID_);
        if (prev_it != weights.end()) {
            double sum_abs = 0.0;
            for (auto &kv : prev_it->second) {
                sum_abs += std::abs(kv.second.weight);
            }
            if (sum_abs > 0.0) {
                double scale = norm / sum_abs; // Sg = 3.0
                for (auto &kv : prev_it->second) {
                    kv.second.weight *= scale;
                }
            }
        }
    }

void ProgramWeights::setWeightRoot() {
    double oldWeight = getWeight(PrevProgID_, CurrProgID_, CurrBid_);
    double &w_raw = weights[PrevProgID_][CurrProgID_].weight;

    double WeightChange = CurrLearningRate_ * (CurrBid_ - oldWeight); //learning_rate * (postSynapticActivity - weight); 
    w_raw += WeightChange;
    calculatePlasticity(WeightChange, 0);
    // w_raw *= 0.95;
    // w_raw = std::clamp(w_raw, -5.0, 5.0);
}

void ProgramWeights::setWeightRootTD() {
    double oldWeight = getWeight(PrevProgID_, CurrProgID_, CurrBid_);
    double &w_raw = weights[PrevProgID_][CurrProgID_].weight;
    double WeightChange = CurrLearningRate_ * (CurrBid_ - oldWeight); //learning_rate * (postSynapticActivity - weight); 
    calculatePlasticity(WeightChange, 0);
    auto prev_it = weights.find(PrevProgID_);
    double e_eff = 0.0;

    if (prev_it != weights.end()) {
        const auto &row = prev_it->second;
        double sumE = 0.0;
        for (const auto &kv : row) {
            // only count nonnegative elig; add epsilon to avoid 0/0
            sumE += std::max(0.0, kv.second.elig);
        }
        const double eps = 1e-8;
        double e_raw = getEligWeight(PrevProgID_, CurrProgID_);
        e_eff = (std::max(0.0, e_raw) + eps) / (sumE + eps * row.size());
    } else {
        e_eff = 1.0; // fallback if row missing
    }

    // Use e_eff instead of raw elig:
    // NOTE: pred_error is now set upstream to a routing-based modulatory signal (m_t)
    // double mod_signal = pred_error;
    if (additive_root){
        // std::cout  << w_raw << " " << WeightChange << " " << pred_error << " " << e_eff << " " << std::endl;
        w_raw += WeightChange; w_raw += (pred_error * e_eff);

    }
    else{w_raw += WeightChange * (pred_error * e_eff);}
    // std::cout<< "initial noise: " << CurrHebbNoise_ << std::endl;
    // std::cout<<  "PROGID: " << CurrProgID_;
    // std::cout << " weight: " << w_raw <<  " weight change : " << WeightChange << " pred error: " << pred_error << " elig: " <<  getEligWeight(PrevProgID_, CurrProgID_) << std::endl;
    // w_raw *= 0.95;
    // w_raw = std::clamp(w_raw, -5.0, 5.0);
}

void ProgramWeights::setWeight() {
    double oldWeight = getWeight(PrevProgID_, CurrProgID_, CurrBid_);
    double &w_raw = weights[PrevProgID_][CurrProgID_].weight;
    double WeightChange = hebbianRuleFunc(PrevBid_, CurrBid_,
                                        PrevLearningRate_, CurrLearningRate_,
                                        oldWeight);
    w_raw +=  WeightChange;
    // w_raw *= 0.95;
    // w_raw = std::clamp(w_raw, -5.0, 5.0);
}
void ProgramWeights::setWeightTD() {
    double oldWeight = getWeight(PrevProgID_, CurrProgID_, CurrBid_);
    // double norm_oldWeight = std::tanh(oldWeight);
    double &w_raw = weights[PrevProgID_][CurrProgID_].weight;
    double WeightChange = hebbianRuleFunc(PrevBid_, CurrBid_,
                                        PrevLearningRate_, CurrLearningRate_,
                                        oldWeight);
    
        auto prev_it = weights.find(PrevProgID_);
    double e_eff = 0.0;

    if (prev_it != weights.end()) {
        const auto &row = prev_it->second;
        double sumE = 0.0;
        for (const auto &kv : row) {
            // only count nonnegative elig; add epsilon to avoid 0/0
            sumE += std::max(0.0, kv.second.elig);
        }
        const double eps = 1e-8;
        double e_raw = getEligWeight(PrevProgID_, CurrProgID_);
        e_eff = (std::max(0.0, e_raw) + eps) / (sumE + eps * row.size());
    } else {
        e_eff = 1.0; // fallback if row missing
    }

    // Use e_eff instead of raw elig:
    // NOTE: pred_error is now set upstream to a routing-based modulatory signal (m_t)
    // double mod_signal = pred_error;
    if (additive_sub_team){
        w_raw += WeightChange; w_raw += (pred_error * e_eff);}
    else{
        w_raw += WeightChange * (pred_error) * e_eff;
        //w_raw +=  WeightChange * (pred_error * getEligWeight(PrevProgID_, CurrProgID_));
    }
    // std::cout<< "initial noise: " << CurrHebbNoise_ << std::endl;
    // std::cout<<  "PROGID: " << CurrProgID_;
    // std::cout << " weight: " << w_raw <<  " weight change : " << WeightChange << " pred error: " << pred_error << " elig: " <<  getEligWeight(PrevProgID_, CurrProgID_) << std::endl;
    // w_raw *= 0.95;
    // w_raw = std::clamp(w_raw, -5.0, 5.0);
}



double ProgramWeights::getEligWeight(const long prev_id, const long curr_id) const {
    auto prev_it = weights.find(prev_id);
    if (prev_it != weights.end()) {
        auto curr_it = prev_it->second.find(curr_id);
        if (curr_it != prev_it->second.end()) {
            return curr_it->second.elig;
        }
    }
    return 0.0;
}

void ProgramWeights::setPrevious(const std::tuple<long, double, double>& prevData) {
    PrevProgID_ = std::get<0>(prevData);
    PrevBid_ = std::get<1>(prevData); // Store prev bid
    PrevLearningRate_ = std::get<2>(prevData);
}

void ProgramWeights::setCurrent(const std::tuple<long, double, double, double>& currentData) {
    CurrProgID_ = std::get<0>(currentData);
    CurrBid_ = std::get<1>(currentData); // Store current bid
    CurrLearningRate_ = std::get<2>(currentData);
    CurrHebbNoise_ = std::get<3>(currentData);
}
void ProgramWeights::clearPrevious() {
    PrevProgID_ = -1;
    PrevBid_ = 1;
    PrevLearningRate_ = 1;
}

void ProgramWeights::clearCurrent() {
    CurrProgID_ = -1;
    CurrBid_ = 1;
    CurrLearningRate_ = 1;
    CurrHebbNoise_ = 0;
}

long ProgramWeights::getPrevProgID() const {
    return PrevProgID_;
}

long ProgramWeights::getCurrentProgID() const {
    return CurrProgID_;
}

double ProgramWeights::getPreviousLearningRate() const {
    return PrevLearningRate_;
}

double ProgramWeights::getCurrentLearningRate() const {
    return CurrLearningRate_;
}

double ProgramWeights::getPreviousBid() const {
    return PrevBid_;
}

double ProgramWeights::getCurrentBid() const {
    return CurrBid_;
}


void ProgramWeights::resetWeights() {
    weights.clear();
    PrevProgID_ = -1;
    CurrProgID_ = -1;

    PrevBid_ =  1;
    CurrBid_ = 1;

    PrevLearningRate_ = 1;
    CurrLearningRate_ = 1;

     CurrHebbNoise_ = 0;
}

void ProgramWeights::resetElig() {
    for (auto& [prev_id, curr_map] : weights) {
        for (auto& [curr_id, data] : curr_map) {
            data.elig = 0.0;
        }
    }
    // running_variance = 0.0;
    pred_error = 0.0;
}
