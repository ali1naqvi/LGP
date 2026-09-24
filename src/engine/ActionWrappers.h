#ifndef ActionWrappers_h
#define ActionWrappers_h

#include "EvalData.h"
#include "RegisterMachine.h"
#include "misc.h"

#include <iostream>
#include <Eigen/Dense>

#include <vector>
#include <algorithm>  // For std::transform
#include <cmath> 

// TPG represents discrete actions as negative ints starting at -1
// Map them to positive ints starting at 0
inline int WrapDiscreteAction(EvalData& eval) {
   return (eval.program_out->action_ * -1) - 1;
}

// Derive a discrete action from the winning program's action register
// (scalar working_memory_[S1]) instead of using the evolved/inherited action_.
// The unbounded register value is squashed to
// (0,1) via a sigmoid and split into n_actions equal-width bins, yielding an
// index in [0, n_actions-1].
inline int WrapDiscreteActionFromRegister(EvalData& eval, int n_actions) {
   if (n_actions < 1) n_actions = 1;
   double v = eval.program_out->private_memory_[MemoryEigen::kScalarType_]
                  ->working_memory_[kActionRegister](0, 0);
   if (!std::isfinite(v)) v = 0.0;
   double s = 1.0 / (1.0 + std::exp(-v));  // squash to (0,1)
   int idx = static_cast<int>(std::floor(s * n_actions));
   if (idx >= n_actions) idx = n_actions - 1;  // edge case: s == 1.0
   if (idx < 0) idx = 0;
   return idx;
}

// Multi-action discrete output: derive one discrete action per element of the
// winning program's vector register 1 (vector working_memory_[1]). Each element
// is squashed to (0,1) via a sigmoid and split into n_actions equal-width bins,
// mirroring WrapDiscreteActionFromRegister. The returned vector therefore has
// one discrete action index per vector dimension, to be applied sequentially.
inline std::vector<int> WrapDiscreteActionsFromVector(EvalData& eval, int n_actions) {
   if (n_actions < 1) n_actions = 1;
   auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                  ->working_memory_[1];
   std::vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
   std::vector<int> actions;
   actions.reserve(vec.size());
   for (double v : vec) {
      if (!std::isfinite(v)) v = 0.0;
      double s = 1.0 / (1.0 + std::exp(-v));  // squash to (0,1)
      int idx = static_cast<int>(std::floor(s * n_actions));
      if (idx >= n_actions) idx = n_actions - 1;  // edge case: s == 1.0
      if (idx < 0) idx = 0;
      actions.push_back(idx);
   }
   return actions;
}

inline double WrapContinuousAction(EvalData& eval) {
   return eval.program_out->private_memory_[MemoryEigen::kScalarType_]
       ->working_memory_[1](0, 0);
}

inline double WrapContinuousActionSigmoid(EvalData& eval) {
   double p = eval.program_out->private_memory_[MemoryEigen::kScalarType_]
                  ->working_memory_[1](0, 0);
   return 1 / (1 + exp(-p));
}

inline vector<double> WrapVectorActionArgmax(const EvalData& eval) {
   auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                  ->working_memory_[1];

   vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
   auto max_iter = max_element(vec.begin(), vec.end());
   size_t max_index = distance(vec.begin(), max_iter);

   fill(vec.begin(), vec.end(), 0.0);
   vec[max_index] = 1.0;

   return vec;
}

inline vector<double> WrapVectorActionSoftmax(EvalData &eval) {
    auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                  ->working_memory_[1];

    vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    double max_val = *std::max_element(vec.begin(), vec.end());

    double sum_exps = 0.0;
    for (auto &v : vec) {
        v = std::exp(v - max_val);
        sum_exps += v;
    }
    
    for (auto &v : vec) {
        v /= sum_exps;
    }

    return vec;
}

inline vector<double> WrapVectorAction(EvalData &eval) {
    auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                   ->working_memory_[1];
    vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    return vec;
}


inline std::vector<double> WrapVectorSoftmax(const std::vector<double>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("Input vector cannot be empty.");
    }

    double max_val = *std::max_element(vec.begin(), vec.end());
    std::vector<double> exps(vec.size());
    double sum_exps = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        exps[i] = std::exp(vec[i] - max_val);
        sum_exps += exps[i];
    }

    for (auto& v : exps) {
        v /= sum_exps;
    }

    return exps;
}

inline vector<double> WrapVectorActionSigmoid(EvalData& eval) {
   auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                  ->working_memory_[1];
   vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
   for (auto& v : vec)
      v = sigmoid(v);  // TODO(skelly): better/faster way?
   return vec;
}

inline vector<double> WrapVectorActionTanh(EvalData& eval) {
   auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                  ->working_memory_[1];
   vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
   for (auto& v : vec)
      v = std::tanh(v);  // TODO(skelly): better/faster way?
   return vec;
}


inline vector<double> WrapVectorActionMuJoco(EvalData &eval) {
    auto mat = eval.program_out->private_memory_[MemoryEigen::kVectorType_]
                   ->working_memory_[1];
    vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
    for (auto &v : vec) {
        if (!std::isfinite(v)) v = 0.0;
        v = std::min(std::max(v, -1.0), 1.0);
    }
    return vec;
}


inline vector<double> WrapScalarActionsMujoco(EvalData &eval) {
    auto scalar_mem = eval.program_out->private_memory_[MemoryEigen::kScalarType_];
    const auto &wm = scalar_mem->working_memory_;

    vector<double> vec;
    vec.reserve(wm.size());
    for (size_t i = 0; i < wm.size(); ++i) {
        // S6 is reserved as a neutral self-modification control. Preserve the
        // action-vector shape, but never expose its evolving value to a task.
        if (eval.program_out->self_modifying_ &&
            i == kSelfModifyingDecoyRegister) {
            vec.push_back(0.0);
            continue;
        }
        double v = wm[i](0, 0);
        if (!std::isfinite(v)) v = 0.0;
        v = std::tanh(v); // squash to [-1, 1] via tanh
        vec.push_back(v);
    }
    return vec;
}

#endif
