#ifndef TPG_MUTATION_RATE_GENOME_H
#define TPG_MUTATION_RATE_GENOME_H

#include <algorithm>
#include <any>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Directly encoded strategy genes, independent of executable registers.
// Logistic-normal update: Kruisselbrink et al., GECCO 2011, Eq. (2).
class MutationRateGenome {
 public:
   using Params = std::unordered_map<std::string, std::any>;
   std::map<std::string, double> genes;

   static double Number(const Params& params, const std::string& key,
                        double fallback) {
      const auto it = params.find(key);
      if (it == params.end()) return fallback;
      if (it->second.type() == typeid(int)) return std::any_cast<int>(it->second);
      return std::any_cast<double>(it->second);
   }

   static MutationRateGenome FromParams(const Params& params) {
      MutationRateGenome result;
      const auto it = params.find("genome_mutation_rates");
      if (it == params.end()) return result;
      if (it->second.type() != typeid(std::string))
         throw std::invalid_argument("genome_mutation_rates must be a quoted comma-separated string");
      const auto selection = std::any_cast<std::string>(it->second);
      if (selection.empty()) return result;
      std::istringstream names(selection);
      std::string name;
      while (std::getline(names, name, ',')) {
         const auto first = name.find_first_not_of(" \t\r\n");
         const auto last = name.find_last_not_of(" \t\r\n");
         name = first == std::string::npos ? "" : name.substr(first, last - first + 1);
         if (name != "p_instructions_swap" && name != "p_instructions_delete" &&
             name != "p_instructions_add" && name != "p_instructions_mutate")
            throw std::invalid_argument("Unknown genome mutation rate: " + name);
         if (result.genes.count(name))
            throw std::invalid_argument("Duplicate genome mutation rate: " + name);
         const double value = Number(params, name, name == "p_instructions_mutate" ? 1.0 : -1.0);
         if (!std::isfinite(value) || value < 0.0 || value > 1.0)
            throw std::invalid_argument("Genome mutation rate must be in [0,1]: " + name);
         result.genes.emplace(name, value);
      }
      if (selection.back() == ',')
         throw std::invalid_argument("Empty entry in genome_mutation_rates");
      if (Number(params, "self_modifying", 0) != 0)
         throw std::invalid_argument("genome_mutation_rates requires self_modifying=0");
      const double lo = Number(params, "genome_rate_min", 1e-6);
      const double hi = Number(params, "genome_rate_max", 1.0 - 1e-6);
      const double sigma = Number(params, "genome_rate_sigma", 0.2);
      if (!std::isfinite(lo) || !std::isfinite(hi) || !(0 < lo && lo < hi && hi < 1))
         throw std::invalid_argument("Require 0 < genome_rate_min < genome_rate_max < 1");
      if (!std::isfinite(sigma) || sigma < 0)
         throw std::invalid_argument("genome_rate_sigma must be finite and nonnegative");
      for (auto& [key, value] : result.genes) value = std::clamp(value, lo, hi);
      return result;
   }

   double Rate(const std::string& name, double fixed) const {
      const auto it = genes.find(name);
      return it == genes.end() ? fixed : it->second;
   }

   // Called once per selected offspring program, before all instruction passes.
   void Mutate(const Params& params, std::mt19937& rng) {
      if (genes.empty()) return; // Disabled mode consumes no random numbers.
      const double sigma = Number(params, "genome_rate_sigma", 0.2);
      if (sigma == 0.0) return;
      const double lo = Number(params, "genome_rate_min", 1e-6);
      const double hi = Number(params, "genome_rate_max", 1.0 - 1e-6);
      std::normal_distribution<double> normal(0.0, 1.0);
      for (auto& [name, p] : genes) {
         const double logit = std::log(p) - std::log1p(-p) - sigma * normal(rng);
         const double e = std::exp(-std::abs(logit));
         p = std::clamp(logit >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e), lo, hi);
      }
   }

   std::string Encode() const {
      if (genes.empty()) return "";
      std::ostringstream out;
      out << "MRG1" << std::setprecision(std::numeric_limits<double>::max_digits10);
      for (const auto& [name, p] : genes) out << '|' << name << '=' << p;
      return out.str();
   }

   void Decode(const std::string& token, const Params& params) {
      if (token.rfind("MRG1|", 0) != 0)
         throw std::invalid_argument("Unsupported mutation-rate genome checkpoint version");
      const auto configured = FromParams(params);
      std::map<std::string, double> restored;
      std::istringstream input(token.substr(5));
      std::string field;
      while (std::getline(input, field, '|')) {
         const auto split = field.find('=');
         if (split == std::string::npos)
            throw std::invalid_argument("Malformed mutation-rate genome checkpoint");
         const auto name = field.substr(0, split);
         size_t used = 0;
         const auto number = field.substr(split + 1);
         const double p = std::stod(number, &used);
         if (!configured.genes.count(name) || restored.count(name) ||
             used != number.size() || !std::isfinite(p) ||
             p < Number(params, "genome_rate_min", 1e-6) ||
             p > Number(params, "genome_rate_max", 1.0 - 1e-6))
            throw std::invalid_argument("Checkpoint mutation-rate genes/bounds disagree with configuration");
         restored.emplace(name, p);
      }
      if (restored.size() != configured.genes.size() || token.back() == '|')
         throw std::invalid_argument("Checkpoint mutation-rate gene selection disagrees with configuration");
      genes = std::move(restored);
   }
};
#endif
