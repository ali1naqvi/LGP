#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

// An encounter credits fitness to the candidate, but BOTH participants retain
// their execution state. Counts include candidate and opponent appearances and
// restart at each generation/phase. No worker identity enters this schedule.
struct XPredPreyEncounter {
  long id = 0;
  long candidate = -1;
  long opponent = -1;
  int episode = 0;
  std::uint32_t seed = 0;
  int candidate_count = 0;
  int opponent_count = 0;

  std::string encode() const {
    std::ostringstream out;
    out << "xpredprey_encounter:" << id << ':' << candidate << ':' << opponent
        << ':' << episode << ':' << seed << ':' << candidate_count << ':'
        << opponent_count << '\n';
    return out.str();
  }

  static std::vector<XPredPreyEncounter> decode(const std::string& packet) {
    std::vector<XPredPreyEncounter> result;
    std::istringstream input(packet);
    std::string line;
    while (std::getline(input, line)) {
      const std::string prefix = "xpredprey_encounter:";
      if (line.rfind(prefix, 0) != 0) continue;
      line.erase(0, prefix.size());
      std::replace(line.begin(), line.end(), ':', ' ');
      std::istringstream fields(line);
      XPredPreyEncounter encounter;
      if (!(fields >> encounter.id >> encounter.candidate >> encounter.opponent
                   >> encounter.episode >> encounter.seed
                   >> encounter.candidate_count >> encounter.opponent_count)) {
        throw std::invalid_argument("Malformed XPredPrey encounter");
      }
      result.push_back(encounter);
    }
    return result;
  }
};

inline std::vector<XPredPreyEncounter> MakeXPredPreySchedule(
    std::vector<long> predators, std::vector<long> prey, int episodes,
    std::uint32_t seed, int generation, int phase) {
  if (predators.empty() || prey.empty() || episodes < 1) {
    throw std::invalid_argument("XPredPrey needs both populations and positive episodes");
  }
  std::sort(predators.begin(), predators.end());
  std::sort(prey.begin(), prey.end());
  std::vector<long> all = predators;
  all.insert(all.end(), prey.begin(), prey.end());
  std::sort(all.begin(), all.end());
  if (std::adjacent_find(all.begin(), all.end()) != all.end()) {
    throw std::invalid_argument("XPredPrey schedule contains duplicate team IDs");
  }
  std::seed_seq seeds{seed, static_cast<std::uint32_t>(generation),
                      static_cast<std::uint32_t>(phase)};
  std::mt19937 rng(seeds);
  std::shuffle(predators.begin(), predators.end(), rng);
  std::shuffle(prey.begin(), prey.end(), rng);
  std::vector<long> populations[2]{predators, prey};
  std::map<long, int> counts;
  std::vector<XPredPreyEncounter> result;
  // Alternating which role goes first avoids always putting one role's
  // candidate appearances at the end. Encounter order is intentional state.
  const int first_role = static_cast<int>(rng() % 2);
  for (int episode = 0; episode < episodes; ++episode) {
    for (int turn = 0; turn < 2; ++turn) {
      const int role = (first_role + episode + turn) % 2;
      const auto& candidates = populations[role];
      const auto& opponents = populations[1 - role];
      // Equal sizes: rotate pairings each episode. Unequal sizes: consume a
      // continuous opponent cycle so opponent loads differ by at most one
      // within each role over the complete phase.
      const size_t offset = candidates.size() == opponents.size()
          ? static_cast<size_t>(episode)
          : static_cast<size_t>(episode) * candidates.size();
      for (size_t i = 0; i < candidates.size(); ++i) {
        const long candidate = candidates[i];
        const long opponent = opponents[(offset + i) % opponents.size()];
        result.push_back({static_cast<long>(result.size()), candidate, opponent,
                          episode, static_cast<std::uint32_t>(rng()),
                          ++counts[candidate], ++counts[opponent]});
      }
    }
  }
  return result;
}
