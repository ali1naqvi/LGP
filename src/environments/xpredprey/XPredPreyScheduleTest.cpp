#include "XPredPreySchedule.h"
#include <iostream>
#include <set>

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    for (int n_predator : {1, 2, 3, 6}) {
      for (int n_prey : {1, 2, 5, 6}) {
        std::vector<long> predators, prey;
        for (int i = 0; i < n_predator; ++i) predators.push_back(i * 2);
        for (int i = 0; i < n_prey; ++i) prey.push_back(i * 2 + 1);
        const auto schedule = MakeXPredPreySchedule(predators, prey, 4, 42, 7, 0);
        std::string encoded;
        std::map<long, int> candidates, opponents, counts;
        for (const auto& match : schedule) {
          require(match.candidate % 2 != match.opponent % 2, "same-role match");
          ++candidates[match.candidate];
          ++opponents[match.opponent];
          require(++counts[match.candidate] == match.candidate_count, "bad candidate count");
          require(++counts[match.opponent] == match.opponent_count, "bad opponent count");
          encoded += match.encode();
        }
        require(schedule.size() == 4 * (predators.size() + prey.size()), "wrong budget");
        for (const auto& population : {predators, prey}) {
          int minimum = 100000, maximum = 0;
          for (long id : population) {
            require(candidates[id] == 4, "unequal candidate exposure");
            minimum = std::min(minimum, opponents[id]);
            maximum = std::max(maximum, opponents[id]);
          }
          require(maximum - minimum <= 1, "unbalanced opponent exposure");
        }
        std::string roundtrip;
        for (const auto& match : XPredPreyEncounter::decode(encoded)) roundtrip += match.encode();
        require(roundtrip == encoded, "schedule serialization changed data");
        std::reverse(predators.begin(), predators.end());
        std::string repeat;
        for (const auto& match : MakeXPredPreySchedule(predators, prey, 4, 42, 7, 0))
          repeat += match.encode();
        require(repeat == encoded, "input order changed seeded schedule");
        auto different = MakeXPredPreySchedule(predators, prey, 4, 43, 7, 0);
        require(different.front().seed != schedule.front().seed, "seed has no effect");
      }
    }
    bool rejected = false;
    try { MakeXPredPreySchedule({0}, {}, 2, 42, 0, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "missing population accepted");
    std::cout << "Balanced scheduling and serialization passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
