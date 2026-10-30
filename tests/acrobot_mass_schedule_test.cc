#include "../src/environments/classic_control/acrobot_mass_schedule.h"
#include <cassert>

int main() {
    const auto mass = [](int generation, unsigned seed = 42) {
        return AcrobotMassForGeneration(generation, 100, 0.5, 1.5, seed);
    };
    assert(mass(0) == mass(99));
    assert(mass(100) == mass(199));
    assert(mass(99) != mass(100));
    assert(mass(199) != mass(200));
    assert(mass(100, 42) != mass(100, 43));
    // Access order cannot affect checkpoint replay or MPI workers.
    const double resumed = mass(1700);
    (void)mass(50);
    assert(resumed == mass(1799));
    for (int generation = 0; generation <= 10000; ++generation) {
        assert(mass(generation) >= 0.5 && mass(generation) < 1.5);
    }
    bool rejected = false;
    try { AcrobotMassForGeneration(0, 0, 0.5, 1.5, 42); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
}
