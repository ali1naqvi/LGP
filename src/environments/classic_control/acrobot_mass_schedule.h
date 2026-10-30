#ifndef ACROBOT_MASS_SCHEDULE_H
#define ACROBOT_MASS_SCHEDULE_H

#include <cmath>
#include <random>
#include <stdexcept>

// A private RNG makes the schedule independent of policy execution, MPI rank,
// episode count, and checkpoint resume. Blocks are [0,99], [100,199], ... .
inline double AcrobotMassForGeneration(int generation, int interval,
                                      double minimum, double maximum,
                                      unsigned int seed) {
    if (generation < 0 || interval <= 0 || !std::isfinite(minimum) ||
        !std::isfinite(maximum) || minimum <= 0 || maximum < minimum) {
        throw std::invalid_argument("Invalid dynamic Acrobot mass schedule");
    }
    std::mt19937 rng(seed);
    rng.discard(static_cast<unsigned long long>(generation / interval));
    // Explicit conversion gives the same schedule across standard libraries.
    const double unit = static_cast<double>(rng()) / 4294967296.0;
    return minimum + (maximum - minimum) * unit;
}
#endif
