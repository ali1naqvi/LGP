#ifndef TRAINING_EPISODE_SCHEDULE_H
#define TRAINING_EPISODE_SCHEDULE_H

#include <algorithm>

// A positive cap enables accumulating fresh batches; zero preserves caching.
inline int TrainingEpisodesNeeded(int completed, int batch, int lifetime_cap) {
   return std::max(0, lifetime_cap > 0
                          ? std::min(batch, lifetime_cap - completed)
                          : batch - completed);
}

#endif
