#include "../src/engine/training_episode_schedule.h"
#include <cassert>

int main() {
   int completed = 0;
   for (int generation = 0; generation < 10; ++generation) {
      assert(TrainingEpisodesNeeded(completed, 20, 200) == 20);
      completed += 20;
   }
   assert(TrainingEpisodesNeeded(completed, 20, 200) == 0);
   assert(TrainingEpisodesNeeded(195, 20, 200) == 5);
   assert(TrainingEpisodesNeeded(205, 20, 200) == 0);
   // Existing checkpoints with five outcomes continue with new episode IDs.
   assert(TrainingEpisodesNeeded(5, 20, 200) == 20);
   // Other environments retain their original cached-evaluation behavior.
   assert(TrainingEpisodesNeeded(0, 5, 0) == 5);
   assert(TrainingEpisodesNeeded(5, 5, 0) == 0);
}
