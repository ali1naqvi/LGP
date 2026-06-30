#ifndef FastSimEnvironment_H
#define FastSimEnvironment_H

#include <fastsim.hpp>
#include <TaskEnv.h>
#include <misc.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <cctype>
#include <algorithm>
#include <cstdlib>


class FastSimEnv : public TaskEnv {
public:
    std::shared_ptr<fastsim::Map> map_   = nullptr;
    std::shared_ptr<fastsim::Robot> robot_ = nullptr;
    std::unique_ptr<fastsim::Display> display_;
    fastsim::Goal goal_pos_;

    FastSimEnv() : goal_pos_(0.f, 0.f, 0.f, 0){}

    std::string  model_path_;
    int obs_size_ = 0;
    std::vector<std::pair<int,int>> path_;

    int step_ = 0;
    int max_step_ = 1000;
    bool sticky_walls_ = false;
    bool replay_ = false;
    bool simple_discrete_ = false;
    bool save_map_image_ = false;
    bool dynamic_maze_ = false;
    virtual void reset(std::mt19937& rng, int& episode_number) = 0;
    virtual void set_baseline_mean(double &mean) = 0;
    virtual bool terminal() = 0;
    virtual Results sim_step(std::vector<double>& action) = 0;
    virtual Results sim_step(int &action) = 0;

    // Tasks with evolvable perception can override this to expose one current
    // resource observation plus additional resource observations ahead.
    virtual void SetResourceObservationCount(int count) { (void)count; }

    void initialize_simulation()
    {
        fastsim::Settings settings(model_path_);
        map_ = settings.map();
        robot_ = settings.robot();
        display_ = std::make_unique<fastsim::Display>(map_, robot_);
    }
    
    int GetObsSize() { return obs_size_; }
};

#endif
