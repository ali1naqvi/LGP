#include "TaskEnvFactory.h"
#include "CartPole.h"
#include "Acrobot.h"
#include "CartCentering.h"
#include "Pendulum.h"
#include "MountainCar.h"
#include "MountainCarContinuous.h"
#include "RecursiveForecast.h"
#include "Mujoco_Ant_v4.h"
#include "Mujoco_Inverted_Pendulum_v4.h"
#include "Mujoco_Inverted_Double_Pendulum_v4.h"
#include "Mujoco_Half_Cheetah_v4.h"
#include "Mujoco_Half_Cheetah_Hurdles_v4.h"
#include "Mujoco_Reacher_v4.h"
#include "FastSim_Maze.h"
#include "FastSim_Gradient.h"
#include "Mujoco_Hopper_v4.h"
#include "Mujoco_Humanoid_Standup_v4.h"
#include "XPredPreyTask.h"

TaskEnv* TaskEnvFactory::createTask(const std::string& name, std::unordered_map<std::string, std::any>& params) {
    static const std::map<std::string, CreatorFunc> registry = {
        {"Cartpole", [](std::unordered_map<std::string, std::any>&) { return new CartPole(); }},
        {"Acrobot", [](std::unordered_map<std::string, std::any>&) { return new Acrobot(); }},
        {"CartCentering", [](std::unordered_map<std::string, std::any>&) { return new CartCentering(); }},
        {"Pendulum", [](std::unordered_map<std::string, std::any>& params) {
            auto value = [&params](const char* key, int fallback) {
                auto it = params.find(key);
                return it == params.end() ? fallback : std::any_cast<int>(it->second);
            };
            return new Pendulum(value("pendulum_max_timesteps", 200),
                                value("pendulum_n_eval_train", 20),
                                value("pendulum_n_eval_validation", 0),
                                value("pendulum_n_eval_test", 100));
        }},
        {"MountainCar", [](std::unordered_map<std::string, std::any>&) { return new MountainCar(); }},
        {"MountainCarContinuous", [](std::unordered_map<std::string, std::any>&) { return new MountainCarContinuous(); }},
        {"Sunspots", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Sunspots"); }},
        {"Mackey", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Mackey"); }},
        {"Laser", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Laser"); }},
        {"Offset", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Offset"); }},
        {"Duration", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Duration"); }},
        {"Pitch", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Pitch"); }},
        {"PitchBach", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("PitchBach"); }},
        {"Bach", [](std::unordered_map<std::string, std::any>&) { return new RecursiveForecast("Bach"); }},
        {"FastSimMaze", [](std::unordered_map<std::string, std::any>& params) { return new FastSim_Maze(params); }},
        {"FastSimGradient", [](std::unordered_map<std::string, std::any>& params) { return new FastSim_Gradient(params); }},
        {"Mujoco_Ant_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Ant_v4(params); }},
        {"Mujoco_Inverted_Pendulum_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Inverted_Pendulum_v4(params); }},
        {"Mujoco_Inverted_Double_Pendulum_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Inverted_Double_Pendulum_v4(params); }},
        {"Mujoco_Half_Cheetah_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Half_Cheetah_v4(params); }},
        {"Mujoco_Half_Cheetah_Hurdles_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Half_Cheetah_Hurdles_v4(params); }},
        {"Mujoco_Reacher_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Reacher_v4(params); }},
        {"Mujoco_Hopper_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Hopper_v4(params); }},
        {"Mujoco_Humanoid_Standup_v4", [](std::unordered_map<std::string, std::any>& params) { return new Mujoco_Humanoid_Standup_v4(params); }},
        {"XPredPrey", [](std::unordered_map<std::string, std::any>& params) { return new XPredPreyTask(params); }},
    };

    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second(params);
    }
    return nullptr;
}
