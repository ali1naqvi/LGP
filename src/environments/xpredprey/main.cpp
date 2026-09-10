#include "XPredPrey.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int steps = 2000;
    int seed = 1;
    float predator_left = 1.0F;
    float predator_right = 1.0F;
    float prey_left = 0.0F;
    float prey_right = 0.0F;
    std::filesystem::path assets;
};

int parse_int(const char* value, const char* option) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + value);
    }
}

float parse_action(const char* value, const char* option) {
    try {
        return std::clamp(std::stof(value), -1.0F, 1.0F);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid action for ") + option + ": " + value);
    }
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Runs one native C++ xpredprey simulation episode.\n\n"
        << "Options:\n"
        << "  --steps N              Maximum simulation steps (default: 2000)\n"
        << "  --seed N               Random seed (default: 1)\n"
        << "  --predator LEFT RIGHT  Predator wheel actions in [-1, 1] (default: 1 1)\n"
        << "  --prey LEFT RIGHT      Prey wheel actions in [-1, 1] (default: 0 0)\n"
        << "  --assets PATH          Directory containing the MarXBot .sample files\n"
        << "  -h, --help             Show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    options.assets = std::filesystem::absolute(argv[0]).parent_path() / "assets";

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            print_help(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--steps" && i + 1 < argc) {
            options.steps = parse_int(argv[++i], "--steps");
        } else if (argument == "--seed" && i + 1 < argc) {
            options.seed = parse_int(argv[++i], "--seed");
        } else if (argument == "--assets" && i + 1 < argc) {
            options.assets = argv[++i];
        } else if (argument == "--predator" && i + 2 < argc) {
            options.predator_left = parse_action(argv[++i], "--predator");
            options.predator_right = parse_action(argv[++i], "--predator");
        } else if (argument == "--prey" && i + 2 < argc) {
            options.prey_left = parse_action(argv[++i], "--prey");
            options.prey_right = parse_action(argv[++i], "--prey");
        } else {
            throw std::runtime_error("unknown or incomplete option: " + argument);
        }
    }

    if (options.steps <= 0) {
        throw std::runtime_error("--steps must be greater than zero");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!std::filesystem::is_directory(options.assets)) {
            throw std::runtime_error("asset directory not found: " + options.assets.string());
        }
        for (const char* filename : {
                 "marxbot-wall.sample",
                 "marxbot-scylinder.sample",
                 "marxbot-cylinder.sample",
             }) {
            if (!std::filesystem::is_regular_file(options.assets / filename)) {
                throw std::runtime_error("required asset not found: " +
                                         (options.assets / filename).string());
            }
        }

        XPredPrey environment(options.steps, options.assets.string());
        environment.seed(options.seed);

        std::vector<float> observations(static_cast<std::size_t>(environment.ninputs) * 2U);
        std::vector<float> actions{
            options.predator_left,
            options.predator_right,
            options.prey_left,
            options.prey_right,
        };
        int done = 0;

        environment.copyObs(observations.data());
        environment.copyAct(actions.data());
        environment.copyDone(&done);
        environment.reset();

        double reward = 0.0;
        int completed_steps = 0;
        while (!done && completed_steps < options.steps) {
            reward = environment.step();
            ++completed_steps;
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "result"
                  << " seed=" << options.seed
                  << " steps=" << completed_steps
                  << " captured=" << (environment.isCaptured() ? "true" : "false")
                  << " predator_reward=" << reward
                  << " inputs_per_robot=" << environment.ninputs
                  << " outputs_per_robot=" << environment.noutputs
                  << '\n';
        environment.close();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "xpredprey: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
