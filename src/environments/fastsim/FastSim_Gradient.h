#ifndef FASTSIM_GRADIENT_H
#define FASTSIM_GRADIENT_H

#include <FastSimEnvironment.h>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <any>
#include <iostream>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <vector>
#include <array>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <limits>
#include <string>
#include <cctype>

namespace fs = std::filesystem;

class FastSim_Gradient : public FastSimEnv {
public:
    double grid_size_   = 50.0; 
    double peak_height_ = 80.0; 
    double spread_      = 35.0; 
    double plateau_     = -1.0; 
    double scale_       = 1.0;   // pixels per logical unit (pixel_w / grid_size)
    static constexpr float action_step_pixels_ = 10.0f;

    // Optional deterministic Perlin landscape that replaces the normal cone. Keeping the
    // seed in the config (rather than drawing it from the episode RNG) makes
    // training, evaluation, images, and checkpoint replays use the same field.
    bool     perlin_enabled_     = false;
    uint32_t field_seed_         = 0;    // derived from seed_aux
    double   perlin_scale_       = 8.0;  // logical units per noise period
    double   perlin_strength_    = 0.35; // contrast around half peak height
    int      perlin_octaves_     = 4;
    static constexpr double perlin_persistence_ = 0.7;

    struct Peak {
        double px;
        double py;
        double height;
    };
    int n_peaks_ = 1;
    bool depletable_ = false;
    double extraction_rate_ = 0.2;
    std::vector<Peak> peaks_;
    // Fraction of the original resource remaining at visited pixels. Pixels
    // absent from the map retain their full resource level (1.0).
    std::unordered_map<uint64_t, double> resource_fraction_;

    double peak_px_ = 0.0;
    double peak_py_ = 0.0;

    bool   reward_obs_ = false;  // expose previous-step reward as an extra obs
    int    resource_obs_count_ = 1; // obs[0] is current; obs[1..] look ahead

    double obs_noise_       = 0.0;
    bool   obs_noise_cue_   = false;
    std::mt19937 noise_rng_;

    double curr_conc_ = 0.0;
    double prev_conc_ = 0.0;
    double prev_step_reward_ = 0.0;
    double episode_return_ = 0.0;

    bool save_map_image_     = false;
    bool save_video_         = false;
    bool save_summed_image_  = false;

    static fs::path unique_path(const fs::path& dir,
                                const std::string& stem,
                                const std::string& ext,
                                int padding = 2){
        fs::path candidate = dir / (stem + ext);
        int index = 1;
        do {
            std::ostringstream oss;
            oss << stem << "_"
                << std::setw(padding) << std::setfill('0') << index
                << ext;
            candidate = dir / oss.str();
            ++index;
        } while (fs::exists(candidate));
        return candidate;
    }

    void set_baseline_mean(double &mean) override {
        if (step_ == 0) mean = 0;
        return;
    }

    FastSim_Gradient(std::unordered_map<std::string, std::any>& params)
    {
        eval_type_ = "Gradient";
        n_eval_train_      = std::any_cast<int>(params["gradient_n_eval_train"]);
        n_eval_validation_ = std::any_cast<int>(params["gradient_n_eval_validation"]);
        n_eval_test_       = std::any_cast<int>(params["gradient_n_eval_test"]);
        sticky_walls_      = std::any_cast<int>(params["gradient_sticky_walls"]) == 1;
        replay_            = std::any_cast<int>(params["replay"]) == 1;
        simple_discrete_   = std::any_cast<int>(params["gradient_discrete"]) == 1;

        save_map_image_    = std::any_cast<int>(params["gradient_save_image"]) == 1;
        save_video_        = std::any_cast<int>(params["gradient_save_video"]) == 1;
        save_summed_image_ = std::any_cast<int>(params["gradient_save_summed_image"]) == 1;
        reward_obs_        = std::any_cast<int>(params["gradient_reward_obs"]) == 1;

        obs_noise_ = param_as_double(params, "gradient_obs_noise", 0.0);
        if (params.find("gradient_obs_noise_cue") != params.end()) {
            try {
                obs_noise_cue_ = std::any_cast<int>(params["gradient_obs_noise_cue"]) == 1;
            } catch (const std::bad_any_cast&) { obs_noise_cue_ = false; }
        }

        peak_height_ = param_as_double(params, "gradient_peak_height", 80.0);
        spread_      = param_as_double(params, "gradient_spread",      35.0);
        plateau_     = param_as_double(params, "gradient_plateau",     -1.0);
        grid_size_   = param_as_double(params, "gradient_grid_size",   50.0);

        perlin_enabled_     = param_as_int(params, "gradient_perlin_noise", 0) == 1;
        field_seed_         = static_cast<uint32_t>(param_as_int(params, "seed_aux", 42));
        perlin_scale_       = param_as_double(params, "gradient_perlin_scale", 8.0);
        perlin_strength_    = param_as_double(params, "gradient_perlin_strength", 0.35);
        perlin_octaves_     = param_as_int(params, "gradient_perlin_octaves", 4);
        n_peaks_            = param_as_int(params, "gradient_n_peaks", 1);
        depletable_         = param_as_int(params, "gradient_depletable", 0) == 1;
        extraction_rate_    = param_as_double(params, "gradient_extraction_rate", 0.2);

        if (spread_ <= 0.0) spread_ = 1.0;
        if (grid_size_ <= 0.0) grid_size_ = 50.0;
        if (perlin_scale_ <= 0.0) perlin_scale_ = 8.0;
        perlin_strength_ = std::max(0.0, perlin_strength_);
        perlin_octaves_ = std::clamp(perlin_octaves_, 1, 12);
        n_peaks_ = std::max(1, n_peaks_);
        extraction_rate_ = std::clamp(extraction_rate_, 0.0, 1.0);

        model_path_ = ExpandEnvVars(std::any_cast<std::string>(params["gradient_model_path"])) + "gradient.xml";
        max_step_   = std::any_cast<int>(params["gradient_max_timestep"]);

        initialize_simulation();

        scale_  = static_cast<double>(map_->get_pixel_w()) / grid_size_;
        peak_px_ = map_->get_pixel_w() / 2.0;
        peak_py_ = map_->get_pixel_h() / 2.0;
        initialize_peaks();
        reset_resources();

        path_.clear();
        update_observation_size();
        state_.resize(obs_size_);
    }

    ~FastSim_Gradient(){}

    void SetResourceObservationCount(int count) override {
        resource_obs_count_ = std::max(1, count);
        update_observation_size();
        state_.resize(obs_size_);
        if (robot_ && map_) get_obs(state_);
    }

    bool terminal() override {
        return step_ >= max_step_;
    }

    Results sim_step(int& a) override {
        std::vector<double> temp{ static_cast<double>(a) };
        return sim_step(temp);
    }

    Results sim_step(std::vector<double>& action) override
    {
        if (simple_discrete_) {
            int dir = static_cast<int>(std::round(action[0]));
            dir = std::clamp(dir, 0, 4);

            if (dir != 0) {
                float theta_old = robot_->get_pos().theta();
                static constexpr float DTHETA[4] = { 0, M_PI/2, M_PI, -M_PI/2 };
                float theta_new = fastsim::Posture::normalize_angle(theta_old + DTHETA[dir - 1]);

                float dx = action_step_pixels_ * std::cos(theta_new);
                float dy = action_step_pixels_ * std::sin(theta_new);
                int new_x = int(robot_->get_pos().get_x() + dx);
                int new_y = int(robot_->get_pos().get_y() + dy);

                const int r = robot_->get_radius();
                auto free_circle = [&](int cx, int cy) {
                    if (cx < 0 || cy < 0 || cx >= map_->get_pixel_w() || cy >= map_->get_pixel_h()) return false;
                    for (int ddy = -r; ddy <= r; ++ddy)
                        for (int ddx = -r; ddx <= r; ++ddx) {
                            if (ddx*ddx + ddy*ddy > r*r) continue;
                            int tx = cx + ddx, ty = cy + ddy;
                            if (tx < 0 || ty < 0 || tx >= map_->get_pixel_w() || ty >= map_->get_pixel_h()) return false;
                            if (map_->get_pixel(tx, ty) == fastsim::Map::obstacle) return false;
                        }
                    return true;
                };

                if (free_circle(new_x, new_y)) {
                    robot_->set_pos(fastsim::Posture(new_x, new_y, theta_new));
                    robot_->move(0.f, 0.f, map_, sticky_walls_);
                }
            }
        } else {
            auto clean_action = [](const std::vector<double>& values, size_t idx) {
                double v = idx < values.size() ? values[idx] : 0.0;
                if (!std::isfinite(v)) v = 0.0;
                return std::clamp(v, -1.0, 1.0);
            };
            double turn = clean_action(action, 0);
            double forward = (clean_action(action, 1) + 1.0) * 0.5;
            // Match the discrete action's 10-pixel step. With the default
            // 600-pixel arena and 60-unit grid, this is one logical cell.
            float left = action_step_pixels_ *
                         static_cast<float>(std::clamp(forward - turn, -1.0, 1.0));
            float right = action_step_pixels_ *
                          static_cast<float>(std::clamp(forward + turn, -1.0, 1.0));
            robot_->move(left, right, map_, sticky_walls_);
        }

        if (save_video_) {
            fs::path frames_dir = fs::current_path() / "frames";
            std::ostringstream oss;
            oss << std::setw(5) << std::setfill('0') << step_;
            auto frame_path = frames_dir / ("frame_" + oss.str() + ".ppm");
            save_map_image(frame_path.string());
        }
        step_++;

        // Reward is the resource actually extracted, rather than the full
        // concentration of a cell. This conserves the finite resource while
        // allowing a location to be harvested gradually over several visits.
        double step_reward = concentration_at(robot_->get_pos().get_x(), robot_->get_pos().get_y());
        if (depletable_) {
            step_reward *= extraction_rate_;
            deplete_at(robot_->get_pos().get_x(), robot_->get_pos().get_y());
        }
        curr_conc_ = concentration_at(robot_->get_pos().get_x(), robot_->get_pos().get_y());
        prev_step_reward_ = step_reward;
        prev_conc_ = curr_conc_;

        if (replay_) {
            path_.push_back({static_cast<int>(robot_->get_pos().get_x()),
                             static_cast<int>(robot_->get_pos().get_y())});
        }

        bool done = step_ >= max_step_;
        episode_return_ += step_reward;

        if (done) {
            if (save_video_) {
                // -pix_fmt yuv420p is required for the h264 stream to play in
                // QuickTime / browsers; without it libx264 emits yuv444p which
                // most players refuse to open. -vf pad keeps width/height even.
                std::string cmd =
                    "ffmpeg -y -framerate 10 -i frames/frame_%05d.ppm "
                    "-vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" "
                    "-c:v libx264 -pix_fmt yuv420p gradient_video.mp4";
                int ret = std::system(cmd.c_str());
                if (ret == 0) {
                    std::cout << "Gradient video saved to "
                              << (fs::current_path() / "gradient_video.mp4").string()
                              << std::endl;
                    fs::remove_all(fs::current_path() / "frames");
                } else {
                    std::cerr << "ffmpeg failed (code " << ret
                              << "); leaving frames/ in place for inspection."
                              << std::endl;
                }
            } else if (save_map_image_) {
                fs::path save_dir = fs::current_path();
                auto snap_path = unique_path(save_dir, "gradient_reset", ".ppm", 3);
                save_map_image(snap_path.string());
            }
            if (save_summed_image_) {
                fs::path summed_path = fs::current_path() / "summed_positions.ppm";
                save_summed_positions(summed_path.string());
            }
            return {step_reward, 0.0};
        }
        get_obs(state_);
        return {step_reward, 0.0};
    }

    void reset(std::mt19937& rng, int& episode_number) override
    {
        episode_return_ = 0.0;
        step_ = 0;
        prev_step_reward_ = 0.0;
        path_.clear();
        map_->clear_goals();
        reset_resources();

        // Seed the per-episode noise stream from the main RNG so sensory noise
        // is reproducible for a given experiment seed.
        noise_rng_.seed(rng());

        spawn_on_border(rng);

        // The stationary energy field peaks at the centre; it is not a goal.

        if (robot_ && map_) {
            robot_->move(0.f, 0.f, map_, sticky_walls_);
        }

        prev_conc_ = concentration_at(robot_->get_pos().get_x(), robot_->get_pos().get_y());
        curr_conc_ = prev_conc_;

        if (replay_)
            path_.push_back({static_cast<int>(robot_->get_pos().get_x()),
                             static_cast<int>(robot_->get_pos().get_y())});

        if (save_video_) {
            fs::path frames_dir = fs::current_path() / "frames";
            if (fs::exists(frames_dir)) fs::remove_all(frames_dir);
            fs::create_directory(frames_dir);
            save_map_image((frames_dir / "frame_00000.ppm").string());
        }

        get_obs(state_);
    }

    void get_obs(std::vector<double>& obs)
    {
        // obs[0] is the resource at the robot. Each subsequent resource
        // channel is one logical grid cell farther along its current heading.
        size_t idx = 0;
        const double x = robot_->get_pos().get_x();
        const double y = robot_->get_pos().get_y();
        const double heading = robot_->get_pos().theta();
        std::normal_distribution<double> noise(0.0, obs_noise_);
        for (int offset = 0; offset < resource_obs_count_; ++offset) {
            double sensed = 0.0;
            const double sample_x = x + offset * scale_ * std::cos(heading);
            const double sample_y = y + offset * scale_ * std::sin(heading);
            if (sample_x >= 0.0 && sample_x < map_->get_pixel_w() &&
                sample_y >= 0.0 && sample_y < map_->get_pixel_h()) {
                sensed = (offset == 0 ? curr_conc_
                                      : concentration_at(sample_x, sample_y)) /
                         peak_height_;
            }
            if (obs_noise_ > 0.0) sensed += noise(noise_rng_);
            obs[idx++] = std::clamp(sensed, 0.0, 1.0);
        }
        if (reward_obs_) {
            obs[idx++] = std::clamp(prev_step_reward_ / peak_height_, 0.0, 1.0);
        }
        if (obs_noise_cue_) {
            // Perceivable cue: the current noise level (clamped to [0,1]).
            obs[idx++] = std::clamp(obs_noise_, 0.0, 1.0);
        }
    }

    int GetObsSize() { return obs_size_; }

    // Map a normalised concentration in [0,1] to a colour. Uses a
    // perceptually-uniform "viridis" ramp (dark purple -> blue -> teal ->
    // green -> yellow) where brightness increases with concentration. This
    // reads as a smooth gradient instead of the over-saturated banding a
    // jet/rainbow map produces.
    static inline void heat_color(double t, unsigned char& r,
                                  unsigned char& g, unsigned char& b) {
        t = std::clamp(t, 0.0, 1.0);
        struct Stop { double p, r, g, b; };
        static const Stop stops[6] = {
            {0.00,  68,   1,  84},   // dark purple (lowest concentration)
            {0.20,  72,  40, 120},
            {0.40,  59,  82, 139},   // blue
            {0.60,  33, 145, 140},   // teal
            {0.80,  94, 201,  98},   // green
            {1.00, 253, 231,  37},   // yellow      (peak)
        };
        for (int i = 0; i < 5; ++i) {
            if (t <= stops[i + 1].p) {
                double span = stops[i + 1].p - stops[i].p;
                double f = span > 0.0 ? (t - stops[i].p) / span : 0.0;
                r = static_cast<unsigned char>(stops[i].r + f * (stops[i + 1].r - stops[i].r));
                g = static_cast<unsigned char>(stops[i].g + f * (stops[i + 1].g - stops[i].g));
                b = static_cast<unsigned char>(stops[i].b + f * (stops[i + 1].b - stops[i].b));
                return;
            }
        }
        r = static_cast<unsigned char>(stops[5].r);
        g = static_cast<unsigned char>(stops[5].g);
        b = static_cast<unsigned char>(stops[5].b);
    }

    void save_map_image(const std::string& filename) const
    {
        const int width  = map_->get_pixel_w();
        const int height = map_->get_pixel_h();
        std::ofstream ofs(filename, std::ios::binary);

        const int rx = static_cast<int>(robot_->get_pos().get_x());
        const int ry = static_cast<int>(robot_->get_pos().get_y());

        auto encode = [](int x, int y) -> uint64_t {
            return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
                   static_cast<uint32_t>(y);
        };

        std::unordered_set<uint64_t> path_set;
        if (replay_) {
            const int trail_radius = 2;
            for (const auto& p : path_) {
                for (int dy = -trail_radius; dy <= trail_radius; ++dy)
                    for (int dx = -trail_radius; dx <= trail_radius; ++dx) {
                        if (dx*dx + dy*dy > trail_radius*trail_radius) continue;
                        path_set.insert(encode(p.first + dx, p.second + dy));
                    }
            }
        }

        // Heading line so the agent's orientation is visible in the video.
        const int robot_r = static_cast<int>(robot_->get_radius());
        const double va = robot_->get_pos().theta();
        const int line_length = robot_r * 2;
        const int line_radius = 1;
        std::unordered_set<uint64_t> line_set;
        for (int i = 0; i < line_length; ++i) {
            int lx = static_cast<int>(std::round(rx + std::cos(va) * i));
            int ly = static_cast<int>(std::round(ry + std::sin(va) * i));
            for (int dy = -line_radius; dy <= line_radius; ++dy)
                for (int dx = -line_radius; dx <= line_radius; ++dx) {
                    if (dx*dx + dy*dy > line_radius*line_radius) continue;
                    int nx = lx + dx, ny = ly + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        line_set.insert(encode(nx, ny));
                }
        }

        ofs << "P6\n" << width << " " << height << "\n255\n";

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                unsigned char r = 0, g = 0, b = 0;
                const bool in_robot = ((x - rx)*(x - rx) + (y - ry)*(y - ry)) <= (robot_r * robot_r);
                const bool on_line  = line_set.count(encode(x, y)) > 0;
                const bool in_path  = replay_ && (path_set.count(encode(x, y)) > 0);

                if (map_->get_pixel(x, y) == fastsim::Map::obstacle) {
                    r = g = b = 0;                 // wall
                } else if (in_robot) {
                    r = g = b = 255;               // robot (white, stands out)
                } else if (on_line) {
                    r = 0; g = 0; b = 0;           // heading line (black)
                } else if (in_path) {
                    r = 255; g = 0; b = 255;       // trail (magenta)
                } else {
                    // Render the gradient field as a vivid heat-map.
                    double norm = std::clamp(concentration_at(x, y) / peak_height_, 0.0, 1.0);
                    heat_color(norm, r, g, b);
                }
                ofs.write(reinterpret_cast<char*>(&r), 1);
                ofs.write(reinterpret_cast<char*>(&g), 1);
                ofs.write(reinterpret_cast<char*>(&b), 1);
            }
        }
    }

private:
    void update_observation_size() {
        obs_size_ = resource_obs_count_ + (reward_obs_ ? 1 : 0) +
                    (obs_noise_cue_ ? 1 : 0);
    }

    static double param_as_double(std::unordered_map<std::string, std::any>& params,
                                  const std::string& key, double fallback) {
        auto it = params.find(key);
        if (it == params.end()) return fallback;
        try { return std::any_cast<double>(it->second); } catch (const std::bad_any_cast&) {}
        try { return static_cast<double>(std::any_cast<int>(it->second)); } catch (const std::bad_any_cast&) {}
        return fallback;
    }

    static int param_as_int(std::unordered_map<std::string, std::any>& params,
                            const std::string& key, int fallback) {
        auto it = params.find(key);
        if (it == params.end()) return fallback;
        try { return std::any_cast<int>(it->second); } catch (const std::bad_any_cast&) {}
        try { return static_cast<int>(std::any_cast<double>(it->second)); } catch (const std::bad_any_cast&) {}
        return fallback;
    }

    // Parse an Avida-style GRADIENT_RESOURCE spec, e.g.
    // "food0:height=80:spread=35:plateau=-1:peakx=25:peaky=25". Only the cone
    // shape fields are used; movement / inflow fields are ignored (peak is
    // stationary and centred).
    void parse_gradient_resource(const std::string& spec) {
        std::stringstream ss(spec);
        std::string token;
        while (std::getline(ss, token, ':')) {
            auto eq = token.find('=');
            if (eq == std::string::npos) continue; // e.g. the leading name
            std::string key = token.substr(0, eq);
            double val = 0.0;
            try { val = std::stod(token.substr(eq + 1)); } catch (...) { continue; }
            if (key == "height")      peak_height_ = val;
            else if (key == "spread") spread_      = val;
            else if (key == "plateau") plateau_    = val;
        }
    }

    static double perlin_fade(double t) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    static double perlin_lerp(double a, double b, double t) {
        return a + t * (b - a);
    }

    uint32_t perlin_hash(int x, int y) const {
        uint32_t h = static_cast<uint32_t>(x) * 0x8da6b343u;
        h ^= static_cast<uint32_t>(y) * 0xd8163841u;
        h ^= field_seed_ * 0xcb1ab31fu;
        h ^= h >> 16;
        h *= 0x7feb352du;
        h ^= h >> 15;
        h *= 0x846ca68bu;
        return h ^ (h >> 16);
    }

    double perlin_gradient_dot(int ix, int iy, double x, double y) const {
        // Eight evenly distributed, deterministic gradient directions.
        static constexpr double diagonal = 0.7071067811865475;
        static constexpr double gradients[8][2] = {
            {1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0},
            {diagonal, diagonal}, {-diagonal, diagonal},
            {diagonal, -diagonal}, {-diagonal, -diagonal}
        };
        const auto& gradient = gradients[perlin_hash(ix, iy) & 7u];
        return gradient[0] * (x - ix) + gradient[1] * (y - iy);
    }

    double perlin_noise(double x, double y) const {
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const double u = perlin_fade(x - x0);
        const double v = perlin_fade(y - y0);

        const double top = perlin_lerp(perlin_gradient_dot(x0, y0, x, y),
                                       perlin_gradient_dot(x1, y0, x, y), u);
        const double bottom = perlin_lerp(perlin_gradient_dot(x0, y1, x, y),
                                          perlin_gradient_dot(x1, y1, x, y), u);
        return perlin_lerp(top, bottom, v);
    }

    double fractal_perlin(double x, double y) const {
        double value = 0.0;
        double amplitude = 1.0;
        double amplitude_sum = 0.0;
        double frequency = 1.0;
        for (int octave = 0; octave < perlin_octaves_; ++octave) {
            value += amplitude * perlin_noise(x * frequency, y * frequency);
            amplitude_sum += amplitude;
            amplitude *= perlin_persistence_;
            frequency *= 2.0;
        }
        return amplitude_sum > 0.0 ? value / amplitude_sum : 0.0;
    }

    void initialize_peaks() {
        peaks_.clear();
        if (n_peaks_ == 1) {
            peaks_.push_back({peak_px_, peak_py_, peak_height_});
            return;
        }

        // Multiple peaks are evenly separated on a ring. seed_aux rotates the
        // layout without changing its geometry, and therefore replaying with
        // the same auxiliary seed reproduces the exact landscape.
        std::mt19937 layout_rng(field_seed_);
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
        const double angle_offset = angle_dist(layout_rng);
        const double radius = 0.30 * std::min(map_->get_pixel_w(), map_->get_pixel_h());
        for (int i = 0; i < n_peaks_; ++i) {
            const double angle = angle_offset + 2.0 * M_PI * i / n_peaks_;
            const double height = i == 0 ? peak_height_ : peak_height_ * 0.8;
            peaks_.push_back({peak_px_ + radius * std::cos(angle),
                              peak_py_ + radius * std::sin(angle),
                              height});
        }
    }

    double undepleted_concentration_at(double px, double py) const {
        double dx = (px - peak_px_) / scale_; // logical units
        double dy = (py - peak_py_) / scale_;
        if (perlin_enabled_) {
            // The seed affects lattice hashes independently of spawn or episode
            // RNG state. Strength controls contrast around half peak height.
            const double nx = (dx + grid_size_ * 0.5) / perlin_scale_;
            const double ny = (dy + grid_size_ * 0.5) / perlin_scale_;
            const double normalized = std::clamp(
                0.5 + perlin_strength_ * fractal_perlin(nx, ny), 0.0, 1.0);
            return peak_height_ * normalized;
        }

        double concentration = 0.0;
        for (const auto& peak : peaks_) {
            const double peak_dx = (px - peak.px) / scale_;
            const double peak_dy = (py - peak.py) / scale_;
            const double dist = std::sqrt(peak_dx*peak_dx + peak_dy*peak_dy);
            const double value = (plateau_ >= 0.0 && dist <= plateau_)
                ? peak.height
                : std::max(peak.height * (1.0 - dist / spread_), 0.0);
            // Taking the maximum keeps overlapping peaks from artificially
            // raising the valleys between them.
            concentration = std::max(concentration, value);
        }
        return concentration;
    }

    double concentration_at(double px, double py) const {
        const int x = std::clamp(static_cast<int>(std::round(px)),
                                 0, map_->get_pixel_w() - 1);
        const int y = std::clamp(static_cast<int>(std::round(py)),
                                 0, map_->get_pixel_h() - 1);
        double remaining_fraction = 1.0;
        if (depletable_) {
            auto it = resource_fraction_.find(cell_key(x, y));
            if (it != resource_fraction_.end()) {
                remaining_fraction = it->second;
            }
        }
        return undepleted_concentration_at(px, py) * remaining_fraction;
    }

    void reset_resources() {
        resource_fraction_.clear();
    }

    void deplete_at(double px, double py) {
        if (!depletable_) return;
        const int width = map_->get_pixel_w();
        const int height = map_->get_pixel_h();
        const int cx = static_cast<int>(std::round(px));
        const int cy = static_cast<int>(std::round(py));
        const int radius = std::max(1, static_cast<int>(robot_->get_radius()));
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx*dx + dy*dy > radius*radius) continue;
                const int x = cx + dx;
                const int y = cy + dy;
                if (x < 0 || x >= width || y < 0 || y >= height) continue;
                const uint64_t key = cell_key(x, y);
                auto it = resource_fraction_.try_emplace(key, 1.0).first;
                it->second *= (1.0 - extraction_rate_);
            }
        }
    }

    static uint64_t cell_key(int x, int y) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
               static_cast<uint32_t>(y);
    }

    // Spawn the agent at a random free point along the inner edge of the arena.
    void spawn_on_border(std::mt19937& rng) {
        const int W = map_->get_pixel_w();
        const int H = map_->get_pixel_h();
        const int rr = robot_->get_radius();
        const int margin = rr + 10; // clear of the border wall

        std::uniform_int_distribution<int> side(0, 3);
        std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * M_PI);

        for (int tries = 0; tries < 400; ++tries) {
            int sx = 0, sy = 0;
            switch (side(rng)) {
                case 0: sx = std::uniform_int_distribution<int>(margin, W - margin)(rng); sy = margin;       break; // top
                case 1: sx = std::uniform_int_distribution<int>(margin, W - margin)(rng); sy = H - margin;   break; // bottom
                case 2: sx = margin;       sy = std::uniform_int_distribution<int>(margin, H - margin)(rng); break; // left
                default: sx = W - margin;  sy = std::uniform_int_distribution<int>(margin, H - margin)(rng); break; // right
            }
            if (map_->get_pixel(sx, sy) != fastsim::Map::obstacle) {
                robot_->set_pos(fastsim::Posture(sx, sy, theta_dist(rng)));
                return;
            }
        }
        // Fallback: top-left interior corner.
        robot_->set_pos(fastsim::Posture(margin, margin, 0));
    }

    void save_summed_positions(const std::string& filename) const
    {
        const int width  = map_->get_pixel_w();
        const int height = map_->get_pixel_h();
        std::vector<unsigned char> buffer(width * height * 3, 255);

        std::ifstream ifs(filename, std::ios::binary);
        if (ifs.good()) {
            std::string header; int w, h, maxval;
            ifs >> header >> w >> h >> maxval; ifs.ignore(1);
            if (w == width && h == height)
                ifs.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        }
        ifs.close();

        if (!fs::exists(filename)) {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    int pidx = (y * width + x) * 3;
                    if (map_->get_pixel(x, y) == fastsim::Map::obstacle) {
                        buffer[pidx] = buffer[pidx+1] = buffer[pidx+2] = 0;
                    } else {
                        double norm = std::clamp(concentration_at(x, y) / peak_height_, 0.0, 1.0);
                        buffer[pidx]   = static_cast<unsigned char>(255 * (1.0 - norm));
                        buffer[pidx+1] = 255;
                        buffer[pidx+2] = static_cast<unsigned char>(255 * (1.0 - norm));
                    }
                }
        }

        int rx = static_cast<int>(robot_->get_pos().get_x());
        int ry = static_cast<int>(robot_->get_pos().get_y());
        const int mark_radius = 6;
        for (int dy = -mark_radius; dy <= mark_radius; ++dy)
            for (int dx = -mark_radius; dx <= mark_radius; ++dx) {
                if (dx*dx + dy*dy > mark_radius*mark_radius) continue;
                int px = rx + dx, py = ry + dy;
                if (px < 0 || px >= width || py < 0 || py >= height) continue;
                int pidx = (py * width + px) * 3;
                buffer[pidx]   = 255;
                buffer[pidx+1] = static_cast<unsigned char>(buffer[pidx+1] / 2);
                buffer[pidx+2] = static_cast<unsigned char>(buffer[pidx+2] / 2);
            }

        std::ofstream ofs(filename, std::ios::binary);
        ofs << "P6\n" << width << " " << height << "\n255\n";
        ofs.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
    }
};

#endif
