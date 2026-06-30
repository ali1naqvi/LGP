#ifndef FASTSIM_MAZE_H
#define FASTSIM_MAZE_H

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
#include <cstdlib>      // for system()
#include <limits>
#include <string>
#include <cctype>

namespace fs = std::filesystem;

class FastSim_Maze : public FastSimEnv {
public:
    double total = 0.0;
    double curr_distance = 0.0; 
    double prev_distance = 0.0; 
    fastsim::Posture initial_pose;
    bool actions_flipped_ = false;
    std::array<int, 4> action_mapping_;
    bool flip_episode_ = false;   
    int  flip_step_     = -1; 
    bool save_map_image_ = false;
    bool maze_reward_obs_ = false;
    bool save_video_ = false;
    bool save_summed_image_ = false;

    // Select which coloured start box to use ("red", "orange", "blue", "purple", "red_br" = lightblue or "random")
    std::string start_color_ = "red"; // default keeps current behaviour (top-left red box)

    // bool sequential_position_ = false;
    double prev_step_reward_ = 0.0; // for observation
    size_t seq_index_ = 0;            // next start position to use
    size_t seq_completed_ = 0;        // how many sequential starts have been evaluated in the current sweep
    int start_x_ = 0, start_y_ = 0;   // starting pixel used for the current episode
    double episode_return_ = 0.0;     // cumulative fitness of the current episode

    std::vector<std::pair<int,int>> seq_positions_;   // all viable sequential start pixels

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

    void set_baseline_mean(double &mean) {
        if (step_ == 0){
            // prev_distance = sqrt(pow(robot_->get_pos().get_x() - goal_pos_.get_x(),2) + pow(robot_->get_pos().get_y() - goal_pos_.get_y(),2)); 
            mean = 0;
        }
        return;
    }


    FastSim_Maze(std::unordered_map<std::string, std::any>& params)
    {
        eval_type_ = "Maze";
        n_eval_train_ = std::any_cast<int>(params["maze_n_eval_train"]);
        n_eval_validation_ = std::any_cast<int>(params["maze_n_eval_validation"]);
        n_eval_test_ = std::any_cast<int>(params["maze_n_eval_test"]);
        sticky_walls_ = std::any_cast<int>(params["maze_sticky_walls"]) == 1;
        replay_ = std::any_cast<int>(params["replay"]) == 1;
        simple_discrete_ = std::any_cast<int>(params["maze_discrete"]) == 1;

        dynamic_maze_ = std::any_cast<int>(params["dynamic_maze"]) == 1;

        save_map_image_ = std::any_cast<int>(params["maze_save_image"]) == 1;
        save_video_     = std::any_cast<int>(params["maze_save_video"]) == 1;
        save_summed_image_ = std::any_cast<int>(params["maze_save_summed_image"]) == 1;

        maze_reward_obs_ = std::any_cast<int>(params["maze_reward_obs"]) == 1;
        // sequential_position_ = std::any_cast<int>(params["sequential_position"]) == 1;

        // Optional: choose which coloured start box to spawn the agent in
        if (params.find("maze_start_color") != params.end()) {
            start_color_ = std::any_cast<std::string>(params["maze_start_color"]);
        } else if (params.find("maze_start_colour") != params.end()) { // allow UK spelling
            start_color_ = std::any_cast<std::string>(params["maze_start_colour"]);
        }
        // to lower-case
        std::transform(start_color_.begin(), start_color_.end(), start_color_.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

        model_path_ = ExpandEnvVars(std::any_cast<std::string>(params["maze_model_path"])) + "cuisine.xml";

        max_step_ = std::any_cast<int>(params["maze_max_timestep"]);
        initialize_simulation();         
        // if (sequential_position_) {
        //     rebuild_seq_positions();
        // }

        total = sqrt((pow(map_->get_pixel_w(),2) + pow(map_->get_pixel_h(),2)));
        path_.clear();
        if (maze_reward_obs_) {obs_size_ = 8;} else {obs_size_ = 7;} // 3 sensors, 4 radar.

        state_.resize(obs_size_);

        actions_flipped_ = false;
    }

    ~FastSim_Maze(){
    }

    bool terminal() override { 
        if (reached_goal()){
        return true;}
        return step_ >= max_step_;}

    Results sim_step(int& a) {
        std::vector<double> temp{ static_cast<double>(a) };
        return sim_step(temp);
    }

    Results sim_step(std::vector<double>& action)
    {   
        // cout << "step " << step_ << endl;
        if (dynamic_maze_ && !actions_flipped_ && flip_episode_ && step_ == flip_step_) {
            actions_flipped_ = true;
            // if (simple_discrete_) {
            //     std::array<int,4> before = {0,1,2,3};
            //     // std::cout << "[Dynamic] Actions switched at step " << step_
            //     //           << ": discrete mapping "
            //     //           << "[0->0, 1->1, 2->2, 3->3] -> ["
            //     //           << "0->" << action_mapping_[0] << ", "
            //     //           << "1->" << action_mapping_[1] << ", "
            //     //           << "2->" << action_mapping_[2] << ", "
            //     //           << "3->" << action_mapping_[3] << "]" << std::endl;
            // } else {
            //     // Report continuous swap of action order
            //     // std::cout << "[Dynamic] Actions switched at step " << step_
            //     //           << ": continuous order [a0, a1] -> [a1, a0]" << std::endl;
            // }
        }

        if (simple_discrete_) {
            int new_x = 0;
            int new_y = 0;
            float theta_new = 0;
            int dir_raw = static_cast<int>(std::round(action[0])); // edge case although always will be rounded
            int dir = actions_flipped_ ? action_mapping_[dir_raw] : dir_raw;
            const int step_pix = 3; // step size in pixels (MAKE SURE LESS THAN WALL = 6)

            float theta_old = robot_->get_pos().theta();
            static constexpr float DTHETA[4] = { 0,  M_PI/2, M_PI, -M_PI/2 };
            theta_new = fastsim::Posture::normalize_angle(theta_old + DTHETA[dir]);

            float dx = step_pix * std::cos(theta_new);
            float dy = step_pix * std::sin(theta_new);

            new_x = int(robot_->get_pos().get_x() + dx);
            new_y = int(robot_->get_pos().get_y() + dy);

            // Radius-aware free-space check
            const int r = robot_->get_radius();
            auto free_circle = [&](int cx, int cy) {
                if (cx < 0 || cy < 0 || cx >= map_->get_pixel_w() || cy >= map_->get_pixel_h()) return false;
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        if (dx*dx + dy*dy > r*r) continue;
                        int tx = cx + dx, ty = cy + dy;
                        if (tx < 0 || ty < 0 || tx >= map_->get_pixel_w() || ty >= map_->get_pixel_h()) return false;
                        if (map_->get_pixel(tx, ty) == fastsim::Map::obstacle) return false;
                    }
                }
                return true;
            };

            if (free_circle(new_x, new_y)) {
                robot_->set_pos(fastsim::Posture(new_x, new_y, theta_new));
                // Refresh sensors for the new pose
                robot_->move(0.f, 0.f, map_, sticky_walls_);
            }
            // // --- Debug: observation printout after discrete update ---
                // std::vector<double> obs_dbg(obs_size_);
                // get_obs(obs_dbg);
                // std::cerr << std::fixed << std::setprecision(3);
                // std::cerr << "[Obs] step=" << step_
                //           << " pos=(" << static_cast<int>(robot_->get_pos().get_x())
                //           << "," << static_cast<int>(robot_->get_pos().get_y())
                //           << ") theta=" << robot_->get_pos().theta()
                //           << " -> [";
                // for (size_t i = 0; i < obs_dbg.size(); ++i) {
                //     if (i) std::cerr << ", ";
                //     std::cerr << obs_dbg[i];
                // }
                // std::cerr << "]" << std::endl;
        } else {
        float a0 = static_cast<float>(action[0]);
        float a1 = static_cast<float>(action[1]);
        if (actions_flipped_) {
            std::swap(a0, a1);
        }
        // cout << "action: [" << a0 << ", " << a1 << " ]" << endl; 
        robot_->move(a0, a1, map_, sticky_walls_);
        {
            // float theta_new = fastsim::Posture::normalize_angle(std::atan2(a1, a0));
            // float theta_new = robot_->get_pos().theta();
            // auto pos = robot_->get_pos();
            // robot_->set_pos(fastsim::Posture(pos.get_x(),pos.get_y(), theta_new));
            // robot_->move(0.f, 0.f, map_, sticky_walls_);
                //             std::vector<double> obs_dbg(obs_size_);
                // get_obs(obs_dbg);
                // std::cerr << std::fixed << std::setprecision(3);
                // std::cerr << "[Obs] step=" << step_
                //           << " pos=(" << static_cast<int>(robot_->get_pos().get_x())
                //           << "," << static_cast<int>(robot_->get_pos().get_y())
                //           << ") theta=" << robot_->get_pos().theta()
                //           << " -> [";
                // for (size_t i = 0; i < obs_dbg.size(); ++i) {
                //     if (i) std::cerr << ", ";
                //     std::cerr << obs_dbg[i];
                // }
                // std::cerr << "]" << std::endl;
        }
        }
        if (save_video_) {
            fs::path frames_dir = fs::current_path() / "frames";
            std::ostringstream oss;
            oss << std::setw(5) << std::setfill('0') << step_;
            auto frame_path = frames_dir / ("frame_" + oss.str() + ".ppm");
            save_map_image(frame_path.string());
        }
        step_++;

        //reward calculation
        curr_distance = sqrt(pow(robot_->get_pos().get_x() - goal_pos_.get_x(),2) + pow(robot_->get_pos().get_y() - goal_pos_.get_y(),2)); 
        double step_reward = ((prev_distance - curr_distance) / total); // multiply all steps by 100
        
        prev_step_reward_ = step_reward;
        // std::cout << "step_reward (step " << step_ << "): " << step_reward << std::endl; //TODO(ali): step reward for maze

        prev_distance = curr_distance;
        // Paper is 1 - distance / total. However, this is deceptive
        
        if (replay_){
            path_.push_back({static_cast<int>(robot_->get_pos().get_x()),
                 static_cast<int>(robot_->get_pos().get_y())});
        }

        bool success = reached_goal();
        bool done = success || (step_ >= max_step_);
        double final_reward = step_reward;

        if (success) {
            final_reward += (max_step_ - step_) / max_step_;
            cout << "reached goal" << endl; //TODO(ali): comment
            // final_reward += 100.0;
        }
        episode_return_ += final_reward;

        if (done){
            if (save_video_) {
                std::string cmd = 
                    "ffmpeg -y -framerate 10 -i frames/frame_%05d.ppm "
                    "-c:v libx264 maze_video.mp4";
                std::system(cmd.c_str());
                fs::remove_all(fs::current_path() / "frames");
            }
            else if (save_map_image_) {
                fs::path save_dir = fs::current_path();
                auto snap_path = unique_path(save_dir, "maze_reset", ".ppm", 3);
                save_map_image(snap_path.string());
            }
            if (save_summed_image_) {
                fs::path summed_path = fs::current_path() / "summed_positions.ppm";
                save_summed_positions(summed_path.string());
            }
            return {final_reward, 0.0};
        }
        get_obs(state_);
        return {final_reward, 0.0};  
    }

    void reset(std::mt19937& rng, int& episode_number) override
    {
        episode_return_ = 0.0;
        step_ = 0;
        prev_step_reward_ = 0.0;
        actions_flipped_ = false;
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        flip_episode_ = dynamic_maze_ && (prob_dist(rng) < 0.9583);
        if (flip_episode_) {
            std::uniform_int_distribution<int> step_dist(0, 20); 
            flip_step_ = step_dist(rng);
        } else {
            flip_step_ = -1;
        }

        action_mapping_ = {0, 1, 2, 3};
        if (flip_episode_) {
            std::shuffle(action_mapping_.begin(), action_mapping_.end(), rng);
        }

        path_.clear();
        map_->clear_goals();
        // Try to spawn inside the requested coloured start box; if it fails, fall back to legacy box.
        if (!spawn_in_color_box(rng)) {
            std::uniform_int_distribution<> robot_box(30, 120);
            robot_->set_pos(fastsim::Posture(robot_box(rng), robot_box(rng), 0));
        }
        map_->add_goal(fastsim::Goal(550,306,20,0));
        // sample_free_space(rng); //first image generator
        if (replay_)
            path_.push_back({static_cast<int>(robot_->get_pos().get_x()),
                     static_cast<int>(robot_->get_pos().get_y())});
        goal_pos_ = map_->get_goals()[0];


        // Ensure sensors (lasers/radar) are initialized before first get_obs.
        // FastSim updates sensors during move(), so trigger a no-op move.
        if (robot_ && map_) {
            robot_->move(0.f, 0.f, map_, sticky_walls_);
        }
        prev_distance = sqrt(pow(robot_->get_pos().get_x() - goal_pos_.get_x(),2) +
                pow(robot_->get_pos().get_y() - goal_pos_.get_y(),2));

        if (save_video_) {
            fs::path frames_dir = fs::current_path() / "frames";
            if (fs::exists(frames_dir)) fs::remove_all(frames_dir);
            fs::create_directory(frames_dir);
            save_map_image((frames_dir / "frame_00000.ppm").string());
        }
        
        get_obs(state_);
        // cerr << "reset: step: " << step_ << " previous distance: " << prev_distance << " theta: " << robot_->get_pos().theta() << endl;
    }

    void get_obs(std::vector<double>& obs)
    {
        // Lasers are updated by Robot::move(). After reset we call a zero‑move to prime them.
        //FROM PAPER: Encouraging Creative Thinking in Robots Improves Their Ability to Solve Challenging Problems
        // Sensors include three range sensors, which provide the normalized distance to the nearest obstacle in the direction that
        // sensor points, and four pie-slice goal sensors that fire if the
        // goal is within their purview, irrespective of intervening obstacles, and thus serve as a compass toward the goal
        // const auto& p = robot_->get_pos();
        for (size_t i = 0; i < 3; ++i) {
            double d = robot_->get_lasers()[i].get_dist();
            const double maxr = 100.0; // matches XML <laser range="100"/>
            if (d < 0.0) {
                // No obstacle detected within sensor range: treat as max normalized distance.
                obs[i] = 1.0;
            } else {
                const double dc = std::min(std::max(d, 0.0), maxr);
                obs[i] = dc / maxr;
            }
        }
        int radar_slice = robot_->get_radars()[0].get_activated_slice();

        for (int i = 0; i < 4; ++i){
            obs[3 + i] = (radar_slice == i) ? 1.0 : 0.0;
        }
        if (maze_reward_obs_) {
            obs[7] = std::clamp(prev_step_reward_, -1.0, 1.0);;
        }

        // obs[7] = p.x();
        // obs[8] = p.y();
        // obs[9] = p.theta();
        // obs[10] = goal_pos_.get_x();
        // obs[11] = goal_pos_.get_y();
    }

    void save_map_image(const std::string& filename) const
    {   
        const int width  = map_->get_pixel_w();
        const int height = map_->get_pixel_h();

        std::ofstream ofs(filename, std::ios::binary);

        int rx = static_cast<int>(robot_->get_pos().get_x());
        int ry = static_cast<int>(robot_->get_pos().get_y());
        const int gx = goal_pos_.get_x();
        const int gy = goal_pos_.get_y();

        std::unordered_set<uint64_t> path_set;
        auto encode = [](int x, int y) -> uint64_t {
            return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
                   static_cast<uint32_t>(y);
        };
        if (replay_) {
            path_set.reserve(path_.size() * 9);
            const int trail_radius = 2; // make path points appear larger
            for (const auto& p : path_) {
                const int cx = static_cast<int>(p.first);
                const int cy = static_cast<int>(p.second);
                for (int dy = -trail_radius; dy <= trail_radius; ++dy) {
                    for (int dx = -trail_radius; dx <= trail_radius; ++dx) {
                        if (dx*dx + dy*dy > trail_radius*trail_radius) continue; // circle mask
                        const int nx = cx + dx;
                        const int ny = cy + dy;
                        if (nx < 0 || ny < 0) continue; // bounds checked later during write
                        path_set.insert(encode(nx, ny));
                    }
                }
            }
        }

        ofs << "P6\n" << width << " " << height << "\n255\n";

        const int robot_r = static_cast<int>(robot_->get_radius()); 
        const int goal_r  = static_cast<int>(goal_pos_.get_diam()) / 2;

        const double va = robot_->get_pos().theta();
        const int line_length = robot_r * 2;
        std::unordered_set<uint64_t> line_set;
        const int line_radius = 1; // thickness of heading line
        for (int i = 0; i < line_length; ++i) {
            int lx = static_cast<int>(std::round(robot_->get_pos().get_x() + std::cos(va) * i));
            int ly = static_cast<int>(std::round(robot_->get_pos().get_y() + std::sin(va) * i));
            if (lx >= 0 && lx < width && ly >= 0 && ly < height) {
                for (int dy = -line_radius; dy <= line_radius; ++dy) {
                    for (int dx = -line_radius; dx <= line_radius; ++dx) {
                        if (dx*dx + dy*dy > line_radius*line_radius) continue;
                        int nx = lx + dx;
                        int ny = ly + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            line_set.insert(encode(nx, ny));
                        }
                    }
                }
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                unsigned char r = 0, g = 0, b = 0;
                int dx_robot = x - rx;
                int dy_robot = y - ry;
                int dx_goal  = x - gx;
                int dy_goal  = y - gy;

                bool in_path = replay_ && (path_set.count(encode(x, y)) > 0);

                bool in_robot = (dx_robot * dx_robot + dy_robot * dy_robot) <= (robot_r * robot_r);
                bool in_goal  = (dx_goal  * dx_goal  + dy_goal  * dy_goal) <= (goal_r  * goal_r);

                bool on_line = (line_set.count(encode(x, y)) > 0);
                if (on_line) {
                    r = 255; g = 255; b = 0;
                } else
                if (in_robot)                { r=255; g=0;   b=0;   }    
                else if (in_goal)            { r=0;   g=255; b=0;   }                     else if (in_path)            
                { r=0;   g=0;   b=255; }     
                else if (map_->get_pixel(x,y) != fastsim::Map::obstacle) {
                    r = g = b = 255;                                    
                } else {
                    r = g = b = 0;                                      
                }

                ofs.write(reinterpret_cast<char*>(&r), 1);
                ofs.write(reinterpret_cast<char*>(&g), 1);
                ofs.write(reinterpret_cast<char*>(&b), 1);
            }
        }
    }

    int GetObsSize() { return obs_size_; }
private:
    // // Simple red->blue heat color map: norm=0→blue, norm=1→red
    static inline void to_heat(double norm, unsigned char &r, unsigned char &g, unsigned char &b) {
        norm = std::clamp(norm, 0.0, 1.0);
        r = static_cast<unsigned char>(255.0 * norm);      // low fitness -> blue, high -> red
        g = 0;
        b = static_cast<unsigned char>(255.0 * (1.0 - norm));
    }
    // Spawn the robot inside one of the coloured start boxes from the figures.
    // Returns true on success (position set), false if sampling failed (caller should fallback).
    bool spawn_in_color_box(std::mt19937 &rng) {
        if (!map_ || !robot_) return false;
        const int W = map_->get_pixel_w();
        const int H = map_->get_pixel_h();
        const int rr = robot_->get_radius();

        auto sample_rect = [&](double fx0, double fx1, double fy0, double fy1) -> bool {
            int x0 = std::clamp(static_cast<int>(std::round(fx0 * W)), rr + 1, W - rr - 1);
            int x1 = std::clamp(static_cast<int>(std::round(fx1 * W)), rr + 1, W - rr - 1);
            int y0 = std::clamp(static_cast<int>(std::round(fy0 * H)), rr + 1, H - rr - 1);
            int y1 = std::clamp(static_cast<int>(std::round(fy1 * H)), rr + 1, H - rr - 1);
            if (x1 < x0) std::swap(x0, x1);
            if (y1 < y0) std::swap(y0, y1);
            std::uniform_int_distribution<> dx(x0, x1);
            std::uniform_int_distribution<> dy(y0, y1);
            for (int tries = 0; tries < 200; ++tries) {
                int sx = dx(rng), sy = dy(rng);
                if (map_->get_pixel(sx, sy) != fastsim::Map::obstacle) {
                    robot_->set_pos(fastsim::Posture(sx, sy, 0));
                    return true;
                }
            }
            return false;
        };

        // Allow synonyms and a random choice
        std::string c = start_color_;
        if (c == "any") c = "random";
        if (c == "random") {
            static const std::array<const char*, 5> opts = {"red", "orange", "blue", "purple", "red_br"};
            std::uniform_int_distribution<int> pick(0, static_cast<int>(opts.size()) - 1);
            c = opts[pick(rng)];
        }

        // The following rectangles are given in fractions of W/H and
        // correspond to the coloured squares in the reference image.
        // "red"   : top-left     box (keeps legacy behaviour)
        // "orange": upper-left/centre-left
        // "blue"  : centre
        // "purple": mid-left
        // "red_br": bottom-right red box
        if (c == "red" || c == "red_tl") {
            if (sample_rect(0.05, 0.20, 0.06, 0.20)) return true; // top-left
        } else if (c == "orange") {
            if (sample_rect(0.25, 0.36, 0.23, 0.38)) return true; // upper-left/centre-left
        } else if (c == "blue") {
            if (sample_rect(0.43, 0.55, 0.32, 0.45)) return true; // centre
        } else if (c == "purple") {
            if (sample_rect(0.06, 0.16, 0.46, 0.60)) return true; // mid-left
        } else if (c == "red_br" || c == "bottom_right" || c == "red2") {
            if (sample_rect(0.83, 0.95, 0.70, 0.88)) return true; // bottom-right
        }
        return false;
    }
        bool reached_goal() const {
        double robot_r = static_cast<double>(robot_->get_radius());
        const double goal_r  = static_cast<double>(goal_pos_.get_diam()) / 2.0;

        const double dx = robot_->get_pos().get_x() - goal_pos_.get_x();
        const double dy = robot_->get_pos().get_y() - goal_pos_.get_y();
        const double dist2 = dx * dx + dy * dy;

        const double reach_dist = robot_r + goal_r;
        return dist2 <= reach_dist * reach_dist;
    }

    void sample_free_space(std::mt19937& rng) const
    {
        // borders will be at 0 and and at limit
        // constraint will be the absolute limit it should be
        const int width = map_->get_pixel_w();
        const int height = map_->get_pixel_h();
        const int margin = 6;
        const int goal_rad = goal_pos_.get_diam() / 2;
        const int robot_rad = robot_->get_radius();
        std::uniform_int_distribution<> robot_x(margin + robot_rad,
                                               width / 4 - (margin + robot_rad));
        std::uniform_int_distribution<> robot_y(margin + robot_rad,
                                               height / 3 - (margin + robot_rad));
        
        // Goal in right quarter, bottom third
        std::uniform_int_distribution<> goal_x((2 * width)  / 3 + margin,
                                       width - (margin + goal_rad));
        std::uniform_int_distribution<> goal_y((2 * height) / 3 + margin,
                                       height - (margin + goal_rad));                       

        //std::uniform_real_distribution<> dtheta(0.0, 2*M_PI);

        // this is just in case but this will always turn out false on first run 
        bool check = true;
        while (check) { //robot
            int x = robot_x(rng), y = robot_y(rng);
            if (map_->get_pixel(x, y) != fastsim::Map::obstacle) {
                robot_->set_pos(fastsim::Posture(x, y, 0)); 
                check = false;
            }
        }
        check = true;
        while (check) { //goal
            int x = goal_x(rng), y = goal_y(rng);
            if (map_->get_pixel(x, y) != fastsim::Map::obstacle){
                map_->add_goal(fastsim::Goal(x,y,20,0));
                check = false;
            }
        }
        }
    void save_summed_positions(const std::string& filename) const
    {
        const int width  = map_->get_pixel_w();
        const int height = map_->get_pixel_h();
        std::vector<unsigned char> buffer(width * height * 3, 255);

        // Load existing summed image if it exists
        std::ifstream ifs(filename, std::ios::binary);
        if (ifs.good()) {
            std::string header;
            int w, h, maxval;
            ifs >> header >> w >> h >> maxval;
            ifs.ignore(1); // skip newline
            if (w == width && h == height) {
                ifs.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            }
        }
        ifs.close();

        bool image_previously_exists = fs::exists(filename);
        if (!image_previously_exists) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int pidx = (y * width + x) * 3;
                    if (map_->get_pixel(x, y) == fastsim::Map::obstacle) {
                        buffer[pidx + 0] = 0;
                        buffer[pidx + 1] = 0;
                        buffer[pidx + 2] = 0;
                    } else {
                        buffer[pidx + 0] = 255;
                        buffer[pidx + 1] = 255;
                        buffer[pidx + 2] = 255;
                    }
                }
            }
            // Draw the goal as a solid green circle
            int gx = goal_pos_.get_x();
            int gy = goal_pos_.get_y();
            int goal_r = static_cast<int>(goal_pos_.get_diam()) / 2;
            for (int dy = -goal_r; dy <= goal_r; ++dy) {
                for (int dx = -goal_r; dx <= goal_r; ++dx) {
                    if (dx * dx + dy * dy > goal_r * goal_r) continue; // circle mask
                    int px = gx + dx;
                    int py = gy + dy;
                    if (px < 0 || px >= width || py < 0 || py >= height) continue;
                    int pidx = (py * width + px) * 3;
                    buffer[pidx + 0] = 0;   // Red   = 0
                    buffer[pidx + 1] = 255; // Green = 255
                    buffer[pidx + 2] = 0;   // Blue  = 0
                }
            }
        }

        // Stamp a larger red circle (~6 px radius) at the agent's final position
        int rx = static_cast<int>(robot_->get_pos().get_x());
        int ry = static_cast<int>(robot_->get_pos().get_y());
        const int mark_radius = 6;
        for (int dy = -mark_radius; dy <= mark_radius; ++dy) {
            for (int dx = -mark_radius; dx <= mark_radius; ++dx) {
                if (dx * dx + dy * dy > mark_radius * mark_radius) continue; // circle mask
                int px = rx + dx;
                int py = ry + dy;
                if (px < 0 || px >= width || py < 0 || py >= height) continue;
                int pidx = (py * width + px) * 3;
                // Blend: keep red channel maxed, dim G & B for transparency
                buffer[pidx + 0] = 255;
                buffer[pidx + 1] = static_cast<unsigned char>(buffer[pidx + 1] / 2);
                buffer[pidx + 2] = static_cast<unsigned char>(buffer[pidx + 2] / 2);
            }
        }

        std::ofstream ofs(filename, std::ios::binary);
        ofs << "P6\n" << width << " " << height << "\n255\n";
        ofs.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
    }
};

#endif
