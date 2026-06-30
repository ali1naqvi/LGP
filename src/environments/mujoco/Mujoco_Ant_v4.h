#ifndef Mujoco_Ant_v4_h
#define Mujoco_Ant_v4_h

#include <MujocoEnv.h>
#include <misc.h>
#include <iostream>

class Mujoco_Ant_v4 : public MujocoEnv {
   public:
    bool damage_active_this_episode_ = false;
    int damage_start_step_ = -1;
    int chosen_leg_to_damage_ = -1;
    double current_damage_level_ = 0.0;
    double damage_prob_ = 0.0;   

    double damage_progression_rate_ = 0.01;
    double max_damage_level_ = 1.0; 
    bool linear_damage_ = true; // if false, exponentially

    double control_cost_weight_ = 0.5;
    bool use_contact_forces_ = false;
    double contact_cost_weight_ = 5e-4;
    double healthy_reward_ = 1.0;
    bool terminate_when_unhealthy_ = true;
    std::vector<double> healthy_z_range_;
    std::vector<double> contact_force_range_;
    double reset_noise_scale_ = 0.1;
    bool exclude_current_positions_from_observation_ = false;
    bool observable_damage_ = false;

    Mujoco_Ant_v4(std::unordered_map<std::string, std::any>& params) {
        damage_prob_ = std::any_cast<double>(params["mj_break_leg_ant"]);
        damage_progression_rate_ = std::any_cast<double>(params["mj_damage_progression_rate"]);
        linear_damage_ = bool((std::any_cast<int>(params["mj_linear_damage"]) != 0));
        observable_damage_ = bool((std::any_cast<int>(params["mj_observable_damage"]) != 0));
        
        eval_type_ = "Mujoco";
        n_eval_train_ = std::any_cast<int>(params["mj_n_eval_train"]);
        n_eval_validation_ = std::any_cast<int>(params["mj_n_eval_validation"]);
        n_eval_test_ = std::any_cast<int>(params["mj_n_eval_test"]);
        max_step_ = std::any_cast<int>(params["mj_max_timestep"]);
        max_sim_steps_ = max_step_ * frame_skip_;
        model_path_ =
          ExpandEnvVars(std::any_cast<string>(params["mj_model_path"]) + 
                        "ant.xml");
        healthy_z_range_ = {0.2, 1.0};
        contact_force_range_ = {-1.0, 1.0};
        initialize_simulation();

        obs_size_ = 27;
        if (!exclude_current_positions_from_observation_) obs_size_ += 2;
        if (use_contact_forces_) obs_size_ += 84;
        if (observable_damage_) obs_size_ += 2;

        state_.resize(obs_size_);
    }

    ~Mujoco_Ant_v4() {
        mjv_freeScene(&scn_);
        mjr_freeContext(&con_);

        // Free MuJoCo model and data
        mj_deleteData(d_);
        mj_deleteModel(m_);

        // Terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
#endif
    }

    double healthy_reward() {
        return static_cast<double>(is_healthy() || terminate_when_unhealthy_) *
               healthy_reward_;
    }

    double control_cost(std::vector<double>& action) {
        double cost = 0;
        for (auto& a : action) cost += a * a;
        return control_cost_weight_ * cost;
    }

    std::vector<double> contact_forces() {
        std::vector<double> forces;
        std::copy_n(d_->cfrc_ext, m_->nbody * 6, back_inserter(forces));
        for (auto& f : forces) {
            f = std::max(contact_force_range_[0],
                         std::min(f, contact_force_range_[1]));
        }
        return forces;
    }

    double contact_cost() {
        auto forces = contact_forces();
        double cost = 0;
        for (auto& f : forces) cost += f * f;
        return contact_cost_weight_ * cost;
    }

    bool is_healthy() {
        for (int i = 0; i < m_->nq; i++)
            if (!std::isfinite(d_->qpos[i])) return false;
        for (int i = 0; i < m_->nv; i++)
            if (!std::isfinite(d_->qvel[i])) return false;
        return (d_->qpos[2] >= healthy_z_range_[0] &&
                d_->qpos[2] <= healthy_z_range_[1]);
    }

    bool terminal() {
        bool done = time_limit_reached() ||
                    (terminate_when_unhealthy_ && !is_healthy());

        return done;
    }

    void update_damage_level() {
        if (damage_active_this_episode_ && step_ >= damage_start_step_) {
            if (linear_damage_) {
                current_damage_level_ += damage_progression_rate_;
            } else {
                current_damage_level_ = 1.0 - ((1.0 - current_damage_level_) * (1.0 - damage_progression_rate_));
            }
            
            current_damage_level_ = std::min(current_damage_level_, max_damage_level_);
        }
    }

    Results sim_step(std::vector<double>& action) {
        update_damage_level();
        
        // if (damage_active_this_episode_ && step_ == damage_start_step_) {
        //     std::cout << "Leg break at step " << step_ 
        //               << " (leg " << chosen_leg_to_damage_ << ")" << std::endl;
        // }
        
        if (damage_active_this_episode_ && step_ >= damage_start_step_) {
            int actuator_start = chosen_leg_to_damage_ * 2;
            action[actuator_start] *= (1.0 - current_damage_level_);
            action[actuator_start + 1] *= (1.0 - current_damage_level_);
        }

        auto x_pos_before = d_->qpos[0];
        do_simulation(action, frame_skip_);
        auto x_pos_after = d_->qpos[0];
        const double dt = m_->opt.timestep * std::max(1, frame_skip_);
        auto x_vel = (x_pos_after - x_pos_before) / dt;
        auto forward_reward = x_vel;
        auto rewards = forward_reward + healthy_reward();
        auto ctrl_cost = control_cost(action);
        auto costs = ctrl_cost;
        if (use_contact_forces_) {
            costs += contact_cost();
        }
        auto reward = rewards - costs;
        // std::cout << "Step " << step_ << " Reward: " << reward << std::endl;
        // std::cout << " (" << d_->qpos[0] << ", " << d_->qpos[1] << ") " << std::endl;
        get_obs(state_);
        step_++;
        return {reward, 0.0};  // TODO(skelly): maybe add gym 'info' to results
    }

    void get_obs(std::vector<double>& obs) {
        if (exclude_current_positions_from_observation_) {
            std::copy_n(d_->qpos + 2, m_->nq - 2, obs.begin());
            std::copy_n(d_->qvel, m_->nv, obs.begin() + (m_->nq - 2));
        } else {
            std::copy_n(d_->qpos, m_->nq, obs.begin());
            std::copy_n(d_->qvel, m_->nv, obs.begin() + m_->nq);
        }
        
        if (observable_damage_) {
            size_t damage_index = exclude_current_positions_from_observation_ ? 
                                (m_->nq - 2) + m_->nv : 
                                m_->nq + m_->nv;

            obs[damage_index] = damage_active_this_episode_ && step_ >= damage_start_step_ ? 
                            current_damage_level_ : 0.0;
            obs[damage_index + 1] = damage_active_this_episode_ && step_ >= damage_start_step_ ? 
                                static_cast<double>(chosen_leg_to_damage_) : -1.0;
        }
    }
    void reset(mt19937& rng, int& episode_number) {
        std::uniform_real_distribution<> dis_pos(-reset_noise_scale_,
                                                 reset_noise_scale_);
        std::vector<double> qpos(m_->nq);
        for (size_t i = 0; i < qpos.size(); i++) {
            qpos[i] = init_qpos_[i] + dis_pos(rng);
        }
        std::normal_distribution<double> dis_vel(0.0, reset_noise_scale_);
        std::vector<double> qvel(m_->nv);
        for (size_t i = 0; i < qvel.size(); i++) {
            qvel[i] = init_qvel_[i] + dis_vel(rng);
        }
        mj_resetData(m_, d_);
        set_state(qpos, qvel);
        step_ = 0;
        sim_steps_elapsed_ = 0;
        
        current_damage_level_ = 0.0;

        std::uniform_real_distribution<double> damage_occur_dist(0.0, 1.0);
        double val = damage_occur_dist(rng);
        if (val < damage_prob_) {
            damage_active_this_episode_ = true;
            if (damage_progression_rate_ >= 1.0) {
                damage_start_step_ = 0;
            } else {
                std::uniform_int_distribution<> step_dist(1, max_episode_control_steps() / 3); 
                damage_start_step_ = step_dist(rng);
                //std::cerr << "damage at: " << damage_start_step_ << endl;
            }

            std::uniform_int_distribution<> leg_dist(0, 3); 
            chosen_leg_to_damage_ = leg_dist(rng);
        } else {
            damage_active_this_episode_ = false;
            damage_start_step_ = -1;
            chosen_leg_to_damage_ = -1;
        }
    }
};

#endif