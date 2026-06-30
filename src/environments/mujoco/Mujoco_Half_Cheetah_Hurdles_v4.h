#ifndef Mujoco_Half_Cheetah_Hurdles_v4_h
#define Mujoco_Half_Cheetah_Hurdles_v4_h

#include <MujocoEnv.h>
#include <misc.h>

/**
 * Half-Cheetah with Hurdles Environment
 * 
 * Based on the environment from:
 * "NEUROEVOLUTION IS A COMPETITIVE ALTERNATIVE TO REINFORCEMENT LEARNING FOR SKILL DISCOVERY"
 * 
 * This environment adds 10 hurdles spaced 3 meters apart to the standard half-cheetah task.
 * The agent must learn to navigate over or around the hurdles while moving forward.
 */
class Mujoco_Half_Cheetah_Hurdles_v4 : public MujocoEnv {
  public:
   // Parameters
   double forward_reward_weight = 1.0;
   double control_cost_weight_ = 0.5;
   double reset_noise_scale_ = 0.1;
   bool exclude_current_positions_from_observation_ = true;
   // Optional termination condition to prevent "jumping exploit" behaviours.
   // `rootz` is qpos[1] for this model (see XML state-space ordering).
   bool terminate_on_high_rootz_ = false;
   double max_rootz_ = 0.0;
   static constexpr int kNumHurdles = 10;
   static constexpr double kBaseHurdleSpacing = 3.0;  // Base spacing in meters
   static constexpr double kFirstHurdleOffsetJitter = 1.0;  // Max offset (+/-) for first hurdle
   std::vector<int> hurdle_body_ids_;  // MuJoCo body IDs for hurdles

   Mujoco_Half_Cheetah_Hurdles_v4(std::unordered_map<std::string, std::any>& params) {
      eval_type_ = "Mujoco";
      n_eval_train_ = std::any_cast<int>(params["mj_n_eval_train"]);
      n_eval_validation_ = std::any_cast<int>(params["mj_n_eval_validation"]);
      n_eval_test_ = std::any_cast<int>(params["mj_n_eval_test"]);
      max_step_ = std::any_cast<int>(params["mj_max_timestep"]);
      // Match Gymnasium HalfCheetah-v4 default; helps keep dynamics comparable.
      frame_skip_ = 5;
      max_sim_steps_ = max_step_ * frame_skip_;
      control_cost_weight_ =
          std::any_cast<double>(params["mj_reward_control_weight"]);
      // Optional termination if the agent goes unrealistically high.
      // Defaults preserve legacy behaviour unless enabled in the YAML.
      if (params.find("mj_terminate_on_high_rootz") != params.end()) {
         terminate_on_high_rootz_ =
             std::any_cast<int>(params["mj_terminate_on_high_rootz"]) != 0;
      }
      if (params.find("mj_max_rootz") != params.end()) {
         max_rootz_ = std::any_cast<double>(params["mj_max_rootz"]);
      } else {
         // Reasonable default for HalfCheetah (torso starts at z~0.7).
         // Only used if mj_terminate_on_high_rootz is enabled.
         max_rootz_ = 1.2;
      }
      model_path_ =
          ExpandEnvVars(std::any_cast<string>(params["mj_model_path"]) + 
                        "half_cheetah_hurdles.xml");
      initialize_simulation();
      hurdle_body_ids_.resize(kNumHurdles);
      for (int i = 0; i < kNumHurdles; i++) {
         std::string name = "hurdle_" + std::to_string(i + 1);
         hurdle_body_ids_[i] = mj_name2id(m_, mjOBJ_BODY, name.c_str());
      }
      obs_size_ = 17;
      if (!exclude_current_positions_from_observation_)
         obs_size_ += 1;
      state_.resize(obs_size_);
   }

   ~Mujoco_Half_Cheetah_Hurdles_v4() {
      // Free visualization storage
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

   double control_cost(std::vector<double>& action) {
      double cost = 0;
      for (auto& a : action)
         cost += a * a;
      return control_cost_weight_ * cost;
   }

   bool terminal() {
      if (time_limit_reached()) return true;
      if (terminate_on_high_rootz_) {
         const double z = d_->qpos[1];  // rootz
         if (!std::isfinite(z)) return true;
         if (z > max_rootz_) return true;
      }
      return false;
   }

   Results sim_step(std::vector<double>& action) {
      auto x_pos_before = d_->qpos[0];
      do_simulation(action, frame_skip_);
      auto x_pos_after = d_->qpos[0];
      const double dt = m_->opt.timestep * std::max(1, frame_skip_);
      auto x_vel = (x_pos_after - x_pos_before) / dt;

      auto ctrl_cost = control_cost(action);

      auto forward_reward = forward_reward_weight * x_vel;

      auto rewards = forward_reward;
      auto costs = ctrl_cost;

      get_obs(state_);

      auto reward = rewards - costs;
      step_++;
      return {reward, 0.0};
   }

   void get_obs(std::vector<double>& obs) {
      if (exclude_current_positions_from_observation_) {
         std::copy_n(d_->qpos + 1, m_->nq - 1, obs.begin());
         std::copy_n(d_->qvel, m_->nv, obs.begin() + (m_->nq - 1));
      } else {
         std::copy_n(d_->qpos, m_->nq, obs.begin());
         std::copy_n(d_->qvel, m_->nv, obs.begin() + m_->nq);
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
      // Randomize only the starting offset; uniform spacing after
      std::uniform_real_distribution<> dis_offset(-kFirstHurdleOffsetJitter,
                                                   kFirstHurdleOffsetJitter);
      double start_offset = dis_offset(rng);
      for (int i = 0; i < kNumHurdles; i++) {
         int body_id = hurdle_body_ids_[i];
         if (body_id >= 0) {
            m_->body_pos[3 * body_id + 0] = (i + 1) * kBaseHurdleSpacing + start_offset;
         }
      }
      
      mj_resetData(m_, d_);
      set_state(qpos, qvel);
      step_ = 0;
      sim_steps_elapsed_ = 0;
      get_obs(state_);
   }
};

#endif

