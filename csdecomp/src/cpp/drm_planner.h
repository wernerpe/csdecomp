// Copyright 2024 Toyota Research Institute.  All rights reserved.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "roadmap.h"
#include "roadmap_builder.h"

namespace csdecomp {
struct DrmPlannerOptions {
  // Nodes to expand before declaring A* failed
  int max_nodes_to_expand{100000};
  // Max iterations of trying to connect the start and target configurations to
  // the roadmap.
  int max_iterations_steering_to_node{50};
  // Voxel padding for drm search to avoid getting too close to obstacles.
  float voxel_padding{0.02};
  // Attempt to shortcut path
  bool try_shortcutting{true};
  // Edge step size used for the collision checker used during the lazy
  // collision checking step during online path generation.
  float online_edge_step_size{0.01};
  // Max number of outgoing edges to explore before adding node to collision
  // set.
  int max_num_blocked_edges_before_discard_node{10};
  // Max number of planning attempts.
  int max_number_planning_attempts{100};
};

class DrmPlanner {
 public:
  DrmPlanner(const Plant& plant,
             const DrmPlannerOptions options = DrmPlannerOptions(),
             bool quiet_mode = true);

  void LoadRoadmap(const std::string& road_map_filename);

  double Distance(const Eigen::VectorXf& joints_1,
                  const Eigen::VectorXf& joints_2);

  //   Plans kinematic path for target arm from start_joint configuration to a
  //   target pose. The DRM must first be loaded.
  bool Plan(const Eigen::VectorXf& start_configuration,
            const Eigen::VectorXf& target_configuration,
            const Voxels& root_p_voxels, float online_voxel_radius,
            std::vector<Eigen::VectorXf>* robot_joint_path);

  // Runs a single planning attempt
  int _plan(const Eigen::VectorXf& start_configuration,
            const Eigen::VectorXf& target_configuration,
            const Voxels& root_p_voxels, float online_voxel_radius,
            std::vector<Eigen::VectorXf>* robot_joint_path);

  // Build collision set for the sensed environment, allowing quick PRM pruning.
  void BuildCollisionSet(const Voxels& root_p_voxels);

  //   // TODO(richard.cheng): Update robot model with attached bodies.
  //   void AttachCollisionBody();
  //   void DetachCollisionBody();
  //   void SetCollisionBoxPadding(double x, double y, double z);

  // Reset the tree.
  void Reset();

  //   // Get valid joint targets in PRM near the desired goal pose.
  //   bool GetValidJointTargets(const common::math::Pose3& robot_T_tip_target,
  //                             const common::math::VectorN& start_joint,
  //                             double min_translation_pose_distance,
  //                             double min_rotation_pose_distance,
  //                             std::vector<int32_t>* goal_joints);

  // Get closest node in PRM to the start_joint.
  int32_t GetClosestNonCollidingConfigurationInRoadMap(
      const Eigen::VectorXf& desired_joints, Eigen::VectorXf* closest_joints);

  std::vector<Eigen::VectorXf> GetClosestNonCollidingConfigurationsByPose(
      const Eigen::Matrix4f& X_W_targ, const int num_configs_to_try,
      const float search_cutoff = 0.3);

  bool CheckPathExists(const int32_t start_node_id,
                       const int32_t target_node_id,
                       const std::vector<int32_t>& secondary_target_node_ids,
                       std::vector<int32_t>* reachable_target_node_ids);

  bool FindPath(const int32_t start_node_id,
                const Eigen::VectorXf& start_joints,
                const int32_t target_node_id,
                std::vector<Eigen::VectorXf>* joint_path,
                std::vector<int32_t>* joint_id_path);

  void PopulatePath(const int32_t start_node_id,
                    const Eigen::VectorXf& start_joints,
                    const int32_t target_node_id,
                    const std::unordered_map<int32_t, int32_t>& parents,
                    std::vector<Eigen::VectorXf>* joints_path,
                    std::vector<int>* node_id_path);

  // Generate motion plan and trajectory. Keep other arm still if dual-arm.
  bool PlanStatePath(const std::vector<Eigen::VectorXf>& joints_path,
                     const std::vector<int>& node_id_path, bool try_shortcut,
                     const Voxels& root_p_voxels,
                     const float& online_voxel_radius,
                     std::vector<Eigen::VectorXf>* robot_joints_path);

  bool SteerFunctionIsSafe(const Eigen::VectorXf& current_configuration,
                           const Eigen::VectorXf& next_configuration,
                           const Voxels& root_p_voxels,
                           const float& online_voxel_radius);
  bool IsEdgeBlocked(int32_t node1, int32_t node2) const;

  HPolyhedron domain_;
  KinematicTree tree_;
  MinimalPlant mplant_;
  SceneInspector inspector_;
  int num_joints_ = -1;
  DrmPlannerOptions options_;
  DRM drm_;

  std::unordered_set<int32_t> collision_set_;
  std::unordered_set<std::vector<int32_t>, roadmaputils::VectorHash>
      forbidden_edge_set_;

  std::unordered_map<int32_t, int32_t> attempts_to_leave_node;
  bool is_initialized_ = false;
  bool drm_loaded_ = false;
  bool quiet_mode_ = false;
};
}  // namespace csdecomp