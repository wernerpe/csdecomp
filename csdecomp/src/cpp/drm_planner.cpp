// Copyright 2024 Toyota Research Institute.  All rights reserved.
#include "drm_planner.h"

#include <fmt/core.h>
// #ifdef _OPENMP
// #include <omp.h>
// #endif

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <utility>

#include "csdecomp/src/cuda/cuda_collision_checker.h"

namespace {
// Priority queue to store nodes by distance for BFS.
template <typename T, typename PriorityType>
struct PriorityQueue {
  // Compare pairs by first value, which is used to sort priority queue.
  class ElementCompare {
   public:
    bool operator()(std::pair<double, int32_t> a,
                    std::pair<double, int32_t> b) {
      return a.first > b.first;
    }
  };

  using Element = std::pair<PriorityType, T>;
  std::priority_queue<Element, std::vector<Element>, ElementCompare> elements;

  inline bool Empty() const { return elements.empty(); }

  inline void Put(T item, PriorityType priority) {
    elements.emplace(priority, item);
  }

  T Get() {
    T best_item = elements.top().second;
    elements.pop();
    return best_item;
  }
};

inline float measurePoseDistance(const Eigen::Matrix4f& pose1,
                                 const Eigen::Matrix4f& pose2) {
  // Compute translation distance
  float translation =
      (pose2.block<3, 1>(0, 3) - pose1.block<3, 1>(0, 3)).squaredNorm();

  // Compute rotation difference
  Eigen::Matrix3f R_diff =
      pose1.block<3, 3>(0, 0).transpose() * pose2.block<3, 3>(0, 0);

  // Use trace method to compute rotation angle
  float trace = R_diff.trace();
  float rotation = std::acos((trace - 1.0f) * 0.5f);

  // Combine translation and rotation
  return std::sqrt(translation) + 0.2f * rotation;
}

}  // namespace

namespace csdecomp {

DrmPlanner::DrmPlanner(const Plant& plant, const DrmPlannerOptions options,
                       bool quiet_mode)
    : options_(options) {
  tree_ = plant.getKinematicTree();
  mplant_ = plant.getMinimalPlant();
  inspector_ = plant.getSceneInspector();
  domain_.MakeBox(tree_.getPositionLowerLimits(),
                  tree_.getPositionUpperLimits());
  num_joints_ = tree_.numConfigurationVariables();

  is_initialized_ = true;
  quiet_mode_ = quiet_mode;
}

void DrmPlanner::LoadRoadmap(const std::string& road_map_filename) {
  drm_.Read(road_map_filename);
  drm_loaded_ = true;
  std::cout << fmt::format(
      "[DRMPlanner] DRM loaded which contains a pose map for '{}'.\n",
      drm_.target_link_name_);

  if (drm_.target_link_index_ !=
      tree_.getLinkIndexByName(drm_.target_link_name_)) {
    std::cout << fmt::format(
        "[DRMPlanner] Warning! The link index of the target frame used to "
        "construct posemap in the roadmap ({}) is different than the index in "
        "the "
        "supplied plant ({}). This can happen if you provide a different plant "
        "or have modified the URDF since building the roadmap. You might need "
        "to rebuild the roadmap.",
        drm_.target_link_index_,
        tree_.getLinkIndexByName(drm_.target_link_name_));
  };
}

double DrmPlanner::Distance(const Eigen::VectorXf& joints_1,
                            const Eigen::VectorXf& joints_2) {
  return (joints_1 - joints_2).norm();
}

void DrmPlanner::BuildCollisionSet(const Voxels& root_p_voxels) {
  assert(is_initialized_);
  assert(drm_loaded_);
  assert(drm_.options.offline_voxel_resolution >
         std::numeric_limits<double>::epsilon());

  collision_set_.clear();
  forbidden_edge_set_.clear();

  std::unordered_set<int32_t> checked_id;
  std::cout << fmt::format(
      "used options\n x: {} y: {} z: {} \n", drm_.options.robot_map_size_x,
      drm_.options.robot_map_size_y, drm_.options.robot_map_size_z);
  for (int ii = 0; ii < root_p_voxels.cols(); ++ii) {
    const int32_t id = GetCollisionVoxelId(root_p_voxels.col(ii), drm_.options);
    // if (true) {
    //   // std::cout << "\n Voxel: \n"
    //   //           << root_p_voxels.col(ii).transpose() << "  Voxel ID: " <<
    //   id
    //   //           << std::endl;
    // }
    if (checked_id.find(id) == checked_id.end()) {
      checked_id.insert(id);
    } else {
      continue;
    }
    // Skip if ID has been processed already.
    if (drm_.collision_map.find(id) != drm_.collision_map.end()) {
      for (const auto& element : drm_.collision_map.at(id)) {
        if (!quiet_mode_) {
          std::cout << "Inserting element ID: " << element
                    << "  , which represents position."
                    << drm_.id_to_node_map.at(element).transpose() << std::endl;
        }
        collision_set_.insert(element);
      }
    }

    // Build edge_collision_set
    if (drm_.edge_collision_map.find(id) != drm_.edge_collision_map.end()) {
      for (const auto& edge : drm_.edge_collision_map.at(id)) {
        // std::unordered_set<std::vector<int32_t>, roadmaputils::VectorHash>
        // tmp = loc_edge_collision_set;
        forbidden_edge_set_.insert(
            std::vector<int32_t>{edge.at(0), edge.at(1)});
      }
    }
  }

  std::cout << "Built collision set of size: " << collision_set_.size()
            << std::endl;
  std::cout << "Built edge collision set of size: "
            << forbidden_edge_set_.size() << std::endl;
}

int32_t DrmPlanner::GetClosestNonCollidingConfigurationInRoadMap(
    const Eigen::VectorXf& desired_joints, Eigen::VectorXf* closest_joints) {
  assert(closest_joints != nullptr);
  int32_t closest_node = -1;
  double closest_dist = std::numeric_limits<double>::infinity();

  for (const auto& element : drm_.id_to_node_map) {
    if (collision_set_.find(element.first) == collision_set_.end()) {
      const double dist = Distance(element.second, desired_joints);
      if (dist < closest_dist) {
        closest_dist = dist;
        closest_node = element.first;
      }
    }
  }
  assert(closest_node >= 0);
  *closest_joints = drm_.id_to_node_map.at(closest_node);
  return closest_node;
}

bool DrmPlanner::CheckPathExists(
    const int32_t start_node_id, const int32_t target_node_id,
    const std::vector<int32_t>& secondary_target_node_ids,
    std::vector<int32_t>* reachable_target_node_ids) {
  assert(is_initialized_);
  assert(drm_loaded_);

  // Clear reachability flag for secondary target joints.
  std::unordered_set<int32_t> target_node_ids_set;
  for (size_t ii = 0; ii < secondary_target_node_ids.size(); ++ii) {
    target_node_ids_set.insert(secondary_target_node_ids[ii]);
  }

  std::unordered_set<int32_t> explored_nodes;
  std::vector<int32_t> nodes_to_explore;
  nodes_to_explore.push_back(start_node_id);
  while (!nodes_to_explore.empty()) {
    // Expand next node, and remove from explored_nodes.
    const int32_t node = nodes_to_explore[nodes_to_explore.size() - 1];
    nodes_to_explore.pop_back();
    explored_nodes.insert(node);
    if (drm_.node_adjacency_map.find(node) == drm_.node_adjacency_map.end()) {
      continue;
    }
    for (const int32_t& next_node : drm_.node_adjacency_map.at(node)) {
      if (IsEdgeBlocked(next_node, node)) {
        continue;
      }
      if (next_node == target_node_id) {
        std::cout << "Path exists (not found yet)." << std::endl;
        return true;
      }
      if (reachable_target_node_ids != nullptr &&
          target_node_ids_set.count(next_node) > 0) {
        reachable_target_node_ids->push_back(next_node);
      }
      if (explored_nodes.find(next_node) == explored_nodes.end() &&
          collision_set_.find(next_node) == collision_set_.end()) {
        nodes_to_explore.push_back(next_node);
      }
    }
  }
  std::cout << "Path does not exist." << std::endl;
  return false;
}

// Find kinematic path from start joint to target joint in PRM.
bool DrmPlanner::FindPath(const int32_t start_node_id,
                          const Eigen::VectorXf& start_joints,
                          const int32_t target_node_id,
                          std::vector<Eigen::VectorXf>* joints_path,
                          std::vector<int32_t>* node_id_path) {
  assert(is_initialized_);
  assert(drm_loaded_);
  assert(joints_path != nullptr);
  assert(node_id_path != nullptr);
  joints_path->clear();
  node_id_path->clear();

  std::unordered_map<int32_t, double> cost_so_far;
  std::unordered_map<int32_t, int32_t> parents;
  PriorityQueue<int32_t, double> frontier;
  frontier.Put(start_node_id, 0.0);
  cost_so_far[start_node_id] = 0.0;
  parents[start_node_id] = start_node_id;
  const Eigen::VectorXf& target_node = drm_.id_to_node_map.at(target_node_id);
  size_t expanded_nodes = 0;
  while (!frontier.Empty()) {
    int32_t current = frontier.Get();
    if (drm_.id_to_node_map.find(current) == drm_.id_to_node_map.end()) {
      continue;
    }
    const Eigen::VectorXf& current_node = drm_.id_to_node_map.at(current);
    if (current == target_node_id) {
      PopulatePath(start_node_id, start_joints, current, parents, joints_path,
                   node_id_path);
      return true;
    }
    ++expanded_nodes;
    if (expanded_nodes > options_.max_nodes_to_expand) {
      std::cout << "Reached max nodes to expand." << std::endl;
      return false;
    }

    const std::vector<int32_t>& neighbors = drm_.node_adjacency_map.at(current);
    for (const auto& next : neighbors) {
      if (collision_set_.find(next) != collision_set_.end()) {
        continue;
      }
      if (drm_.id_to_node_map.find(next) == drm_.id_to_node_map.end()) {
        continue;
      }
      if (IsEdgeBlocked(current, next)) {
        continue;
      }
      const Eigen::VectorXf& next_node = drm_.id_to_node_map.at(next);
      const double new_cost =
          cost_so_far[current] + Distance(current_node, next_node);
      if (cost_so_far.find(next) == cost_so_far.end() ||
          new_cost < cost_so_far.at(next)) {
        cost_so_far[next] = new_cost;
        const double priority = new_cost + Distance(next_node, target_node);
        frontier.Put(next, priority);
        parents[next] = current;
      }
    }
  }
  std::cout << "Failed to find path in roadmap after verification step."
            << std::endl;
  std::abort();  // Raise SIGABRT
}

bool DrmPlanner::IsEdgeBlocked(int32_t node1, int32_t node2) const {
  return forbidden_edge_set_.count({node1, node2});
}

void DrmPlanner::PopulatePath(
    const int32_t start_node_id, const Eigen::VectorXf& start_joints,
    const int32_t target_node_id,
    const std::unordered_map<int32_t, int32_t>& parents,
    std::vector<Eigen::VectorXf>* joints_path, std::vector<int>* node_id_path) {
  assert(joints_path != nullptr);
  assert(node_id_path != nullptr);
  joints_path->clear();
  node_id_path->clear();
  int32_t node = target_node_id;
  while (node != start_node_id) {
    const Eigen::VectorXf& joint = drm_.id_to_node_map.at(node);
    joints_path->push_back(joint);
    node_id_path->push_back(node);
    node = parents.at(node);
  }
  assert(drm_.id_to_node_map.find(node) != drm_.id_to_node_map.end());
  const Eigen::VectorXf& joint = drm_.id_to_node_map.at(node);
  joints_path->push_back(joint);
  node_id_path->push_back(node);
  joints_path->push_back(start_joints);
  node_id_path->push_back(-1);  // Represent non-node joint with ID -1.
  std::reverse(joints_path->begin(), joints_path->end());
  std::reverse(node_id_path->begin(), node_id_path->end());
}

bool DrmPlanner::Plan(const Eigen::VectorXf& start_configuration,
                      const Eigen::VectorXf& target_configuration,
                      const Voxels& root_p_voxels, float online_voxel_radius,
                      std::vector<Eigen::VectorXf>* robot_joints_path) {
  auto start_plan = std::chrono::high_resolution_clock::now();
  assert(is_initialized_);
  assert(drm_loaded_);
  assert(robot_joints_path != nullptr);
  assert(options_.max_num_blocked_edges_before_discard_node > 1);
  assert(options_.voxel_padding >= 0);
  assert(options_.max_number_planning_attempts > 1);
  assert(options_.max_num_blocked_edges_before_discard_node > 1);

  attempts_to_leave_node.clear();

  int success_flag{0};
  int attempt = 0;
  for (; attempt < options_.max_number_planning_attempts; ++attempt) {
    success_flag = _plan(start_configuration, target_configuration,
                         root_p_voxels, online_voxel_radius, robot_joints_path);
    if (success_flag == 1) {
      break;
    }
    if (success_flag == -1) {
      throw std::runtime_error(
          "[DRMPlanner] No collision-free path exists in roadmap.");
    }
  }
  auto stop_plan = std::chrono::high_resolution_clock::now();
  if (success_flag == 1) {
    std::cout << fmt::format(
        "[DRMPlanner] Success after {} attemtps in {} ms\n", attempt + 1,
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_plan -
                                                              start_plan)
            .count());
  } else {
    std::cout << fmt::format(
        "[DRMPlanner] Failed after {} attemtps in {} ms\n", attempt + 1,
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_plan -
                                                              start_plan)
            .count());
  }
  return success_flag;
}

int DrmPlanner::_plan(const Eigen::VectorXf& start_configuration,
                      const Eigen::VectorXf& target_configuration,
                      const Voxels& root_p_voxels, float online_voxel_radius,
                      std::vector<Eigen::VectorXf>* robot_joints_path) {
  robot_joints_path->clear();

  // Check start configuration for collision.
  Eigen::MatrixXf start_config_as_matrix(start_configuration.size(), 2);
  start_config_as_matrix.col(0) = start_configuration;
  start_config_as_matrix.col(1) = target_configuration;
  std::vector<uint8_t> is_config_col_free = checkCollisionFreeVoxelsCuda(
      &start_config_as_matrix, &root_p_voxels, online_voxel_radius, &mplant_,
      inspector_.robot_geometry_ids);
  if (!is_config_col_free.at(0)) {
    std::cout << "[DrmPlanner:Plan] Robot is starting in collision."
              << std::endl;
    return -1;
  }
  if (!is_config_col_free.at(1)) {
    std::cout << "[DrmPlanner:Plan] Target configuration is in collision."
              << std::endl;
    return -1;
  }

  // Get start node in roadmap.
  int32_t start_id = -1;
  size_t steer_to_start_iterations = 0;
  bool steer_to_start_success = false;
  while (steer_to_start_iterations < options_.max_iterations_steering_to_node &&
         !steer_to_start_success) {
    Eigen::VectorXf closest_start_joint(start_configuration.size());
    start_id = GetClosestNonCollidingConfigurationInRoadMap(
        start_configuration, &closest_start_joint);
    if (start_id == -1) {
      std::cout << "[DrmPlanner:Plan] No collision-free configurations in "
                   "roadmap are close to starting configuration."
                << std::endl;
      return -1;
    }

    // Check that we can connect to roadmap without collision.
    if (SteerFunctionIsSafe(start_configuration, closest_start_joint,
                            root_p_voxels, online_voxel_radius)) {
      steer_to_start_success = true;
      break;
    }
    collision_set_.insert(start_id);
    ++steer_to_start_iterations;
  }

  if (!steer_to_start_success) {
    std::cout << "[DrmPlanner:Plan] Failed steer to start joint." << std::endl;
    return 0;
  }

  // Get target node in roadmap.
  int32_t target_id = -1;
  size_t steer_to_target_iterations = 0;
  bool steer_to_target_success = false;
  while (steer_to_target_iterations <
             options_.max_iterations_steering_to_node &&
         !steer_to_target_success) {
    Eigen::VectorXf closest_target_joint(target_configuration.size());
    target_id = GetClosestNonCollidingConfigurationInRoadMap(
        target_configuration, &closest_target_joint);
    if (target_id == -1) {
      std::cout << "[DrmPlanner:Plan] No collision-free configurations in "
                   "roadmap are close to target configuration."
                << std::endl;
      return 0;
    }

    // Check that we can connect to roadmap without collision.
    if (SteerFunctionIsSafe(target_configuration, closest_target_joint,
                            root_p_voxels, online_voxel_radius)) {
      steer_to_target_success = true;
      break;
    }
    collision_set_.insert(target_id);
    ++steer_to_target_iterations;
  }

  if (!steer_to_target_success) {
    std::cout << "[DrmPlanner:Plan] Failed steer to target configuration."
              << std::endl;
    return 0;
  }

  // Find path in roadmap from start node to target node.
  std::vector<Eigen::VectorXf> nominal_joints_path;
  std::vector<int> nominal_node_id_path;
  if (CheckPathExists(start_id, target_id, {}, nullptr)) {
    if (FindPath(start_id, start_configuration, target_id, &nominal_joints_path,
                 &nominal_node_id_path) &&
        nominal_joints_path.size() > 1) {
      std::cout << "[DrmPlanner:Plan] Found path of length: "
                << nominal_joints_path.size() << std::endl;
      // Add target configuration to end of path. We know that the robot can get
      // from the final node to target configuration collision-free based on our
      // steer_to_target_success check above.
      nominal_node_id_path.push_back(-1);
      nominal_joints_path.push_back(target_configuration);

      // Do lazy collision-checking and optional shortcutting through
      // robot_joints_path.
      if (PlanStatePath(nominal_joints_path, nominal_node_id_path,
                        options_.try_shortcutting, root_p_voxels,
                        online_voxel_radius, robot_joints_path)) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  } else {
    // no path exists return direct exit flag
    return -1;
  }
}

bool DrmPlanner::PlanStatePath(const std::vector<Eigen::VectorXf>& joint_path,
                               const std::vector<int>& node_id_path,
                               bool try_shortcut, const Voxels& root_p_voxels,
                               const float& online_voxel_radius,
                               std::vector<Eigen::VectorXf>* robot_joint_path) {
  assert(joint_path.size() > 0);
  assert(robot_joint_path != nullptr);
  robot_joint_path->clear();
  robot_joint_path->push_back(joint_path.front());

  Eigen::VectorXf previous_joint = joint_path.front();
  std::vector<int> reduced_node_id_path;
  std::vector<Eigen::VectorXf> reduced_joint_path;

  for (size_t i = 0; i < node_id_path.size(); ++i) {
    const int id = node_id_path[i];
    // explicitly invalidate the check if we are not storing edge nodes
    reduced_node_id_path.push_back(id);
    reduced_joint_path.push_back(joint_path.at(i));
  }

  for (size_t ii = 1; ii < reduced_joint_path.size(); ++ii) {
    double padded_voxel_resolution =
        options_.voxel_padding + online_voxel_radius;
    if (ii == 1 || ii == reduced_joint_path.size() - 1) {
      padded_voxel_resolution = online_voxel_radius;
    }

    if (!SteerFunctionIsSafe(previous_joint, reduced_joint_path[ii],
                             root_p_voxels, padded_voxel_resolution)) {
      if (!quiet_mode_) {
        std::cout << "Steer failed between nodes: " << ii - 1 << " and " << ii
                  << std::endl;
      }

      auto it = std::find(node_id_path.begin(), node_id_path.end(),
                          reduced_node_id_path[ii - 1]);
      int32_t node1 = *it;
      int32_t node2 = *(it + 1);
      if (node2 == -1) {
        // this indicates the target node.
        node2 = -2;
      }
      forbidden_edge_set_.insert({node1, node2});
      forbidden_edge_set_.insert({node2, node1});
      if (attempts_to_leave_node.find(node1) == attempts_to_leave_node.end()) {
        attempts_to_leave_node[node1] = 1;
      } else {
        attempts_to_leave_node.at(node1) += 1;
      }

      // insert node into collision set if too many outgoing edges are blocked.

      if (attempts_to_leave_node.at(node1) >=
          options_.max_num_blocked_edges_before_discard_node) {
        collision_set_.insert(node1);
      }
      return false;
    }
    robot_joint_path->push_back(reduced_joint_path[ii]);
    previous_joint = reduced_joint_path[ii];
  }
  // now do shortcutting if the path is found. At this point it must succeed
  // because returning the original path is a feasible solution.
  if (try_shortcut) {
    std::vector<Eigen::VectorXf> short_cut_path;
    short_cut_path.push_back(joint_path.front());
    size_t path_index = reduced_joint_path.size() - 1;
    size_t current_index = 0;
    previous_joint = reduced_joint_path.front();
    while (path_index > current_index) {
      // only do skip connections to major index targets (not edge nodes)
      // int32_t nodeid_target = reduced_node_id_path[path_index];

      Eigen::VectorXf target_joint = reduced_joint_path[path_index];
      double padded_voxel_resolution =
          options_.voxel_padding + online_voxel_radius;
      // Do not add voxel padding on joint move between first and last nodes in
      // path. This helps planning into and out of tight spaces.
      if ((path_index == reduced_joint_path.size() - 1 &&
           current_index == path_index - 1) ||
          (path_index == 1 && current_index == 0)) {
        padded_voxel_resolution = online_voxel_radius;
      }
      // we already checked the direct connections so we can skip those.
      if (path_index - current_index == 1 ||
          SteerFunctionIsSafe(previous_joint, target_joint, root_p_voxels,
                              padded_voxel_resolution)) {
        // Add target joint to path, assuming current joint is approximately
        // target joint.
        short_cut_path.push_back(target_joint);
        if (path_index == reduced_joint_path.size() - 1) {
          robot_joint_path->clear();
          robot_joint_path->swap(short_cut_path);
          return true;
        } else {
          current_index = path_index;
          path_index = reduced_joint_path.size() - 1;
          previous_joint = target_joint;
          continue;
        }
      }
      path_index -= 1;
    }
  } else {
    return true;
  }
  // this should not happen
  abort();
}

bool DrmPlanner::SteerFunctionIsSafe(
    const Eigen::VectorXf& current_configuration,
    const Eigen::VectorXf& next_configuration, const Voxels& root_p_voxels,
    const float& online_voxel_radius) {
  double distance_between_configurations =
      (next_configuration - current_configuration).norm();
  int num_configurations_to_check =
      distance_between_configurations / options_.online_edge_step_size + 1;
  Eigen::MatrixXf configurations(num_joints_, num_configurations_to_check);
  const Eigen::VectorXf edge_direction =
      (next_configuration - current_configuration) /
      distance_between_configurations;
  for (int ii = 0; ii < num_configurations_to_check; ++ii) {
    const Eigen::VectorXf intermediate_configuration =
        current_configuration +
        ii * options_.online_edge_step_size * edge_direction;
    configurations.col(ii) = intermediate_configuration;
  }
  const std::vector<uint8_t> is_config_col_free = checkCollisionFreeVoxelsCuda(
      &configurations, &root_p_voxels, online_voxel_radius, &mplant_,
      inspector_.robot_geometry_ids);
  for (size_t ii = 0; ii < is_config_col_free.size(); ++ii) {
    if (!is_config_col_free[ii]) {
      if (!quiet_mode_) {
        std::cout << "Collision between " << current_configuration.transpose()
                  << " and " << next_configuration.transpose() << std::endl;
      }
      return false;
    }
  }
  return true;
}

std::vector<Eigen::VectorXf>
DrmPlanner::GetClosestNonCollidingConfigurationsByPose(
    const Eigen::Matrix4f& X_W_targ, const int num_configs_to_try,
    const float search_cutoff) {
  assert(search_cutoff >= 0.05);
  std::vector<std::pair<int32_t, float>> pose_dist;
  pose_dist.reserve(drm_.id_to_pose_map.size() - collision_set_.size());

  for (const auto& [id, X] : drm_.id_to_pose_map) {
    if (collision_set_.find(id) != collision_set_.end()) {
      continue;
    }
    if ((X.block<3, 1>(0, 3) - X_W_targ.block<3, 1>(0, 3)).norm() >
        search_cutoff) {
      continue;
    }
    float dist = measurePoseDistance(X_W_targ, X);
    pose_dist.emplace_back(std::make_pair(id, dist));
  }

  std::cout << fmt::format("Found {} candidates.\n", pose_dist.size());

  // Get the closest num_configs_to_try configurations
  int num_to_try = std::min(num_configs_to_try, (int)pose_dist.size());
  // Sort pose_dist based on distance
  std::partial_sort(
      pose_dist.begin(), pose_dist.begin() + num_to_try, pose_dist.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });

  std::vector<Eigen::VectorXf> closest_configs;
  for (const auto& p : pose_dist) {
    int32_t id = p.first;
    closest_configs.push_back(drm_.id_to_node_map.at(id));
    if (closest_configs.size() == num_to_try) break;
  }

  return closest_configs;
}
}  // namespace csdecomp