#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "drm_planner.h"
#include "voxel_wrapper.h"

namespace py = pybind11;
using namespace csdecomp;

void add_drm_bindings(py::module &m) {
  py::class_<RoadmapOptions>(m, "RoadmapOptions")
      .def(py::init<>())
      .def_readwrite("robot_map_size_x", &RoadmapOptions::robot_map_size_x)
      .def_readwrite("robot_map_size_y", &RoadmapOptions::robot_map_size_y)
      .def_readwrite("robot_map_size_z", &RoadmapOptions::robot_map_size_z)
      .def_readwrite("map_center", &RoadmapOptions::map_center)
      .def_readwrite("nodes_processed_before_debug_statement",
                     &RoadmapOptions::nodes_processed_before_debug_statement)
      .def_readwrite("max_task_space_distance_between_nodes",
                     &RoadmapOptions::max_task_space_distance_between_nodes)
      .def_readwrite("max_configuration_distance_between_nodes",
                     &RoadmapOptions::max_configuration_distance_between_nodes)
      .def_readwrite("offline_voxel_resolution",
                     &RoadmapOptions::offline_voxel_resolution)
      .def_readwrite("edge_step_size", &RoadmapOptions::edge_step_size)
      .def("get_offline_voxel_radius", &RoadmapOptions::GetOfflineVoxelRadius)
      .def("__repr__", [](const RoadmapOptions &self) {
        return "RoadmapOptions(\n"
               "robot_map_size_x=" +
               std::to_string(self.robot_map_size_x) +
               ", \n"
               "robot_map_size_y=" +
               std::to_string(self.robot_map_size_y) +
               ", \n"
               "robot_map_size_z=" +
               std::to_string(self.robot_map_size_z) +
               ", \n"
               "map_center=[" +
               std::to_string(self.map_center.x()) + ", " +
               std::to_string(self.map_center.y()) + ", " +
               std::to_string(self.map_center.z()) +
               "], \n"
               "nodes_processed_before_debug_statement=" +
               std::to_string(self.nodes_processed_before_debug_statement) +
               ", \n"
               "max_task_space_distance_between_nodes=" +
               std::to_string(self.max_task_space_distance_between_nodes) +
               ", \n"
               "max_configuration_distance_between_nodes=" +
               std::to_string(self.max_configuration_distance_between_nodes) +
               ", \n"
               "offline_voxel_resolution=" +
               std::to_string(self.offline_voxel_resolution) +
               ", \n"
               "edge_step_size=" +
               std::to_string(self.edge_step_size) + "\n)";
      });

  py::class_<DRM>(m, "DRM")
      .def(py::init<>())
      .def_readwrite("collision_map", &DRM::collision_map)
      .def_property_readonly(
          "edge_collision_map",
          [](const DRM &self) {
            std::unordered_map<int32_t,
                               std::vector<std::pair<int32_t, int32_t>>>
                result;
            for (const auto &[k, v] : self.edge_collision_map) {
              for (const auto &e : v) {
                result[k].push_back({e.at(0), e.at(1)});
              }
            }
            return result;
          })
      .def_readwrite("node_adjacency_map", &DRM::node_adjacency_map)
      .def_readwrite("id_to_node_map", &DRM::id_to_node_map)
      .def_readwrite("id_to_pose_map", &DRM::id_to_pose_map)
      .def_readwrite("options", &DRM::options)
      .def_readwrite("target_link_index_", &DRM::target_link_index_)
      .def_readwrite("target_link_name_", &DRM::target_link_name_)
      .def("Read", &DRM::Read)
      .def("Write", &DRM::Write)
      .def("GetWorkspaceCorners", &DRM::GetWorkspaceCorners)
      .def("Clear", &DRM::Clear);

  py::class_<RoadmapBuilder>(m, "RoadmapBuilder")
      .def(py::init<const Plant &, const std::string &,
                    const RoadmapOptions &>())
      .def("build_roadmap", &RoadmapBuilder::BuildRoadmap,
           py::arg("max_neighbors"))
      .def("get_random_samples", &RoadmapBuilder::GetRandomSamples)
      .def("add_nodes_manual", &RoadmapBuilder::AddNodesManual)
      .def("build_collision_map",
           [](RoadmapBuilder &self) { self.BuildCollisionMap(); })
      .def("build_edge_collision_map", &RoadmapBuilder::BuildEdgeCollisionMap)
      .def("build_pose_map", &RoadmapBuilder::BuildPoseMap)
      .def("write", &RoadmapBuilder::Write)
      .def("read", &RoadmapBuilder::Read)
      .def("reset", &RoadmapBuilder::Reset)
      .def("get_drm", &RoadmapBuilder::GetDRM);

  m.def("GetCollisionVoxelId", &GetCollisionVoxelId, py::arg("robot_p_voxel"),
        py::arg("options"));

  m.def("GetCollisionVoxelCenter", &GetCollisionVoxelCenter, py::arg("index"),
        py::arg("options"));

  py::class_<DrmPlannerOptions>(m, "DrmPlannerOptions")
      .def(py::init<>())
      .def_readwrite("max_nodes_to_expand",
                     &DrmPlannerOptions::max_nodes_to_expand)
      .def_readwrite("max_iterations_steering_to_node",
                     &DrmPlannerOptions::max_iterations_steering_to_node)
      .def_readwrite("voxel_padding", &DrmPlannerOptions::voxel_padding)
      .def_readwrite("try_shortcutting", &DrmPlannerOptions::try_shortcutting)
      .def_readwrite("online_edge_step_size",
                     &DrmPlannerOptions::online_edge_step_size)
      .def_readwrite(
          "max_num_blocked_edges_before_discard_node",
          &DrmPlannerOptions::max_num_blocked_edges_before_discard_node)
      .def_readwrite("max_number_planning_attempts",
                     &DrmPlannerOptions::max_number_planning_attempts);

  py::class_<DrmPlanner>(m, "DrmPlanner")
      .def(py::init<const Plant &, const DrmPlannerOptions &, bool>(),
           py::arg("plant"), py::arg("options") = DrmPlannerOptions(),
           py::arg("quiet_mode") = true)
      .def("LoadRoadmap", &DrmPlanner::LoadRoadmap)
      .def("Distance", &DrmPlanner::Distance)
      .def(
          "Plan",
          [](DrmPlanner &self, const Eigen::VectorXf &start_configuration,
             const Eigen::VectorXf &target_configuration,
             const VoxelsWrapper &root_p_voxels, double voxel_resolution)
              -> std::pair<bool, std::vector<Eigen::VectorXf>> {
            std::vector<Eigen::VectorXf> robot_joint_path;
            bool success = self.Plan(start_configuration, target_configuration,
                                     root_p_voxels.getMatrix(),
                                     voxel_resolution, &robot_joint_path);

            std::cout << fmt::format("Solution found with {} nodes.\n",
                                     robot_joint_path.size());
            for (const auto &p : robot_joint_path) {
              std::cout << "[";
              for (int i = 0; i < p.size(); i++) {
                std::cout << fmt::format(" {}, ", p[i]);
              }
              std::cout << "]\n";
            }
            return std::make_pair(success, robot_joint_path);
          },
          py::arg("start_configuration"), py::arg("target_configuration"),
          py::arg("online_voxel_positions_in_world_frame"),
          py::arg("voxel_radius"))
      // needs overhaul
      //   .def("_plan", &DrmPlanner::_plan)
      .def(
          "BuildCollisionSet",
          [](DrmPlanner &self, const VoxelsWrapper &root_p_voxels) {
            self.BuildCollisionSet(root_p_voxels.getMatrix());
          },
          py::arg("online_voxel_positions_in_world_frame"))
      .def(
          "GetClosestNonCollidingConfigurationsByPose",
          [](DrmPlanner &self, const Eigen::Matrix4f &X_W_targ,
             int num_configs_to_try, const float search_cutoff) {
            return self.GetClosestNonCollidingConfigurationsByPose(
                X_W_targ, num_configs_to_try, search_cutoff);
          },
          py::arg("X_W_targ"), py::arg("num_configs_to_try"),
          py::arg("search_cutoff") = 0.3,
          "Get the closest non-colliding configurations by pose")
      // .def("Reset", &DrmPlanner::Reset)
      //   .def("GetClosestNonCollidingConfigurationInRoadMap",
      //        [](const Eigen::VectorXf& desired_joints){
      //         Eigen::VectorXf result;
      //         int32_t nodeid =
      //
      // DrmPlanner::GetClosestNonCollidingConfigurationInRoadMap(desired_joints,
      //         &result); return {nodeid, result};
      //        })
      // .def("CheckPathExists", &DrmPlanner::CheckPathExists)
      // .def("FindPath", &DrmPlanner::FindPath)
      // .def("PopulatePath", &DrmPlanner::PopulatePath)
      // .def("PlanStatePath", &DrmPlanner::PlanStatePath)
      // .def("SteerFunctionIsSafe", &DrmPlanner::SteerFunctionIsSafe)
      // .def("IsEdgeBlocked", &DrmPlanner::IsEdgeBlocked)
      .def_readonly("domain", &DrmPlanner::domain_)
      .def_readonly("tree", &DrmPlanner::tree_)
      .def_readonly("mplant", &DrmPlanner::mplant_)
      .def_readonly("inspector", &DrmPlanner::inspector_)
      .def_readonly("num_joints", &DrmPlanner::num_joints_)
      .def_readonly("options", &DrmPlanner::options_)
      .def_readonly("drm", &DrmPlanner::drm_)
      .def_readonly("collision_set", &DrmPlanner::collision_set_)
      // this requires work, cannot convert to python type -> manually build
      // type
      .def_property_readonly(
          "forbidden_edge_set",
          [](const DrmPlanner &self) {
            std::vector<std::pair<int32_t, int32_t>> result;
            for (const auto &edge : self.forbidden_edge_set_) {
              if (edge.size() == 2) {
                result.emplace_back(edge[0], edge[1]);
              }
            }
            return result;
          })
      .def_readonly("is_initialized", &DrmPlanner::is_initialized_)
      .def_readonly("drm_loaded", &DrmPlanner::drm_loaded_)
      .def_readonly("quiet_mode", &DrmPlanner::quiet_mode_);
}