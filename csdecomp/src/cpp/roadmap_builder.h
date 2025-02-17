// Copyright 2024 Toyota Research Institute.  All rights reserved.
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "collision_checker.h"
#include "hpolyhedron.h"
#include "roadmap.h"
#include "urdf_parser.h"

namespace csdecomp {

class RoadmapBuilder {
 public:
  RoadmapBuilder(const Plant& plant, const std::string& target_link_name,
                 const RoadmapOptions& options);

  // Build roadmap.
  void BuildRoadmap(int32_t max_neighbors);

  void GetRandomSamples(const int num_samples);

  int AddNodesManual(const Eigen::MatrixXf& nodes_to_add);

  // After building the roadmap, check every node for collision
  // with potential voxels in the environment. In the RoadmapOptions the field
  // offline_voxel_resolution determines the discretization at which we consider
  // potential collision voxels. Returns indices of deleted voxels. The num
  // batches decides how to chunk up the collision pairs to avoid going OOM ont
  // he GPU
  void BuildCollisionMap();

  void BuildEdgeCollisionMap(const float step_size_edge_collision_map = 0.01);

  // Build pose map mapping joint ids to robot_T_part_tip pose.
  void BuildPoseMap();

  void Write(const std::string& file_path);
  void Read(const std::string& file_path);

  void Reset();

  DRM GetDRM() const {
    DRM copy;
    copy.collision_map = drm_.collision_map;
    copy.node_adjacency_map = drm_.node_adjacency_map;
    copy.id_to_node_map = drm_.id_to_node_map;
    copy.id_to_pose_map = drm_.id_to_pose_map;
    copy.options = drm_.options;
    copy.edge_collision_map = drm_.edge_collision_map;
    copy.target_link_index_ = drm_.target_link_index_;
    copy.target_link_name_ = drm_.target_link_name_;
    return copy;
  }

 private:
  HPolyhedron domain_;
  KinematicTree tree_;
  MinimalPlant mplant_;
  SceneInspector inspector_;
  int configuration_dimension_ = -1;
  int target_link_index_ = -1;
  // RoadmapOptions options_;
  DRM drm_;

  std::unordered_map<int32_t, Eigen::VectorXf> raw_node_data_;
};

// Get the id of the offline voxel given the center of an online voxel.
int32_t GetCollisionVoxelId(const Eigen::Vector3f& robot_p_voxel,
                            const RoadmapOptions& options);

Eigen::Vector3f GetCollisionVoxelCenter(const int32_t index,
                                        const RoadmapOptions& options);

}  // namespace csdecomp