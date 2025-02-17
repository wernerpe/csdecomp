#pragma once
#include <Eigen/Dense>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace roadmaputils {

struct VectorHash {
  std::size_t operator()(const std::vector<int32_t>& vec) const {
    size_t hash = std::accumulate(
        vec.begin(), vec.end(), 0, [](size_t hash, const int32_t& elem) {
          return hash ^ (std::hash<int32_t>{}(elem) + 0x9e3779b9 + (hash << 6) +
                         (hash >> 2));
        });
    return hash;
  }
};

}  // namespace roadmaputils

namespace cereal {

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
void save(Archive& archive, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                                _MaxRows, _MaxCols>& m) {
  std::size_t rows = m.rows();
  std::size_t cols = m.cols();
  archive(rows, cols);
  archive(cereal::binary_data(m.data(), rows * cols * sizeof(_Scalar)));
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
void load(
    Archive& archive,
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  std::size_t rows, cols;
  archive(rows, cols);
  m.resize(rows, cols);
  archive(cereal::binary_data(m.data(), rows * cols * sizeof(_Scalar)));
}

}  // namespace cereal

namespace csdecomp {
struct RoadmapOptions {
  float robot_map_size_x{1};
  float robot_map_size_y{1};
  float robot_map_size_z{1};
  Eigen::Vector3f map_center{0.0, 0.0, 0.0};  // meters.
  int nodes_processed_before_debug_statement{1000};
  float max_task_space_distance_between_nodes{
      0.25};  // dist(W_X_F(q1).T@W_X_F(q2)) = dist(F1_X_F2)
  float max_configuration_distance_between_nodes{1};  // ||q1-q2||_2
  // This is the half of the sidelength of the voxel box in meters.
  float offline_voxel_resolution{0.05};
  float edge_step_size{0.01};  // resolution for cspace edge collision checks

  float GetOfflineVoxelRadius() { return sqrt(3) * offline_voxel_resolution; };
  template <class Archive>
  void serialize(Archive& archive) {
    archive(map_center, nodes_processed_before_debug_statement,
            max_task_space_distance_between_nodes,
            max_configuration_distance_between_nodes, offline_voxel_resolution,
            edge_step_size, robot_map_size_x, robot_map_size_y,
            robot_map_size_z);
  }
};

struct DRM {
  // (offline) voxel id -> node ids  (nodes that are in collision if this voxel)
  std::unordered_map<int32_t, std::vector<int32_t>> collision_map;
  // (offline) voxel id -> set: {{id1, id2}} of blocked edges. Edges are
  // inserted twice. Here i picked vectors because I couldnt figure out
  // serializing std pairs int he map of unordered set data structure.
  std::unordered_map<int32_t, std::unordered_set<std::vector<int32_t>,
                                                 roadmaputils::VectorHash>>
      edge_collision_map;
  // node id to neighbouring node ids
  std::unordered_map<int32_t, std::vector<int32_t>> node_adjacency_map;
  // node id to configuration
  std::unordered_map<int32_t, Eigen::VectorXf> id_to_node_map;
  // node id to target pose
  std::unordered_map<int32_t, Eigen::Matrix4f> id_to_pose_map;

  RoadmapOptions options;
  int target_link_index_{-1};
  std::string target_link_name_{"TARGET_LINK_IS_UNINITIALIZED"};

  void Read(const std::string& file_name);
  void Write(const std::string& file_name);

  void Clear();

  std::vector<Eigen::Vector3f> GetWorkspaceCorners();

  template <class Archive>
  void serialize(Archive& archive) {
    archive(CEREAL_NVP(collision_map), CEREAL_NVP(edge_collision_map),
            CEREAL_NVP(node_adjacency_map), CEREAL_NVP(id_to_node_map),
            CEREAL_NVP(id_to_pose_map), CEREAL_NVP(options),
            CEREAL_NVP(target_link_name_), CEREAL_NVP(target_link_index_));
  }
};
}  // namespace csdecomp