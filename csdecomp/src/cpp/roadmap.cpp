#include "roadmap.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace csdecomp {

void DRM::Clear() {
  collision_map.clear();
  node_adjacency_map.clear();
  id_to_node_map.clear();
  id_to_pose_map.clear();
  options = RoadmapOptions();
}

void DRM::Write(const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  cereal::BinaryOutputArchive archive(file);

  archive(*this);
}

void DRM::Read(const std::string& filename) {
  Clear();
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  cereal::BinaryInputArchive archive(file);

  archive(*this);
}

std::vector<Eigen::Vector3f> DRM::GetWorkspaceCorners() {
  std::vector<Eigen::Vector3f> corners;
  corners.reserve(8);
  // Calculate half sizes
  float ox = options.robot_map_size_x / 2.0;
  float oy = options.robot_map_size_y / 2.0;
  float oz = options.robot_map_size_z / 2.0;
  // Define the offsets for each corner
  std::vector<Eigen::Vector3f> offsets = {
      {-ox, -oy, -oz}, {-ox, -oy, oz}, {-ox, oy, -oz}, {-ox, oy, oz},
      {ox, -oy, -oz},  {ox, -oy, oz},  {ox, oy, -oz},  {ox, oy, oz}};

  // Calculate each corner
  for (const auto& offset : offsets) {
    Eigen::Vector3f corner = options.map_center + offset;
    corners.push_back(corner);
  }

  return corners;
}
}  // namespace csdecomp