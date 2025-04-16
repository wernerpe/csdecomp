#pragma once

#include <vector>

#include "cuda_set_builder_utils.h"
#include "cuda_utilities.h"
#include "hpolyhedron.h"
#include "minimal_plant.h"

namespace csdecomp {

struct EditRegionsOptions {
  float configuration_margin{0.01};
  u_int32_t bisection_steps{9};
  u_int32_t max_collisions_per_set{100};
  bool verbose{false};
};

std::vector<HPolyhedron> EditRegionsCuda(
    const Eigen::MatrixXf& collisions, const Eigen::MatrixXf& line_start_points,
    const Eigen::MatrixXf& line_end_points,
    const std::vector<HPolyhedron> regions,
    const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids, const Voxels& voxels,
    const float voxel_radius, const EditRegionsOptions& options);

}  // namespace csdecomp