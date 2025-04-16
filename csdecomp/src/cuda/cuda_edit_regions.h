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

/**
 * @brief Refines a set of convex regions by removing collisions detected in
 * trajectories.
 *
 * This function implements the recovery mechanism described in section VI-B of
 * the paper "Superfast Configuration-Space Convex Set Computation on GPUs for
 * Online Motion Planning" found here: https://arxiv.org/pdf/2504.10783. It
 * takes colliding configurations found in trajectories and uses them to modify
 * the corresponding convex sets to exclude these collisions while still
 * ensuring that the line segments (typically from a piecewise linear path)
 * remain contained in the sets.
 *
 * The function performs the following steps:
 * 1. For each set, aggregates all colliding configurations inside that set
 * 2. Updates each set by adding hyperplanes that separate the collisions from
 * the line segments
 * 3. Verifies that all line segments remain contained in at least one convex
 * set
 *
 * @param collisions Matrix where each column represents a colliding
 * configuration
 * @param line_start_points Matrix where each column is the start point of a
 * line segment
 * @param line_end_points Matrix where each column is the end point of a line
 * segment
 * @param regions Vector of convex regions (polytopes) to be modified
 * @param plant Kinematics and collision model of the robot
 * @param robot_geometry_ids Indices of robot geometries to check for collisions
 * @param voxels Voxel representation of obstacles in the environment
 * @param voxel_radius Radius of spheres associated with voxels for collision
 * checking
 * @param options Configuration options for the region editing process
 *
 * @return A pair containing:
 *         - A vector of refined HPolyhedron regions with collisions removed
 *         - A pair of matrices containing any line segments that needed to be
 *           re-inflated (start points and end points)
 */

std::pair<std::vector<HPolyhedron>, std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>
EditRegionsCuda(const Eigen::MatrixXf& collisions,
                const Eigen::MatrixXf& line_start_points,
                const Eigen::MatrixXf& line_end_points,
                const std::vector<HPolyhedron> regions,
                const MinimalPlant& plant,
                const std::vector<GeometryIndex>& robot_geometry_ids,
                const Voxels& voxels, const float voxel_radius,
                const EditRegionsOptions& options);

}  // namespace csdecomp