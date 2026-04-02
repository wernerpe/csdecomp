#pragma once

#include <Eigen/Dense>
#include <vector>

#include "plant/collision_geometry.h"
#include "plant/minimal_plant.h"
#include "utils/cuda_utilities.h"

namespace csdecomp {

struct PlaneUpdateOptions {
  float configuration_margin{0.01};
  uint32_t bisection_steps{10};
};

struct PlaneUpdateResult {
  Eigen::MatrixXf a;                // ndof x M, plane normals (unit)
  Eigen::VectorXf b;                // M, plane offsets (a^T q + b <= 0)
  Eigen::MatrixXf boundary_points;  // ndof x M, bisected boundary points
  Eigen::VectorXf boundary_dists;   // M, ||boundary - feasible||
};

/**
 * @brief Computes separating hyperplanes between feasible and collision
 * configurations using GPU-accelerated bisection.
 *
 * For each pair (feasible_point, collision_point), bisects on the GPU to find
 * the collision boundary, then constructs a separating halfplane a^T q + b <= 0
 * with normal pointing from feasible toward collision.
 *
 * @param feasible_points ndof x M matrix of feasible configurations (columns)
 * @param collision_points ndof x M matrix of collision configurations (columns)
 * @param plant Kinematics and collision model of the robot
 * @param robot_geometry_ids Indices of robot geometries to check against voxels
 * @param voxels 3 x N_voxels matrix of voxel positions (pass empty for no
 * voxels)
 * @param voxel_radius Radius of voxel spheres (ignored if voxels is empty)
 * @param options Bisection steps and configuration margin
 * @return PlaneUpdateResult containing planes and boundary information
 */
PlaneUpdateResult ComputeSeparatingPlanesCuda(
    const Eigen::MatrixXf& feasible_points,
    const Eigen::MatrixXf& collision_points, const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids, const Voxels& voxels,
    float voxel_radius, const PlaneUpdateOptions& options);

}  // namespace csdecomp
