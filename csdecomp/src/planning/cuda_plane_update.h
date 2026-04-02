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
 * configurations using GPU-accelerated bisection (allocation per call).
 */
PlaneUpdateResult ComputeSeparatingPlanesCuda(
    const Eigen::MatrixXf& feasible_points,
    const Eigen::MatrixXf& collision_points, const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids, const Voxels& voxels,
    float voxel_radius, const PlaneUpdateOptions& options);

/**
 * @brief Pre-allocated GPU plane updater for fast repeated calls.
 *
 * Allocates all GPU memory in the constructor. The computePlanes() method
 * only performs cudaMemcpy + kernel launches, avoiding the ~7ms overhead
 * of cudaMalloc/cudaFree per call.
 */
class CudaPlaneUpdater {
 public:
  /**
   * @param plant Kinematics and collision model
   * @param robot_geometry_ids Robot geometry indices for voxel checks
   * @param options Bisection options (margin, steps)
   * @param max_num_points Maximum number of point pairs per call
   * @param max_num_voxels Maximum number of voxels (0 = no voxel support)
   */
  CudaPlaneUpdater(const MinimalPlant& plant,
                   const std::vector<GeometryIndex>& robot_geometry_ids,
                   const PlaneUpdateOptions& options, int max_num_points,
                   int max_num_voxels = 0);

  /**
   * @brief Compute separating planes using pre-allocated GPU memory.
   *
   * @param feasible_points ndof x M feasible configurations
   * @param collision_points ndof x M collision configurations
   * @param voxels 3 x N_voxels voxel positions (empty = no voxels)
   * @param voxel_radius Voxel sphere radius
   * @return PlaneUpdateResult with planes and boundary info
   */
  PlaneUpdateResult computePlanes(const Eigen::MatrixXf& feasible_points,
                                  const Eigen::MatrixXf& collision_points,
                                  const Voxels& voxels, float voxel_radius);

 private:
  const int _dim;
  const int _max_M;
  const int _max_num_voxels;
  const PlaneUpdateOptions _options;
  const MinimalPlant _plant;
  const std::vector<GeometryIndex> _robot_geometry_ids;

  // Persistent GPU memory
  CudaPtr<const MinimalPlant> plant_ptr;
  CudaPtr<const GeometryIndex> robot_geom_ids_ptr;

  // Work buffers (sized for max_M)
  CudaPtr<float> bisection_lo;
  CudaPtr<float> bisection_hi;
  CudaPtr<float> midpoints;
  CudaPtr<float> feas_buffer;
  CudaPtr<float> col_buffer;
  CudaPtr<float> transforms;
  CudaPtr<uint8_t> is_pair_col_free;
  CudaPtr<uint8_t> is_config_col_free;
  CudaPtr<uint8_t> is_geom_to_vox_col_free;
  CudaPtr<float> voxel_buffer;
};

}  // namespace csdecomp
