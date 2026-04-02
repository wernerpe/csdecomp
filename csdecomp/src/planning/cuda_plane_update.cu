#include <fmt/core.h>

#include <cmath>

#include "planning/cuda_edge_inflation_zero_order.h"
#include "planning/cuda_plane_update.h"
#include "planning/cuda_set_builder_utils.h"
#include "plant/cuda_collision_checker.h"
#include "plant/cuda_forward_kinematics.h"

namespace csdecomp {

PlaneUpdateResult ComputeSeparatingPlanesCuda(
    const Eigen::MatrixXf& feasible_points,
    const Eigen::MatrixXf& collision_points, const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids, const Voxels& voxels,
    float voxel_radius, const PlaneUpdateOptions& options) {
  const int dimension = feasible_points.rows();
  const int M = feasible_points.cols();
  assert(collision_points.rows() == dimension);
  assert(collision_points.cols() == M);
  assert(M > 0 && "Must have at least one point pair");

  const bool use_voxels = voxels.cols() > 0;
  if (use_voxels) {
    assert(!robot_geometry_ids.empty() && "Need robot geometry IDs for voxels");
    assert(voxel_radius > 0 && "Voxel radius must be positive");
    if (voxels.size() > 3 * MAX_NUM_VOXELS) {
      throw std::runtime_error("Voxel buffer size insufficient");
    }
  }

  // Allocate GPU memory
  CudaPtr<float> bisection_lo(nullptr, M * dimension);
  CudaPtr<float> bisection_hi(nullptr, M * dimension);
  CudaPtr<float> midpoints(nullptr, M * dimension);

  CudaPtr<const float> feas_ptr(feasible_points.data(), M * dimension);
  CudaPtr<const float> col_ptr(collision_points.data(), M * dimension);

  // FK and CC buffers
  CudaPtr<const MinimalPlant> plant_ptr(&plant, 1);
  CudaPtr<float> transforms(nullptr, 16 * M * plant.kin_tree.num_links);
  CudaPtr<uint8_t> is_pair_col_free(nullptr, plant.num_collision_pairs * M);
  CudaPtr<uint8_t> is_config_col_free(nullptr, M);

  // Voxel buffers (only used if use_voxels)
  CudaPtr<const GeometryIndex> robot_geom_ids_ptr(
      use_voxels ? robot_geometry_ids.data() : nullptr,
      use_voxels ? robot_geometry_ids.size() : 1);
  CudaPtr<uint8_t> is_geom_to_vox_col_free(
      nullptr, use_voxels ? robot_geometry_ids.size() * MAX_NUM_VOXELS * M : 1);
  CudaPtr<float> voxel_buffer(nullptr, 3 * MAX_NUM_VOXELS);

  // Upload data to GPU
  feas_ptr.copyHostToDevice();
  col_ptr.copyHostToDevice();
  plant_ptr.copyHostToDevice();

  // Init bisection: lo = feasible, hi = collision
  cudaMemcpy((void*)bisection_lo.device, feas_ptr.device,
             M * dimension * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy((void*)bisection_hi.device, col_ptr.device,
             M * dimension * sizeof(float), cudaMemcpyDeviceToDevice);

  if (use_voxels) {
    robot_geom_ids_ptr.copyHostToDevice();
    cudaMemcpy((void*)voxel_buffer.device, voxels.data(),
               voxels.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  // Bisection loop
  for (uint32_t step = 0; step < options.bisection_steps; ++step) {
    executeStepConfigToMiddleKernel(midpoints.device, bisection_hi.device,
                                    bisection_lo.device, M, dimension);

    executeForwardKinematicsKernel(transforms.device, midpoints.device, M,
                                   &(plant_ptr.device->kin_tree));

    if (use_voxels) {
      executeCollisionFreeVoxelsKernel(
          is_config_col_free.device, is_pair_col_free.device,
          is_geom_to_vox_col_free.device, voxel_buffer.device, voxel_radius,
          plant_ptr.device, robot_geom_ids_ptr.device, transforms.device, M,
          plant.num_collision_pairs, robot_geometry_ids.size(), voxels.cols());
    } else {
      executeCollisionFreeKernel(
          is_config_col_free.device, is_pair_col_free.device, plant_ptr.device,
          transforms.device, M, plant.num_collision_pairs);
    }

    executeUpdateBisectionBoundsKernel(bisection_lo.device, bisection_hi.device,
                                       midpoints.device,
                                       is_config_col_free.device, M, dimension);
  }

  // Copy boundary points (bisection_hi) back to host
  Eigen::MatrixXf boundary_points(dimension, M);
  cudaMemcpy(boundary_points.data(), bisection_hi.device,
             M * dimension * sizeof(float), cudaMemcpyDeviceToHost);

  // Construct planes on CPU
  PlaneUpdateResult result;
  result.boundary_points = boundary_points;
  result.a.resize(dimension, M);
  result.b.resize(M);
  result.boundary_dists.resize(M);

  for (int i = 0; i < M; ++i) {
    Eigen::VectorXf diff = boundary_points.col(i) - feasible_points.col(i);
    float dist = diff.norm();
    result.boundary_dists(i) = dist;

    if (dist > 1e-10f) {
      Eigen::VectorXf normal = diff / dist;
      float b_val = -normal.dot(boundary_points.col(i));
      float stepback = std::min(options.configuration_margin, 0.999f * dist);
      b_val += stepback;

      result.a.col(i) = normal;
      result.b(i) = b_val;
    } else {
      result.a.col(i).setZero();
      result.b(i) = 0.0f;
    }
  }

  return result;
}

}  // namespace csdecomp
