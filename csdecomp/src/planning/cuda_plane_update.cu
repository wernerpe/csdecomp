#include <fmt/core.h>

#include <cmath>

#include "planning/cuda_edge_inflation_zero_order.h"
#include "planning/cuda_plane_update.h"
#include "planning/cuda_set_builder_utils.h"
#include "plant/cuda_collision_checker.h"
#include "plant/cuda_forward_kinematics.h"

namespace csdecomp {

namespace {

void runBisectionLoop(CudaPtr<float>& bisection_lo,
                      CudaPtr<float>& bisection_hi, CudaPtr<float>& midpoints,
                      CudaPtr<float>& transforms,
                      CudaPtr<uint8_t>& is_pair_col_free,
                      CudaPtr<uint8_t>& is_config_col_free,
                      CudaPtr<uint8_t>& is_geom_to_vox_col_free,
                      CudaPtr<float>& voxel_buffer,
                      const CudaPtr<const MinimalPlant>& plant_ptr,
                      const CudaPtr<const GeometryIndex>& robot_geom_ids_ptr,
                      const MinimalPlant& plant, int num_robot_geom_ids,
                      int num_voxels, float voxel_radius, bool use_voxels,
                      int M, int dimension, uint32_t bisection_steps) {
  for (uint32_t step = 0; step < bisection_steps; ++step) {
    executeStepConfigToMiddleKernel(midpoints.device, bisection_hi.device,
                                    bisection_lo.device, M, dimension);

    executeForwardKinematicsKernel(transforms.device, midpoints.device, M,
                                   &(plant_ptr.device->kin_tree));

    if (use_voxels) {
      executeCollisionFreeVoxelsKernel(
          is_config_col_free.device, is_pair_col_free.device,
          is_geom_to_vox_col_free.device, voxel_buffer.device, voxel_radius,
          plant_ptr.device, robot_geom_ids_ptr.device, transforms.device, M,
          plant.num_collision_pairs, num_robot_geom_ids, num_voxels);
    } else {
      executeCollisionFreeKernel(
          is_config_col_free.device, is_pair_col_free.device, plant_ptr.device,
          transforms.device, M, plant.num_collision_pairs);
    }

    executeUpdateBisectionBoundsKernel(bisection_lo.device, bisection_hi.device,
                                       midpoints.device,
                                       is_config_col_free.device, M, dimension);
  }
}

PlaneUpdateResult constructPlanesFromBoundary(
    const Eigen::MatrixXf& feasible_points, CudaPtr<float>& bisection_hi,
    int dimension, int M, float configuration_margin) {
  Eigen::MatrixXf boundary_points(dimension, M);
  cudaMemcpy(boundary_points.data(), bisection_hi.device,
             M * dimension * sizeof(float), cudaMemcpyDeviceToHost);

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
      float stepback = std::min(configuration_margin, 0.999f * dist);
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

}  // namespace

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

  CudaPtr<const MinimalPlant> plant_ptr(&plant, 1);
  CudaPtr<float> transforms(nullptr, 16 * M * plant.kin_tree.num_links);
  CudaPtr<uint8_t> is_pair_col_free(nullptr, plant.num_collision_pairs * M);
  CudaPtr<uint8_t> is_config_col_free(nullptr, M);

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

  cudaMemcpy((void*)bisection_lo.device, feas_ptr.device,
             M * dimension * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy((void*)bisection_hi.device, col_ptr.device,
             M * dimension * sizeof(float), cudaMemcpyDeviceToDevice);

  if (use_voxels) {
    robot_geom_ids_ptr.copyHostToDevice();
    cudaMemcpy((void*)voxel_buffer.device, voxels.data(),
               voxels.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  runBisectionLoop(
      bisection_lo, bisection_hi, midpoints, transforms, is_pair_col_free,
      is_config_col_free, is_geom_to_vox_col_free, voxel_buffer, plant_ptr,
      robot_geom_ids_ptr, plant, robot_geometry_ids.size(), voxels.cols(),
      voxel_radius, use_voxels, M, dimension, options.bisection_steps);

  return constructPlanesFromBoundary(feasible_points, bisection_hi, dimension,
                                     M, options.configuration_margin);
}

// --- CudaPlaneUpdater ---

CudaPlaneUpdater::CudaPlaneUpdater(
    const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids,
    const PlaneUpdateOptions& options, int max_num_points, int max_num_voxels)
    : _dim(plant.kin_tree.num_configuration_variables),
      _max_M(max_num_points),
      _max_num_voxels(max_num_voxels),
      _options(options),
      _plant(plant),
      _robot_geometry_ids(robot_geometry_ids),
      plant_ptr(&_plant, 1),
      robot_geom_ids_ptr(
          robot_geometry_ids.empty() ? nullptr : _robot_geometry_ids.data(),
          robot_geometry_ids.empty() ? 1 : _robot_geometry_ids.size()),
      bisection_lo(nullptr, max_num_points * _dim),
      bisection_hi(nullptr, max_num_points * _dim),
      midpoints(nullptr, max_num_points * _dim),
      feas_buffer(nullptr, max_num_points * _dim),
      col_buffer(nullptr, max_num_points * _dim),
      transforms(nullptr, 16 * max_num_points * plant.kin_tree.num_links),
      is_pair_col_free(nullptr, plant.num_collision_pairs * max_num_points),
      is_config_col_free(nullptr, max_num_points),
      is_geom_to_vox_col_free(nullptr, max_num_voxels > 0
                                           ? robot_geometry_ids.size() *
                                                 max_num_voxels * max_num_points
                                           : 1),
      voxel_buffer(nullptr, max_num_voxels > 0 ? 3 * max_num_voxels : 1) {
  // Copy persistent data to device
  plant_ptr.copyHostToDevice();
  if (!robot_geometry_ids.empty()) {
    robot_geom_ids_ptr.copyHostToDevice();
  }
  cudaDeviceSynchronize();
}

PlaneUpdateResult CudaPlaneUpdater::computePlanes(
    const Eigen::MatrixXf& feasible_points,
    const Eigen::MatrixXf& collision_points, const Voxels& voxels,
    float voxel_radius) {
  const int M = feasible_points.cols();
  assert(feasible_points.rows() == _dim);
  assert(collision_points.rows() == _dim);
  assert(collision_points.cols() == M);

  if (M > _max_M) {
    throw std::runtime_error(fmt::format(
        "CudaPlaneUpdater: M={} exceeds max_num_points={}", M, _max_M));
  }

  const bool use_voxels = voxels.cols() > 0;
  if (use_voxels) {
    if (voxels.cols() > _max_num_voxels) {
      throw std::runtime_error(
          fmt::format("CudaPlaneUpdater: {} voxels exceeds max_num_voxels={}",
                      voxels.cols(), _max_num_voxels));
    }
    assert(!_robot_geometry_ids.empty());
    assert(voxel_radius > 0);
  }

  // Copy input points to pre-allocated device buffers
  cudaMemcpy((void*)feas_buffer.device, feasible_points.data(),
             M * _dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)col_buffer.device, collision_points.data(),
             M * _dim * sizeof(float), cudaMemcpyHostToDevice);

  // Init bisection bounds from the input buffers
  cudaMemcpy((void*)bisection_lo.device, feas_buffer.device,
             M * _dim * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy((void*)bisection_hi.device, col_buffer.device,
             M * _dim * sizeof(float), cudaMemcpyDeviceToDevice);

  if (use_voxels) {
    cudaMemcpy((void*)voxel_buffer.device, voxels.data(),
               voxels.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  runBisectionLoop(
      bisection_lo, bisection_hi, midpoints, transforms, is_pair_col_free,
      is_config_col_free, is_geom_to_vox_col_free, voxel_buffer, plant_ptr,
      robot_geom_ids_ptr, _plant, _robot_geometry_ids.size(), voxels.cols(),
      voxel_radius, use_voxels, M, _dim, _options.bisection_steps);

  return constructPlanesFromBoundary(feasible_points, bisection_hi, _dim, M,
                                     _options.configuration_margin);
}

}  // namespace csdecomp
