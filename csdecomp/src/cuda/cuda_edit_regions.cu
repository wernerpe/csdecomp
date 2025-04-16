#include <fmt/core.h>

#include "cuda_collision_checker.h"
#include "cuda_edge_inflation_zero_order.h"
#include "cuda_edit_regions.h"
#include "cuda_forward_kinematics.h"

namespace csdecomp {

std::pair<std::vector<HPolyhedron>, std::pair<Eigen::MatrixXf,Eigen::MatrixXf>> EditRegionsCuda(
    const Eigen::MatrixXf& collisions, const Eigen::MatrixXf& line_start_points,
    const Eigen::MatrixXf& line_end_points,
    const std::vector<HPolyhedron> regions, const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids, const Voxels& voxels,
    const float voxel_radius, const EditRegionsOptions& options) {
  // Check that problem data makes sense -> edges, regions, dimensions
  assert(line_start_points.cols() == regions.size() &&
         "Number of line segments must match number of regions");
  assert(line_end_points.cols() == regions.size() &&
         "Number of line segments must match number of regions");
  assert(line_start_points.rows() == line_end_points.rows() &&
         "Line start and end points must have same dimension");
  assert(!regions.empty() && "Regions vector cannot be empty");
  assert(collisions.rows() == regions.at(0).ambient_dimension() &&
         "collisions must live in the same space as the regions");
  assert(line_start_points.rows() == regions.at(0).ambient_dimension() &&
         "the linesegments must live in the same dimension as the regions");
  assert(!robot_geometry_ids.empty() && "Robot geometry IDs cannot be empty");
  assert(voxel_radius > 0 && "Voxel radius must be positive");

  int idx = 0;
  for (const auto& r : regions) {
    assert(r.PointInSet(line_start_points.col(idx)));
    if (!r.PointInSet(line_start_points.col(idx))) {
      throw std::runtime_error(
          fmt::format("line start is not contained in region {}", idx));
    }
    assert(r.PointInSet(line_end_points.col(idx)));
    if (!r.PointInSet(line_end_points.col(idx))) {
      throw std::runtime_error(
          fmt::format("line end is not contained in region {}", idx));
    }
    ++idx;
  }

  const int dimension = regions.at(0).ambient_dimension();

  // Determine what collisions are contained in which set
  std::vector<u_int32_t> line_segment_idxs;
  line_segment_idxs.reserve(regions.size() * collisions.cols());
  Eigen::MatrixXf collisions_to_project(dimension,
                                        regions.size() * collisions.cols());
  int num_push_back = 0;
  for (int col = 0; col < collisions.cols(); ++col) {
    for (int region_index = 0; region_index < regions.size(); region_index++) {
      if (regions.at(region_index).PointInSet(collisions.col(col))) {
        line_segment_idxs.emplace_back(region_index);
        collisions_to_project.col(num_push_back) = collisions.col(col);
        ++num_push_back;
      }
    }
  }

  // prepare GPU memory
  Eigen::MatrixXf projections(dimension, num_push_back);
  Eigen::MatrixXf optimized_collisions(dimension, num_push_back);
  float distances_flat[num_push_back];

  CudaPtr<float> projections_buffer(projections.data(), num_push_back * dimension);
  CudaPtr<float> bisection_upper_bounds_buffer(nullptr,
                                               num_push_back * dimension);
  CudaPtr<float> bisection_lower_bounds_buffer(nullptr,
                                               num_push_back * dimension);
  CudaPtr<float> updated_configs_buffer(nullptr, num_push_back * dimension);

  CudaPtr<float> optimized_collisions_buffer(optimized_collisions.data(),
                                             num_push_back * dimension);

  CudaPtr<float> distances_ptr(distances_flat, num_push_back);

  CudaPtr<const float> line_start_pts_ptr(line_start_points.data(),
                                          line_start_points.size());

  CudaPtr<const float> line_end_pts_ptr(line_end_points.data(),
                                        line_end_points.size());

  CudaPtr<const float> samples_ptr(collisions_to_project.data(),
                                   num_push_back * dimension);
  CudaPtr<const u_int32_t> line_seg_idxs_ptr(line_segment_idxs.data(),
                                             num_push_back);
  // Forward kinematics and CC
  CudaPtr<const MinimalPlant> plant_ptr(&plant, 1);
  CudaPtr<const GeometryIndex> robot_geometry_ids_ptr(
      robot_geometry_ids.data(), robot_geometry_ids.size());
  CudaPtr<float> transforms_buffer(
      nullptr, 16 * num_push_back * plant.kin_tree.num_links);

  CudaPtr<uint8_t> is_pair_col_free_buffer(
      nullptr, plant.num_collision_pairs * num_push_back);
  CudaPtr<uint8_t> is_config_col_free_buffer(nullptr, num_push_back);
  CudaPtr<uint8_t> is_geom_to_vox_pair_col_free_buffer(
      nullptr, robot_geometry_ids.size() * MAX_NUM_VOXELS);

  CudaPtr<float> voxel_buffer(nullptr, 3 * MAX_NUM_VOXELS);

  line_start_pts_ptr.copyHostToDevice();
  line_end_pts_ptr.copyHostToDevice();
  samples_ptr.copyHostToDevice();
  line_seg_idxs_ptr.copyHostToDevice();
  robot_geometry_ids_ptr.copyHostToDevice();
  plant_ptr.copyHostToDevice();

  executeProjectSamplesOntoLineSegmentsKernel(
      distances_ptr.device, projections_buffer.device, samples_ptr.device,
      line_start_pts_ptr.device, line_end_pts_ptr.device,
      line_seg_idxs_ptr.device, num_push_back, dimension);

  // todo remove this
  projections_buffer.copyDeviceToHost();

  // send voxels to GPU
  cudaMemcpy((void*)voxel_buffer.device, voxels.data(),
             voxels.size() * sizeof(float), cudaMemcpyHostToDevice);

  // prepare bisection bounds
  cudaMemcpy((void*)bisection_lower_bounds_buffer.device,
             projections_buffer.device, num_push_back * dimension * sizeof(float),
             cudaMemcpyDeviceToDevice);

  cudaMemcpy((void*)bisection_upper_bounds_buffer.device, samples_ptr.device,
             num_push_back * dimension * sizeof(float),
             cudaMemcpyDeviceToDevice);

  for (int bisection_step = 0; bisection_step < options.bisection_steps;
       ++bisection_step) {
    executeStepConfigToMiddleKernel(
        updated_configs_buffer.device, bisection_upper_bounds_buffer.device,
        bisection_lower_bounds_buffer.device, num_push_back, dimension);

    executeForwardKinematicsKernel(transforms_buffer.device,
                                   updated_configs_buffer.device, num_push_back,
                                   &(plant_ptr.device->kin_tree));

    executeCollisionFreeVoxelsKernel(
        is_config_col_free_buffer.device, is_pair_col_free_buffer.device,
        is_geom_to_vox_pair_col_free_buffer.device, voxel_buffer.device,
        voxel_radius, plant_ptr.device, robot_geometry_ids_ptr.device,
        transforms_buffer.device, num_push_back, plant.num_collision_pairs,
        robot_geometry_ids.size(), voxels.cols());

    executeUpdateBisectionBoundsKernel(
          bisection_lower_bounds_buffer.device,
          bisection_upper_bounds_buffer.device,
          updated_configs_buffer.device,
          is_config_col_free_buffer.device, num_push_back, dimension);

    executeStoreIfCollisionKernel(optimized_collisions_buffer.device,
                                  updated_configs_buffer.device,
                                  is_config_col_free_buffer.device,
                                  num_push_back, dimension);
  }
  optimized_collisions_buffer.copyDeviceToHost();
  
  // dummy return value
  std::vector<HPolyhedron> edited_regions;
  for (const auto r : regions) {
    edited_regions.push_back(r);
  }

  return {edited_regions, {projections, optimized_collisions}};
}

}  // namespace csdecomp