#include <fmt/core.h>

#include <algorithm>
#include <vector>

#include "cuda_collision_checker.h"
#include "cuda_edge_inflation_zero_order.h"
#include "cuda_edit_regions.h"
#include "cuda_forward_kinematics.h"

namespace csdecomp {

namespace {

/**
 * Argsort a slice of a float array
 *
 * @param arr The input float array
 * @param start The start index of the slice (inclusive)
 * @param end The end index of the slice (exclusive)
 * @return A vector of indices that would sort the slice
 */
std::vector<size_t> argsort_slice(const float* arr, size_t start, size_t end) {
  // Validate input
  if (start >= end) {
    return std::vector<size_t>();
  }

  // Create a vector of indices
  std::vector<size_t> indices(end - start);
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = start + i;
  }

  // Sort indices based on the values in the array
  std::sort(indices.begin(), indices.end(),
            [arr](size_t i1, size_t i2) { return arr[i1] < arr[i2]; });

  return indices;
}
}  // namespace

std::pair<std::vector<HPolyhedron>, std::pair<Eigen::MatrixXf, Eigen::MatrixXf>>
EditRegionsCuda(const Eigen::MatrixXf& collisions,
                const Eigen::MatrixXf& line_start_points,
                const Eigen::MatrixXf& line_end_points,
                const std::vector<HPolyhedron> regions,
                const MinimalPlant& plant,
                const std::vector<GeometryIndex>& robot_geometry_ids,
                const Voxels& voxels, const float voxel_radius,
                const EditRegionsOptions& options) {
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
  std::vector<u_int32_t> num_col_per_ls;
  line_segment_idxs.reserve(regions.size() * collisions.cols());
  Eigen::MatrixXf collisions_to_project(dimension,
                                        regions.size() * collisions.cols());
  int num_traj_collisions = 0;
  for (int region_index = 0; region_index < regions.size(); region_index++) {
    int num_col = 0;
    for (int col = 0; col < collisions.cols(); ++col) {
      if (regions.at(region_index).PointInSet(collisions.col(col))) {
        line_segment_idxs.emplace_back(region_index);
        collisions_to_project.col(num_traj_collisions) = collisions.col(col);
        ++num_traj_collisions;
        ++num_col;
      }
    }
    num_col_per_ls.push_back(num_col);
  }

  // prepare GPU memory
  Eigen::MatrixXf projections(dimension, num_traj_collisions);
  Eigen::MatrixXf optimized_collisions(dimension, num_traj_collisions);
  float distances_flat[num_traj_collisions];

  CudaPtr<float> projections_buffer(projections.data(),
                                    num_traj_collisions * dimension);
  CudaPtr<float> bisection_upper_bounds_buffer(nullptr,
                                               num_traj_collisions * dimension);
  CudaPtr<float> bisection_lower_bounds_buffer(nullptr,
                                               num_traj_collisions * dimension);
  CudaPtr<float> updated_configs_buffer(nullptr,
                                        num_traj_collisions * dimension);

  CudaPtr<float> optimized_collisions_buffer(optimized_collisions.data(),
                                             num_traj_collisions * dimension);

  CudaPtr<float> distances_buffer(distances_flat, num_traj_collisions);

  CudaPtr<const float> line_start_pts_ptr(line_start_points.data(),
                                          line_start_points.size());

  CudaPtr<const float> line_end_pts_ptr(line_end_points.data(),
                                        line_end_points.size());

  CudaPtr<const float> samples_ptr(collisions_to_project.data(),
                                   num_traj_collisions * dimension);
  CudaPtr<const u_int32_t> line_seg_idxs_ptr(line_segment_idxs.data(),
                                             num_traj_collisions);
  // Forward kinematics and CC
  CudaPtr<const MinimalPlant> plant_ptr(&plant, 1);
  CudaPtr<const GeometryIndex> robot_geometry_ids_ptr(
      robot_geometry_ids.data(), robot_geometry_ids.size());
  CudaPtr<float> transforms_buffer(
      nullptr, 16 * num_traj_collisions * plant.kin_tree.num_links);

  CudaPtr<uint8_t> is_pair_col_free_buffer(
      nullptr, plant.num_collision_pairs * num_traj_collisions);
  CudaPtr<uint8_t> is_config_col_free_buffer(nullptr, num_traj_collisions);
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
      distances_buffer.device, projections_buffer.device, samples_ptr.device,
      line_start_pts_ptr.device, line_end_pts_ptr.device,
      line_seg_idxs_ptr.device, num_traj_collisions, dimension);

  // send voxels to GPU
  cudaMemcpy((void*)voxel_buffer.device, voxels.data(),
             voxels.size() * sizeof(float), cudaMemcpyHostToDevice);

  // prepare bisection bounds
  cudaMemcpy((void*)bisection_lower_bounds_buffer.device,
             projections_buffer.device,
             num_traj_collisions * dimension * sizeof(float),
             cudaMemcpyDeviceToDevice);

  cudaMemcpy((void*)bisection_upper_bounds_buffer.device, samples_ptr.device,
             num_traj_collisions * dimension * sizeof(float),
             cudaMemcpyDeviceToDevice);

  for (int bisection_step = 0; bisection_step < options.bisection_steps;
       ++bisection_step) {
    executeStepConfigToMiddleKernel(
        updated_configs_buffer.device, bisection_upper_bounds_buffer.device,
        bisection_lower_bounds_buffer.device, num_traj_collisions, dimension);

    executeForwardKinematicsKernel(
        transforms_buffer.device, updated_configs_buffer.device,
        num_traj_collisions, &(plant_ptr.device->kin_tree));

    executeCollisionFreeVoxelsKernel(
        is_config_col_free_buffer.device, is_pair_col_free_buffer.device,
        is_geom_to_vox_pair_col_free_buffer.device, voxel_buffer.device,
        voxel_radius, plant_ptr.device, robot_geometry_ids_ptr.device,
        transforms_buffer.device, num_traj_collisions,
        plant.num_collision_pairs, robot_geometry_ids.size(), voxels.cols());

    executeUpdateBisectionBoundsKernel(
        bisection_lower_bounds_buffer.device,
        bisection_upper_bounds_buffer.device, updated_configs_buffer.device,
        is_config_col_free_buffer.device, num_traj_collisions, dimension);

    executeStoreIfCollisionKernel(
        optimized_collisions_buffer.device, updated_configs_buffer.device,
        is_config_col_free_buffer.device, num_traj_collisions, dimension);
  }
  executePointToPointDistanceKernel(
      distances_buffer.device, optimized_collisions_buffer.device,
      projections_buffer.device, num_traj_collisions, dimension);

  projections_buffer.copyDeviceToHost();
  optimized_collisions_buffer.copyDeviceToHost();
  distances_buffer.copyDeviceToHost();

  std::vector<HPolyhedron> edited_regions;
  int region_idx = 0;
  int curr_coll_idx = 0;
  for (const auto r : regions) {
    if (num_col_per_ls.at(region_idx)) {
      auto cand_idxs =
          argsort_slice(distances_flat, curr_coll_idx,
                        curr_coll_idx + num_col_per_ls.at(region_idx));
      // now loop through in ascending order and add faces
      Eigen::MatrixXf Anew(r.A().rows() + options.max_collisions_per_set,
                           dimension);
      int curr_num_faces = r.A().rows();
      Anew.topRows(curr_num_faces) = r.A();
      Eigen::VectorXf bnew;
      bnew.head(curr_num_faces) = r.b();
      for (const auto cand : cand_idxs) {
        Eigen::VectorXf opt = optimized_collisions.col(cand);
        // check if already redundant
        bool is_opt_contained = false;
        if ((Anew.topRows(curr_num_faces) * opt - bnew.head(curr_num_faces))
                .maxCoeff() <= 0) {
          is_opt_contained = true;
        }

        if (!is_opt_contained) {
          Eigen::VectorXf proj = projections.col(cand);
          Eigen::VectorXf a_face = opt - proj;
          a_face.normalize();
          float b_face =
              a_face.transpose() * opt - options.configuration_margin;
          float val_1 =
              a_face.transpose() * line_start_points.col(region_idx) - b_face;
          float val_2 =
              a_face.transpose() * line_end_points.col(region_idx) - b_face;
          float relaxation = std::max(val_1, val_2);
          if (relaxation > 0) {
            b_face += relaxation;
          }
          Anew.row(curr_num_faces) = a_face.transpose();
          bnew(curr_num_faces) = b_face;
          ++curr_num_faces;
        }
      }
      edited_regions.push_back(
          HPolyhedron(Anew.topRows(curr_num_faces), bnew.head(curr_num_faces)));
    } else {
      edited_regions.push_back(r);
    }
    ++region_idx;
    curr_coll_idx += num_col_per_ls.at(region_idx);
  }

  return {edited_regions, {projections, optimized_collisions}};
}

}  // namespace csdecomp