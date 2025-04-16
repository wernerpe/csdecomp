#include <fmt/core.h>

#include "cuda_edit_regions.h"

namespace csdecomp {

std::vector<HPolyhedron> EditRegionsCuda(
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
  assert(ine_start_points.rows() == regions.at(0).ambient_dimension() &&
         "the linesegments must live in the same dimension as the regions");
  assert(!robot_geometry_ids.empty() && "Robot geometry IDs cannot be empty");
  assert(voxel_radius > 0 && "Voxel radius must be positive");

  int idx = 0;
  for (const auto r : regions) {
    assert(r.PointInSet(line_start_points.col(idx)) &&
           fmt::format("line start is not contained in region {}", idx));
    assert(r.PointInSet(line_end_points.col(idx)) &&
           fmt::format("line end is not contained in region {}", idx));
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
  float projections_flat[num_push_back * dimension];
  float distances_flat[num_push_back];
  CudaPtr<float> projections_ptr(projections_flat, num_push_back * dimension);
  CudaPtr<float> distances_ptr(distances_flat, num_push_back);

  CudaPtr<const float> line_start_pts_ptr(line_start_points.data(),
                                          line_start_points.size());

  CudaPtr<const float> line_end_pts_ptr(line_end_points.data(),
                                        line_end_points.size());

  CudaPtr<const float> samples_ptr(collisions_to_project.data(),
                                   num_push_back * dimension);
  CudaPtr<const u_int32_t> line_seg_idxs_ptr(line_seg_idxs.data(),
                                             num_push_back);

  line_start_pts_ptr.copyHostToDevice();
  line_end_pts_ptr.copyHostToDevice();
  samples_ptr.copyHostToDevice();
  line_seg_idxs_ptr.copyHostToDevice();

  executeProjectSamplesOntoLineSegmentsKernel(
      distances_ptr.device, projections_ptr.device, samples_ptr.device,
      line_start_pts_ptr.device, line_end_pts_ptr.device,
      line_seg_idxs_ptr.device, num_push_back, dimension);

  // dummy return value 
  std::vector<HPolyhedron> edited_regions;
  for(const auto r: regions){
    edited_regions.push_back(r);
  }
  return edited_regions;
}

}  // namespace csdecomp