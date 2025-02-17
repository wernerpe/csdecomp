#pragma once
#include <Eigen/Dense>

#include "cuda_edge_inflation_zero_order.h"
#include "cuda_utilities.h"
#include "hpolyhedron.h"
#include "minimal_plant.h"

namespace csdecomp {
/**
 * @brief A class for performing edge inflation using CUDA. This class
 * pre-allocates memory on the GPU for faster sequential calls.
 *
 * This class encapsulates the functionality for calling EI-ZO.
 */
class CudaEdgeInflator {
 public:
  CudaEdgeInflator(const MinimalPlant& plant,
                   const std::vector<GeometryIndex>& robot_geometry_ids,
                   const EizoOptions& options, const HPolyhedron& domain);

  HPolyhedron inflateEdge(const Eigen::VectorXf& line_start,
                          const Eigen::VectorXf& line_end, const Voxels& voxels,
                          const float voxel_radius, bool verbose = true);

 private:
  EizoGpuMemory mem;
  const HPolyhedron _domain;
  const int _dim;
  const std::vector<GeometryIndex> _robot_geometry_ids;
  const MinimalPlant _plant;
  CudaPtr<const MinimalPlant> plant_ptr;
  EizoOptions _fei_options;
  CudaPtr<const EizoOptions> options_ptr;
  MinimalHPolyhedron _P;
  CudaPtr<MinimalHPolyhedron> P_ptr;
  CudaPtr<const GeometryIndex> robot_geometry_ids_ptr;
  Eigen::VectorXf _line_start;
  Eigen::VectorXf _line_end;
  CudaPtr<float> line_start_ptr;
  CudaPtr<float> line_end_ptr;
  std::vector<u_int32_t> _line_segment_idxs;
  CudaPtr<u_int32_t> line_segment_idxs_buffer;
};
}  // namespace csdecomp