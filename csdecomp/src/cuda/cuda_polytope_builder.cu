
#include "cuda_polytope_builder.h"
#include "cuda_utilities.h"

namespace csdecomp {
namespace {

__global__ void spymemoryF(const float* array, const u_int32_t N) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("spymemory \n");
    printArrayN(array, N);
    printf("\n------------\n end spymemory \n");
  }
}

__global__ void hackmemoryF(float* array) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    array[0] = 1;
    array[1] = 1;
  }
}
__global__ void spymemoryI(const uint8_t* array, const u_int32_t N) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("spymemory \n");
    printArrayN(array, N);
    printf("\n------------\n end spymemory \n");
  }
}

__global__ void spymemoryI(const int* array, const u_int32_t N) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("spymemory \n");
    printArrayN(array, N);
    printf("\n------------\n end spymemory \n");
  }
}
__global__ void spymemoryPlant(const MinimalPlant* pl) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("plant %d num links \n %d", pl->kin_tree.num_links,
           pl->num_scene_geometries);
    printf("\n------------\n end spyplant \n");
  }
}

__global__ void spymemoryPolytope(const MinimalHPolyhedron* p) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("\n------------\n polyhedron %d num faces \n %d dim", p->num_faces,
           p->dim);
    printf("A[:10, :].flattened() = \n");
    printMatrixNxMRowMajor(p->A, 10, p->dim);
    printf("b[:10].flattened() = \n");
    printArrayN(p->b, 10);
    printf("\n------------\n end spyplant \n");
  }
}
}  // namespace

CudaEdgeInflator::CudaEdgeInflator(
    const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids,
    const EizoOptions& options, const HPolyhedron& domain)
    : mem(plant, options, robot_geometry_ids, MAX_NUM_VOXELS),
      _domain(domain),
      _dim(domain.ambient_dimension()),
      _robot_geometry_ids(robot_geometry_ids),
      _plant(plant),
      plant_ptr(&_plant, 1),
      _fei_options(options),
      options_ptr(&_fei_options, 1),
      _P(domain.GetMyMinimalHPolyhedron()),
      P_ptr(&_P, 1),
      robot_geometry_ids_ptr(robot_geometry_ids.data(),
                             robot_geometry_ids.size()),
      _line_start(domain.ambient_dimension()),
      _line_end(domain.ambient_dimension()),
      line_start_ptr(_line_start.data(), domain.ambient_dimension()),
      line_end_ptr(_line_end.data(), domain.ambient_dimension()),
      _line_segment_idxs(options.num_particles, 0),
      line_segment_idxs_buffer(_line_segment_idxs.data(),
                               options.num_particles) {
  cudaDeviceSynchronize();
  assert(_dim == plant.kin_tree.num_configuration_variables);
  auto max_it =
      std::max_element(robot_geometry_ids.begin(), robot_geometry_ids.end());

  // assert we are only indexing valid scene geometries
  assert(plant.num_scene_geometries > *max_it);
  assert(_dim <= MAX_DIM);
  assert(options.num_particles <= MAX_NUM_PARTICLES);
  plant_ptr.copyHostToDevice();
  options_ptr.copyHostToDevice();
  P_ptr.copyHostToDevice();
  robot_geometry_ids_ptr.copyHostToDevice();
  line_segment_idxs_buffer.copyHostToDevice();

  cudaDeviceSynchronize();
}

HPolyhedron CudaEdgeInflator::inflateEdge(const Eigen::VectorXf& line_start,
                                          const Eigen::VectorXf& line_end,
                                          const Voxels& voxels,
                                          const float voxel_radius,
                                          bool verbose) {
  assert(_domain.PointInSet(line_start));
  assert(_domain.PointInSet(line_end));
  assert(voxels.cols() <= MAX_NUM_VOXELS);

  // reset P to the domain
  u_int32_t num_faces = _domain.A().rows();
  // reset face counter to domain. That part of the polytope is persistenl the
  // same.
  MinimalHPolyhedron P = _domain.GetMyMinimalHPolyhedron();
  //   cudaMemcpy((void*)(P_ptr.device), &P, sizeof(MinimalHPolyhedron),
  //              cudaMemcpyHostToDevice);
  cudaMemcpy((void*)&(P_ptr.device->num_faces), &num_faces, sizeof(u_int32_t),
             cudaMemcpyHostToDevice);

  // copy linesegment
  cudaMemcpy((void*)line_start_ptr.device, line_start.data(),
             _dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)line_end_ptr.device, line_end.data(), _dim * sizeof(float),
             cudaMemcpyHostToDevice);

  // copy voxels
  cudaMemcpy((void*)mem.voxel_buffer.device, voxels.data(),
             voxels.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  EizoInfo eizo_info;

  executeEizo(P_ptr.device, line_start_ptr.device, line_end_ptr.device,
              mem.voxel_buffer.device, voxel_radius, voxels.cols(),
              robot_geometry_ids_ptr.device, _robot_geometry_ids.size(),
              plant_ptr.device, plant_ptr.host, options_ptr.device,
              options_ptr.host,
              // memory struct with pointers to allocated buffers
              mem,
              // ---
              line_segment_idxs_buffer.device, eizo_info);
  cudaDeviceSynchronize();
  P_ptr.copyDeviceToHost();
  assert(&_P == P_ptr.host);
  HPolyhedron result(_P);
  if (verbose) {
    printf(
        "======================== Set builder ===========================\n");
    printEizoSummary(eizo_info);
    printf(
        "===================== End Set builder ===========================\n");
  }
  return result;
}
}  // namespace csdecomp