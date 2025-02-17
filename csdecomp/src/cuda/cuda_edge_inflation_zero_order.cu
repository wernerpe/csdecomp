#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <chrono>

#include "cuda_collision_checker.h"
#include "cuda_edge_inflation_zero_order.h"
#include "cuda_forward_kinematics.h"
#include "cuda_hit_and_run_sampling.h"
#include "cuda_utilities.h"

// P_ptr.device,
// line_start_ptr.device,
// line_end_ptr.device,
// plant_ptr.device,
// //buffers that are not transferred between host and device
// configs_to_check_buffer.device,
// transforms_buffer.device,
// is_pair_col_free_buffer.device,
// is_config_col_free_buffer.device,
// configs_to_update_buffer.device,
// updated_configs_buffer.device,
// projections_buffer.device,
// distances_buffer.device
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
void executeEizo(MinimalHPolyhedron* P_device, const float* line_start_device,
                 const float* line_end_device, const float* voxels_device,
                 const float voxel_radius, const u_int32_t num_voxels,
                 const GeometryIndex* robot_geometry_ids,
                 const u_int32_t num_robot_geometries,
                 const MinimalPlant* plant_device,
                 const MinimalPlant* plant_host,
                 const EizoOptions* options_device,
                 const EizoOptions* options_host, EizoGpuMemory& mem,
                 const u_int32_t* line_segment_idxs_buffer,
                 EizoInfo& eizo_info) {
  u_int32_t dim = plant_host->kin_tree.num_configuration_variables;
  u_int32_t num_collision_pairs = plant_host->num_collision_pairs;
  // u_int32_t num_links = plant_host->kin_tree.num_links;

  for (int iteration = 0; iteration < options_host->max_iterations;
       ++iteration) {
    auto start_iteration_timer = std::chrono::high_resolution_clock::now();

    float delta_k = 6 * options_host->delta /
                    (M_PI * M_PI * (iteration + 1) * (iteration + 1));
    u_int32_t num_samples_iter =
        numSamplesUnadaptive(options_host->epsilon, delta_k, options_host->tau);

    u_int32_t num_samples_to_draw =
        max(num_samples_iter, options_host->num_particles);
    // sample configurations
    // spymemoryPolytope<<<1, 1>>>(P_device);
    // std::cout << fmt::format("generating {} samples in buffer of size {}\n",
    //                          num_samples_iter, max_num_configurations);

    executeUniformSampleInHPolyhedronKernel(
        mem.configs_to_check_buffer.device, P_device, line_start_device,
        num_samples_to_draw, dim, 1, options_host->mixing_steps);
    // spymemoryF<<<1, 1>>>(configs_to_check_buffer, 10 * dim);
    // check configurations for collisions
    // std::cout << fmt::format(
    //     "expected tf buffer requirement {} buffer of size {}\n",
    //     num_samples_iter * num_links, max_num_configurations * num_links);
    executeForwardKinematicsKernel(
        mem.transforms_buffer.device, mem.configs_to_check_buffer.device,
        num_samples_to_draw, &(plant_device->kin_tree));
    // printf(" transforms -----------\n");
    // spymemoryF<<<1, 1>>>(transforms_buffer, 10 * 16);

    // printf(" is_col_free -----------\n");
    // spymemoryI<<<1, 1>>>(is_config_col_free_buffer, 10);

    // executeCollisionFreeKernel(
    //     is_config_col_free_buffer, is_pair_col_free_buffer, plant_device,
    //     transforms_buffer, num_samples_iter, num_collision_pairs);
    executeCollisionFreeVoxelsKernel(
        mem.is_config_col_free_buffer.device,
        mem.is_pair_col_free_buffer.device,
        mem.is_geom_to_vox_pair_col_free_buffer.device, voxels_device,
        voxel_radius, plant_device, robot_geometry_ids,
        mem.transforms_buffer.device, num_samples_to_draw, num_collision_pairs,
        num_robot_geometries, num_voxels);
    // cudaDeviceSynchronize();
    // printf("is config col free [:10] \n");
    // spymemoryF<<<1, 1>>>(voxels_device, 10);

    // consolidate collisions in contiguous memory in preparation for the
    // bisection updates
    u_int32_t num_collisions_overall = executeCollisionConsolidationKernel(
        mem.configs_in_collision_buffer.device,
        mem.configs_to_check_buffer.device,
        mem.is_config_col_free_buffer.device, options_host->num_particles,
        num_samples_to_draw, dim, mem.collision_counter_ptr.device);

    u_int32_t num_collisions_unadaptive =
        executeCountCollisionsInFirstMEntriesKernel(
            mem.is_config_col_free_buffer.device, num_samples_iter,
            num_samples_to_draw, mem.first_m_collision_counter_ptr.device);

    if (options_host->verbose) {
      std::cout << fmt::format(
          "iter {}, num collisions found in first M samples {}, thresh {}\n",
          iteration, num_collisions_unadaptive,
          (1 - options_host->tau) * options_host->epsilon * num_samples_iter);
    }
    // break if threshold is passed and store info
    float collision_threshold_unadaptive =
        (1 - options_host->tau) * options_host->epsilon * num_samples_iter;
    if (num_collisions_unadaptive <= collision_threshold_unadaptive) {
      if (options_host->track_iteration_information) {
        auto end_iteration_timer = std::chrono::high_resolution_clock::now();
        float duration_iter =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_iteration_timer - start_iteration_timer)
                .count();
        EizoIterationInfo info;
        // improve the naming here
        info.iteration_index = iteration;
        info.collisions_threshold = collision_threshold_unadaptive;
        info.num_collisions_unadaptive = num_collisions_unadaptive;
        info.num_faces_placed = 0;
        info.num_samples_drawn = num_samples_to_draw;
        info.num_collisions_updated = 0;
        info.iteration_time_ms = duration_iter;
        eizo_info.iteration_info.push_back(info);
      }

      break;
    }

    // only update num_collision options_host->num_particles at most
    u_int32_t num_collisions_to_update =
        (num_collisions_overall > options_host->num_particles)
            ? options_host->num_particles
            : num_collisions_overall;

    executeProjectSamplesOntoLineSegmentsKernel(
        mem.distances_buffer.device, mem.projections_buffer.device,
        mem.configs_in_collision_buffer.device, line_start_device,
        line_end_device, line_segment_idxs_buffer, num_collisions_to_update,
        dim);
    // printf("distane buffer pre bisection\n");
    // spymemoryF<<<1, 1>>>(distances_buffer, 10 * dim);

    cudaMemcpy((void*)mem.bisection_lower_bounds_buffer.device,
               mem.projections_buffer.device,
               num_collisions_to_update * dim * sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy((void*)mem.bisection_upper_bounds_buffer.device,
               mem.configs_in_collision_buffer.device,
               num_collisions_to_update * dim * sizeof(float),
               cudaMemcpyDeviceToDevice);

    for (int bisection_step = 0; bisection_step < options_host->bisection_steps;
         ++bisection_step) {
      executeStepConfigToMiddleKernel(mem.updated_configs_buffer.device,
                                      mem.bisection_upper_bounds_buffer.device,
                                      mem.bisection_lower_bounds_buffer.device,
                                      num_collisions_to_update, dim);

      executeForwardKinematicsKernel(
          mem.transforms_buffer.device, mem.updated_configs_buffer.device,
          num_collisions_to_update, &(plant_device->kin_tree));

      // executeCollisionFreeKernel(
      //     is_config_col_free_buffer, is_pair_col_free_buffer, plant_device,
      //     transforms_buffer, num_collisions_to_update, num_collision_pairs);
      executeCollisionFreeVoxelsKernel(
          mem.is_config_col_free_buffer.device,
          mem.is_pair_col_free_buffer.device,
          mem.is_geom_to_vox_pair_col_free_buffer.device, voxels_device,
          voxel_radius, plant_device, robot_geometry_ids,
          mem.transforms_buffer.device, num_collisions_to_update,
          num_collision_pairs, num_robot_geometries, num_voxels);

      executeUpdateBisectionBoundsKernel(
          mem.bisection_lower_bounds_buffer.device,
          mem.bisection_upper_bounds_buffer.device,
          mem.updated_configs_buffer.device,
          mem.is_config_col_free_buffer.device, num_collisions_to_update, dim);

      // store last config in collision in configs_in_collision_buffer - these
      // are the optimized samples scol
      executeStoreIfCollisionKernel(mem.configs_in_collision_buffer.device,
                                    mem.updated_configs_buffer.device,
                                    mem.is_config_col_free_buffer.device,
                                    num_collisions_to_update, dim);
    }

    // update the distances
    executePointToPointDistanceKernel(
        mem.distances_buffer.device, mem.configs_in_collision_buffer.device,
        mem.projections_buffer.device, num_collisions_to_update, dim);
    // printf("distance buffer pre bisection\n");
    // spymemoryF<<<1, 1>>>(distances_buffer, 10 * dim);

    // clear cuda errors before thrust operation
    cudaError_t err = cudaGetLastError();
    thrust::device_ptr<float> thrust_dist_buf(mem.distances_buffer.device);
    int num_faces_added = 0;

    for (; num_faces_added < options_host->max_hyperplanes_per_iteration;
         ++num_faces_added) {
      //   std::cout << "pre success" << std::endl;
      auto min_iter =
          thrust::min_element(thrust::device, thrust_dist_buf,
                              thrust_dist_buf + num_collisions_to_update);

      int min_index = min_iter - thrust_dist_buf;

      //   printf("min_index %d\n", min_index);

      // this adds a face to the polytope and sets the distance of the
      // invalidated particles to MAX_F
      // spymemoryF<<<1, 1>>>(distances_buffer, 10 * dim);
      uint8_t done = executeAddFaceAndCheckOtherParticleRedundantKernel(
          P_device, mem.distances_buffer.device,
          mem.configs_in_collision_buffer.device, mem.projections_buffer.device,
          line_start_device, line_end_device, min_index,
          num_collisions_to_update, dim, options_host->configuration_margin,
          mem.success_flag_ptr.device);
      // spymemoryF<<<1, 1>>>(distances_buffer, 10 * dim);

      // we are done if adding a face fails.
      if (done == 1) {
        // std::cout<<"no faces added\n";
        break;
      }
      if (done == 2) {
        throw std::runtime_error(
            "Max face count exceeded. MinimalHPolyhedron is statically "
            "allocated. You need to increase MAX_NUM_FACES and recompile for "
            "this scenario.");
      }
    }

    if (options_host->track_iteration_information) {
      auto end_iteration_timer = std::chrono::high_resolution_clock::now();
      float duration_iter =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              end_iteration_timer - start_iteration_timer)
              .count();
      EizoIterationInfo info;
      // improve the naming here
      info.iteration_index = iteration;
      info.collisions_threshold = collision_threshold_unadaptive;
      info.num_collisions_unadaptive = num_collisions_unadaptive;
      info.num_faces_placed = num_faces_added;
      info.num_samples_drawn = num_samples_to_draw;
      info.num_collisions_updated = num_collisions_to_update;
      info.iteration_time_ms = duration_iter;
      if (dim <= 3) {
        // copy points post bisection into host debugging_float_buffer, then
        // copy into iteration info.
        cudaMemcpy((void*)mem.debugging_float_buffer.host,
                   mem.configs_in_collision_buffer.device,
                   num_collisions_to_update * dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        Eigen::Map<Eigen::MatrixXf> debugg_buffer_mapping(
            mem.debugging_float_buffer.host, dim, num_collisions_to_update);
        // make a copy of the particles
        Eigen::MatrixXf updated_collisions = debugg_buffer_mapping;
        info.points_post_bisection = updated_collisions;
      }
      eizo_info.iteration_info.push_back(info);
    }
  }
}

const MinimalHPolyhedron EizoCuda(
    const Eigen::VectorXf& line_start, const Eigen::VectorXf& line_end,
    const Voxels& voxels, const float voxel_radius, const HPolyhedron& domain,
    const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids,
    const EizoOptions& options, EizoInfo& eizo_info) {
  const u_int32_t dim = domain.ambient_dimension();
  const u_int32_t num_collision_pairs = plant.num_collision_pairs;

  assert(domain.ambient_dimension() == line_start.size());
  assert(domain.ambient_dimension() == line_end.size());
  assert(voxels.cols() <= MAX_NUM_VOXELS);
  assert(plant.kin_tree.num_configuration_variables ==
         domain.ambient_dimension());
  if (domain.ambient_dimension() >= MAX_DIM) {
    throw std::runtime_error(fmt::format(
        "The statically allocated memory requires to specify a max "
        "dimension, "
        "please recompile with a higher MAX_DIM, the current value is {}",
        MAX_DIM));
  }
  if (domain.A().rows() > 0.9 * MAX_NUM_FACES) {
    throw std::runtime_error(
        "ERROR: The domain polyhedron is already reserving almost all of the "
        "statically allocated faces. Consider recompiling with a higher "
        "MAX_NUM_FACES");
  };

  for (auto id : robot_geometry_ids) {
    assert(plant.num_scene_geometries > id);
  }

  float delta_min =
      6 * options.delta /
      (M_PI * M_PI * options.max_iterations * options.max_iterations);
  const u_int32_t max_num_configurations =
      numSamplesUnadaptive(options.epsilon, delta_min, options.tau);
  if (options.num_particles > max_num_configurations) {
    throw std::runtime_error(fmt::format(
        "Current required particle count exceeds buffer size. There were {} "
        "num_particles requested, and {} max_num_configurations space "
        "provided "
        "by the unadaptive test. Consider making the probabilistic test more "
        "stringent. ",
        options.num_particles, max_num_configurations));
  }
  // allocate memory

  // plant
  CudaPtr<const MinimalPlant> plant_ptr(&plant, 1);
  CudaPtr<const EizoOptions> options_ptr(&options, 1);

  // resulting region
  MinimalHPolyhedron P = domain.GetMyMinimalHPolyhedron();
  assert(domain.PointInSet(line_start));
  assert(domain.PointInSet(line_end));
  CudaPtr<MinimalHPolyhedron> P_ptr(&P, 1);

  plant_ptr.copyHostToDevice();
  options_ptr.copyHostToDevice();
  P_ptr.copyHostToDevice();

  CudaPtr<const float> line_start_ptr(line_start.data(), dim);
  CudaPtr<const float> line_end_ptr(line_end.data(), dim);

  CudaPtr<const GeometryIndex> robot_geometry_ids_ptr(
      robot_geometry_ids.data(), robot_geometry_ids.size());
  // CudaPtr<const float> voxels_ptr(voxels.data(), voxels.size());

  // one-time copy trivial line segment assignments
  std::vector<u_int32_t> line_segment_idxs(options.num_particles, 0);
  CudaPtr<u_int32_t> line_segment_idxs_buffer(line_segment_idxs.data(),
                                              options.num_particles);

  line_segment_idxs_buffer.copyHostToDevice();

  line_start_ptr.copyHostToDevice();
  line_end_ptr.copyHostToDevice();

  robot_geometry_ids_ptr.copyHostToDevice();

  EizoGpuMemory mem(plant, options, robot_geometry_ids, MAX_NUM_VOXELS);

  cudaMemcpy((void*)mem.voxel_buffer.device, voxels.data(),
             voxels.size() * sizeof(float), cudaMemcpyHostToDevice);

  executeEizo(P_ptr.device, line_start_ptr.device, line_end_ptr.device,
              mem.voxel_buffer.device, voxel_radius, voxels.cols(),
              robot_geometry_ids_ptr.device, robot_geometry_ids.size(),
              plant_ptr.device, plant_ptr.host, options_ptr.device,
              options_ptr.host,
              // memory struct with pointers to allocated buffers
              mem,
              // ---
              line_segment_idxs_buffer.device, eizo_info);

  P_ptr.copyDeviceToHost();
  cudaDeviceSynchronize();
  if (P.num_faces >= MAX_NUM_FACES) {
    std::cout << fmt::format(
        "Warning max num faces is {} and the current polytope has that face "
        "count. Consider recompiling with, and defining MAX_NUM_FACES as a "
        "larger number");
  }
  return P;
}

EizoGpuMemory::EizoGpuMemory(
    const MinimalPlant& plant, const EizoOptions& options,
    const std::vector<GeometryIndex>& robot_geometry_ids,
    const int max_num_voxels)
    : delta_min(
          6 * options.delta /
          (M_PI * M_PI * options.max_iterations * options.max_iterations)),
      max_num_configurations(
          max(options.num_particles,
              numSamplesUnadaptive(options.epsilon, delta_min, options.tau))),
      dim(plant.kin_tree.num_configuration_variables),
      voxel_buffer(nullptr, max_num_voxels * 3),
      // allocate collision checking buffers
      transforms_buffer(nullptr,
                        16 * max_num_configurations * plant.kin_tree.num_links),
      configs_to_check_buffer(nullptr, dim * max_num_configurations),
      is_pair_col_free_buffer(
          nullptr, plant.num_collision_pairs * max_num_configurations),
      is_geom_to_vox_pair_col_free_buffer(
          nullptr,
          max_num_configurations * robot_geometry_ids.size() * max_num_voxels),
      is_config_col_free_buffer(nullptr, max_num_configurations),
      // buffers for the bisection updates
      configs_in_collision_buffer(nullptr, options.num_particles * dim),
      updated_configs_buffer(nullptr, options.num_particles * dim),
      bisection_lower_bounds_buffer(nullptr, options.num_particles * dim),
      bisection_upper_bounds_buffer(nullptr, options.num_particles * dim),
      // buffers for the projections
      projections_buffer(nullptr, options.num_particles * dim),
      distances_buffer(nullptr, options.num_particles),
      // shared
      success_flag_ptr(&success_flag, 1),
      collision_counter_ptr(&collision_counter, 1),
      first_m_collision_counter_ptr(&collision_counter, 1),
      debugging_float_buffer_data(dim * options.num_particles),
      debugging_float_buffer(debugging_float_buffer_data.data(),
                             options.num_particles * dim) {
  if (options.num_particles > max_num_configurations) {
    throw std::runtime_error(fmt::format(
        "Current required particle count exceeds buffer size. There were {} "
        "num_particles requested, and {} max_num_configurations space "
        "provided "
        "by the unadaptive test. Consider making the probabilistic test more "
        "stringent. ",
        options.num_particles, max_num_configurations));
  }
}
}  // namespace csdecomp