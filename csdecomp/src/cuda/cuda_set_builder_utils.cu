#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>

#include <algorithm>
#include <iostream>

#include "cuda_set_builder_utils.h"
#include "cuda_utilities.h"

namespace csdecomp {
namespace {
template <typename T>
size_t find_last_idx(T* arr, size_t size, const T& value) {
  for (size_t i = size;
       i-- > 0;) {  // Start from the last element and move backwards
    if (arr[i] == value) {
      return i;
    }
  }
  return -1;
}
}  // namespace

__global__ void projectSamplesOntoLineSegmentsKernel(
    float* distances, float* projections, const float* samples,
    const float* line_start_points, const float* line_end_points,
    const u_int32_t* line_segment_idxs, const u_int32_t num_samples,
    const u_int32_t dimension) {
  u_int32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample_idx >= num_samples) {
    return;
  }

  const u_int32_t line_segment_idx = *(line_segment_idxs + sample_idx);
  Eigen::Map<const Eigen::VectorXf> q1(
      line_start_points + dimension * line_segment_idx, dimension);
  Eigen::Map<const Eigen::VectorXf> q2(
      line_end_points + dimension * line_segment_idx, dimension);
  Eigen::Map<const Eigen::VectorXf> p(samples + sample_idx * dimension,
                                      dimension);

  float a = (p - q1).dot(q2 - q1);
  float b = (q2 - q1).dot(q2 - q1);
  float t = fminf(fmaxf(a / b, 0.0), 1);

  for (int i = 0; i < dimension; ++i) {
    *(projections + sample_idx * dimension + i) = (1 - t) * q1[i] + t * q2[i];
  }
  Eigen::Map<const Eigen::VectorXf> proj(projections + sample_idx * dimension,
                                         dimension);
  *(distances + sample_idx) = (proj - p).norm();

  //   if (sample_idx == 0) {
  //     printArrayN(q1.data(), dimension);
  //     printArrayN(q2.data(), dimension);
  //     printf("sample \n");
  //     printArrayN(p.data(), dimension);
  //     printf("computed t value %f\n ----------\n", t);
  //     printf("projection \n");
  //     printArrayN(proj.data(), dimension);
  //     printf("projections flat post update\n");
  //     printArrayN(projections, dimension*num_samples);

  //   }
}

void executeProjectSamplesOntoLineSegmentsKernel(
    float* distances, float* projections, const float* samples,
    const float* line_start_points, const float* line_end_points,
    const u_int32_t* line_segment_idxs, const u_int32_t num_samples,
    const u_int32_t dimension) {
  int threads_per_block = 128;
  int num_blocks = (num_samples + threads_per_block - 1) / threads_per_block;

  projectSamplesOntoLineSegmentsKernel<<<num_blocks, threads_per_block>>>(
      distances, projections, samples, line_start_points, line_end_points,
      line_segment_idxs, num_samples, dimension);
}

const std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>>
projectSamplesOntoLineSegmentsCuda(
    const Eigen::MatrixXf& line_start_points,
    const Eigen::MatrixXf& line_end_points,
    const std::vector<Eigen::MatrixXf>& samples) {
  int dim = line_start_points.rows();
  int num_line_segments = line_start_points.cols();

  assert(num_line_segments == line_end_points.cols());
  assert(dim == line_end_points.rows());

  int num_samples = 0;

  for (auto s : samples) {
    num_samples += s.cols();
  }

  float samples_flat[num_samples * dim];
  float projections_flat[num_samples * dim];
  float distances_flat[num_samples];

  u_int32_t line_seg_idxs[num_samples];

  int num_copied = 0, segment_idx = 0;

  for (auto s : samples) {
    memcpy(samples_flat + num_copied * dim, s.data(), s.size() * sizeof(float));
    std::fill(line_seg_idxs + num_copied, line_seg_idxs + num_copied + s.cols(),
              segment_idx);
    num_copied += s.cols();
    ++segment_idx;
  }

  CudaPtr<const float> line_start_pts_ptr(line_start_points.data(),
                                          line_start_points.size());

  CudaPtr<const float> line_end_pts_ptr(line_end_points.data(),
                                        line_end_points.size());
  CudaPtr<const float> samples_ptr(samples_flat, num_samples * dim);
  CudaPtr<const u_int32_t> line_seg_idxs_ptr(line_seg_idxs, num_samples);

  line_start_pts_ptr.copyHostToDevice();
  line_end_pts_ptr.copyHostToDevice();
  samples_ptr.copyHostToDevice();
  line_seg_idxs_ptr.copyHostToDevice();

  // prepare memory for the results
  CudaPtr<float> projections_ptr(projections_flat, num_samples * dim);
  CudaPtr<float> distances_ptr(distances_flat, num_samples);

  // call the projection kernel
  executeProjectSamplesOntoLineSegmentsKernel(
      distances_ptr.device, projections_ptr.device, samples_ptr.device,
      line_start_pts_ptr.device, line_end_pts_ptr.device,
      line_seg_idxs_ptr.device, num_samples, dim);

  // retrieve the results
  projections_ptr.copyDeviceToHost();
  distances_ptr.copyDeviceToHost();

  std::vector<Eigen::MatrixXf> projections;
  std::vector<Eigen::VectorXf> distances;
  int start_offset = 0;
  for (int seg_idx = 0; seg_idx < num_line_segments; ++seg_idx) {
    int last_samp_idx =
        find_last_idx(line_seg_idxs, (size_t)num_samples, (u_int32_t)seg_idx);
    int num_samps_ls = last_samp_idx - start_offset + 1;

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        proj(projections_flat + start_offset * dim, dim, num_samps_ls);
    projections.push_back(proj);

    Eigen::Map<Eigen::VectorXf> dist(distances_flat + start_offset,
                                     num_samps_ls);
    // std::cout << dist << std::endl;

    // std::cout <<fmt::format("projections {}\n", seg_idx)<< proj << std::endl;

    distances.push_back(dist);
    start_offset = last_samp_idx + 1;
  }

  return {distances, projections};
}

__global__ void collisionConsolidationKernel(
    float* configurations_in_collision_device,
    const float* checked_configs_device, const uint8_t* is_config_free_device,
    const u_int32_t max_num_update_particles, const u_int32_t num_configs,
    const u_int32_t dimension, u_int32_t* num_collisions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_configs) {
    if (!is_config_free_device[idx]) {
      // safe incrementing of collision count
      u_int32_t collision_idx = atomicAdd(num_collisions, 1);
      for (int dim = 0; dim < dimension; ++dim) {
        if (collision_idx < max_num_update_particles) {
          configurations_in_collision_device[collision_idx * dimension + dim] =
              checked_configs_device[idx * dimension + dim];
        }
      }
    }
  }
}

const u_int32_t executeCollisionConsolidationKernel(
    float* configurations_in_collision_device,
    const float* checked_configs_device, const uint8_t* is_config_free_device,
    const u_int32_t max_num_update_particles, const u_int32_t num_configs,
    const u_int32_t dimension, u_int32_t* collision_counter) {
  u_int32_t num_collisions = 0;
  // CudaPtr<u_int32_t> num_collisions_ptr(&num_collisions, sizeof(u_int32_t));
  cudaMemcpy((void*)collision_counter, &num_collisions, sizeof(u_int32_t),
             cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;

  collisionConsolidationKernel<<<blocksPerGrid, threadsPerBlock>>>(
      configurations_in_collision_device, checked_configs_device,
      is_config_free_device, max_num_update_particles, num_configs, dimension,
      collision_counter);
  cudaMemcpy((void*)&num_collisions, collision_counter, sizeof(u_int32_t),
             cudaMemcpyDeviceToHost);

  return num_collisions;
}

__global__ void countCollisionsInFirstMEntriesKernel(
    const uint8_t* is_config_free_device, const u_int32_t M,
    u_int32_t* num_collisions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M) {
    if (!is_config_free_device[idx]) {
      // safe incrementing of collision count
      u_int32_t collision_idx = atomicAdd(num_collisions, 1);
    }
  }
}

const u_int32_t executeCountCollisionsInFirstMEntriesKernel(
    const uint8_t* is_config_free_device, const u_int32_t M,
    const u_int32_t num_configs, u_int32_t* first_m_collision_counter) {
  u_int32_t num_collisions = 0;
  cudaMemcpy((void*)first_m_collision_counter, &num_collisions,
             sizeof(u_int32_t), cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;

  countCollisionsInFirstMEntriesKernel<<<blocksPerGrid, threadsPerBlock>>>(
      is_config_free_device, M, first_m_collision_counter);
  cudaMemcpy((void*)&num_collisions, first_m_collision_counter,
             sizeof(u_int32_t), cudaMemcpyDeviceToHost);

  return num_collisions;
}

__global__ void stepConfigToMiddleKernel(float* results_device,
                                         const float* start_points_device,
                                         const float* end_points_device,
                                         const u_int32_t num_configs,
                                         const u_int32_t dimension) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_configs) {
    for (int i = 0; i < dimension; ++i) {
      int loc = idx * dimension + i;
      results_device[loc] =
          0.5 * (start_points_device[loc] + end_points_device[loc]);
    }
  }
}

void executeStepConfigToMiddleKernel(float* results_device,
                                     const float* start_points_device,
                                     const float* end_points_device,
                                     const u_int32_t num_configs,
                                     const u_int32_t dimension) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;
  stepConfigToMiddleKernel<<<blocksPerGrid, threadsPerBlock>>>(
      results_device, start_points_device, end_points_device, num_configs,
      dimension);
}

__global__ void updateBisectionBoundsKernel(
    float* bisection_lower_bound_device, float* bisection_upper_bound_device,
    const float* currently_checked_config, uint8_t* is_config_col_free,
    u_int32_t num_configs, u_int32_t dimension) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_configs) {
    // move the upper bound to the current configuration if it is in collision
    // upper bound is initialized at the starting configuration
    if (!is_config_col_free[idx]) {
      for (int i = 0; i < dimension; ++i) {
        int loc = idx * dimension + i;
        bisection_upper_bound_device[loc] = currently_checked_config[loc];
      }
    } else {
      // move the lower bound to the current configuration if it is
      // collision-free lower bound is initialized at the projection
      for (int i = 0; i < dimension; ++i) {
        int loc = idx * dimension + i;
        bisection_lower_bound_device[loc] = currently_checked_config[loc];
      }
    }
  }
}

void executeUpdateBisectionBoundsKernel(float* bisection_lower_bound_device,
                                        float* bisection_upper_bound_device,
                                        const float* currently_checked_configs,
                                        uint8_t* is_config_col_free,
                                        u_int32_t num_configs,
                                        u_int32_t dimension) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;
  updateBisectionBoundsKernel<<<blocksPerGrid, threadsPerBlock>>>(
      bisection_lower_bound_device, bisection_upper_bound_device,
      currently_checked_configs, is_config_col_free, num_configs, dimension);
}

__global__ void storeIfCollisionKernel(float* configs_in_collision,
                                       const float* current_configs,
                                       uint8_t* is_current_config_col_free,
                                       u_int32_t num_configs,
                                       u_int32_t dimension) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_configs) {
    // move the upper bound to the current configuration if it is in collision
    // upper bound is initialized at the starting configuration
    if (!is_current_config_col_free[idx]) {
      for (int i = 0; i < dimension; ++i) {
        int loc = idx * dimension + i;
        configs_in_collision[loc] = current_configs[loc];
      }
    }
  }
}

void executeStoreIfCollisionKernel(float* configs_in_collision,
                                   const float* current_configs,
                                   uint8_t* is_current_config_col_free,
                                   u_int32_t num_configs, u_int32_t dimension) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;
  storeIfCollisionKernel<<<blocksPerGrid, threadsPerBlock>>>(
      configs_in_collision, current_configs, is_current_config_col_free,
      num_configs, dimension);
}

__global__ void pointToPointDistanceKernel(float* distances_device,
                                           const float* pointsA_device,
                                           const float* pointsB_device,
                                           u_int32_t num_configs,
                                           u_int32_t dimension) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_configs) {
    Eigen::Map<const Eigen::VectorXf> ptA(pointsA_device + idx * dimension,
                                          dimension);
    Eigen::Map<const Eigen::VectorXf> ptB(pointsB_device + idx * dimension,
                                          dimension);
    distances_device[idx] = (ptA - ptB).norm();
  }
}

void executePointToPointDistanceKernel(float* distances_device,
                                       const float* points_a_device,
                                       const float* points_b_device,
                                       u_int32_t num_configs,
                                       u_int32_t dimension) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;
  pointToPointDistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(
      distances_device, points_a_device, points_b_device, num_configs,
      dimension);
}

__global__ void maybePlaceFace(MinimalHPolyhedron* p_device, uint8_t* success,
                               float* dist, const float* point,
                               const float* proj, const float* line_start,
                               const float* line_end, const float stepback) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (*dist < FLT_MAX) {
      if (p_device->num_faces > MAX_NUM_FACES) {
        *success = 2;
      } else {
        // particle is non-redundant, place face, use static preallocated memory
        float a_data[MAX_DIM];
        Eigen::Map<Eigen::VectorXf> a(a_data, p_device->dim);
        Eigen::Map<const Eigen::VectorXf> collision(point, p_device->dim);
        Eigen::Map<const Eigen::VectorXf> projection(proj, p_device->dim);
        Eigen::Map<const Eigen::VectorXf> ls(line_start, p_device->dim);
        Eigen::Map<const Eigen::VectorXf> le(line_end, p_device->dim);
        a = collision - projection;
        a.normalize();
        float b = a.dot(collision) - stepback;

        // assess if we need to relax the cspace margin
        float v1 = ls.dot(a) - b;
        float v2 = le.dot(a) - b;
        float relaxation = v2 > v1 ? v2 : v1;
        relaxation = relaxation > 0 ? relaxation : 0;
        b += relaxation;
        u_int32_t num_f = p_device->num_faces;

        Eigen::Map<Eigen::VectorXf> face_dest(
            &(p_device->A[num_f * p_device->dim]), p_device->dim);
        face_dest = a;
        p_device->b[num_f] = b;
        p_device->num_faces += 1;

        // make the particle redundant
        *dist = FLT_MAX;
        *success = 1;
      }
    } else {
      *success = 0;
    }
  }
}
__global__ void checkParticlesMadeRedundantByLastFace(
    float* distances, const MinimalHPolyhedron* p_device,
    const float* points_in_collision, const u_int32_t num_configs,
    const u_int32_t dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_configs) {
    if (distances[idx] == FLT_MAX) {
      return;
    }
    u_int32_t num_f = p_device->num_faces;
    u_int32_t dim = p_device->dim;
    // get the last face
    Eigen::Map<const Eigen::VectorXf> face_dest(
        &(p_device->A[(num_f - 1) * dim]), dim);
    Eigen::Map<const Eigen::VectorXf> particle(points_in_collision + idx * dim,
                                               dim);
    float face_val = face_dest.dot(particle) - p_device->b[num_f - 1];
    // particle lies outside of the face
    if (face_val > 0) {
      distances[idx] = FLT_MAX;
    }
  }
}

uint8_t executeAddFaceAndCheckOtherParticleRedundantKernel(
    MinimalHPolyhedron* p_device, float* distances, float* points_in_collision,
    const float* projections, const float* line_start, const float* line_end,
    const u_int32_t particle_idx_to_add, const u_int32_t num_configs,
    const u_int32_t dim, const float stepback, uint8_t* success_flag_device) {
  u_int8_t success = 0;
  cudaMemcpy((void*)success_flag_device, &success, sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  u_int32_t offset = particle_idx_to_add * dim;
  maybePlaceFace<<<1, 1>>>(p_device, success_flag_device,
                           distances + particle_idx_to_add,
                           points_in_collision + offset, projections + offset,
                           line_start, line_end, stepback);
  cudaMemcpy((void*)&success, success_flag_device, sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  if (success == 1) {
    // check other particles redundant kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_configs + threadsPerBlock - 1) / threadsPerBlock;
    checkParticlesMadeRedundantByLastFace<<<blocksPerGrid, threadsPerBlock>>>(
        distances, p_device, points_in_collision, num_configs, dim);

    return 0;
  }
  if (success == 2) {
    return 2;
  } else {
    return 1;
  }
}
}  // namespace csdecomp