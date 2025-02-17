
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>

#include <cassert>
#include <cfloat>
#include <chrono>
#include <iostream>

#include "cuda_hit_and_run_sampling.h"
#include "cuda_utilities.h"
#include "hpolyhedron.h"

namespace csdecomp {

__global__ void uniformSampleInHPolyhedronKernel(
    float* samples, const MinimalHPolyhedron* poly_array,
    const float* starting_points, const u_int32_t num_samples_per_hpolyhedron,
    const u_int32_t dim, const u_int32_t num_polytopes,
    const u_int32_t mixing_steps, const u_int64_t seed) {
  u_int32_t cu_index = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(seed, cu_index, 0, &state);

  if (cu_index >= num_polytopes * num_samples_per_hpolyhedron) {
    return;
  }

  u_int32_t hpoly_idx = cu_index / num_samples_per_hpolyhedron;
  u_int32_t sample_idx = cu_index % num_samples_per_hpolyhedron;

  // collect A, b matrices
  const int32_t num_faces = poly_array[hpoly_idx].num_faces;

  // TODO(wernerpe): This might be super inefficient no idea how to improve
  // Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  //                                Eigen::ColMajor>>
  //     A(poly_array[hpoly_idx].A, num_faces, dim);
  const float* A_data = poly_array[hpoly_idx].A;

  // Eigen::Map<const Eigen::Vector<float, Eigen::Dynamic>> b(
  //     poly_array[hpoly_idx].b, num_faces);
  const float* b_data = poly_array[hpoly_idx].b;
  // Eigen::Map<const Eigen::Vector<float, Eigen::Dynamic>> starting_point(
  //     starting_points + hpoly_idx * dim, dim);
  const float* starting_point_data = starting_points + hpoly_idx * dim;
  // Eigen::Vector<float, Eigen::Dynamic>current_sample(starting_point);
  // float max_coef = (A * starting_point - b).maxCoeff();
  // // Eigen::MatrixXf res = A * starting_point;
  float current_sample[MAX_DIM];
  for (int i = 0; i < dim; ++i) {
    current_sample[i] = *(starting_points + hpoly_idx * dim + i);
  }

  float ramF_faces[MAX_NUM_FACES];
  float ramF2_faces[MAX_NUM_FACES];
  float ramF_dim[MAX_DIM];

  // matmulFRowMajor(ramF_faces, A_data, starting_point_data, num_faces, dim,
  // 1);

  // this check is commented out because it is very inefficient
  // float max_coef = findMaxAMinusB(ramF_faces, b_data, num_faces);
  // // float max_coef = (A * starting_point - b).maxCoeff();
  // if (max_coef > 0) {
  //   printf(
  //       "[uniformSampleInHPolyhedronKernel] Error! Starting point not "
  //       "contained in region %d\n",
  //       hpoly_idx);
  // }

  for (int i = 0; i < mixing_steps; ++i) {
    // Sample direction
    for (int i = 0; i < dim; ++i) {
      ramF_dim[i] = curand_normal(&state);
    }

    // computes dunno how do do this any better
    // Eigen::VectorXf line_b = b - A * current_sample;
    matmulFRowMajor(ramF_faces, A_data, current_sample, num_faces, dim, 1);
    vectorAMinusB(ramF_faces, b_data, ramF_faces, num_faces);
    // Eigen::VectorXf line_A = A * direction; direction is in ramF_dim
    matmulFRowMajor(ramF2_faces, A_data, ramF_dim, num_faces, dim, 1);

    // ramF has line_b, ramF2 has line_A, ramF_dim has direction
    float theta_max = FLT_MAX;
    float theta_min = -FLT_MAX;
    for (int i = 0; i < num_faces; ++i) {
      if (ramF2_faces[i] < 0.0f) {
        theta_min = max(theta_min, ramF_faces[i] / ramF2_faces[i]);
      } else if (ramF2_faces[i] > 0.0f) {
        theta_max = min(theta_max, ramF_faces[i] / ramF2_faces[i]);
      }
    }

    if (isinf(theta_max) || isinf(theta_min) || theta_max < theta_min) {
      // printf("The Hit and Run algorithm failed for sample %d in polytope %d
      // \n",
      //        sample_idx, hpoly_idx);
      return;
    }
    // Pick θ uniformly from [θ_min, θ_max)
    float theta =
        curand_uniform(&state) * (0.999999 * theta_max - theta_min) + theta_min;

    // current_sample = current_sample + theta * direction;
    for (int i = 0; i < dim; ++i) {
      current_sample[i] = current_sample[i] + theta * ramF_dim[i];
    }
  }
  for (int i = 0; i < dim; ++i) {
    samples[num_samples_per_hpolyhedron * dim * hpoly_idx + dim * sample_idx +
            i] = current_sample[i];
  }
  return;
}

void executeUniformSampleInHPolyhedronKernel(
    float* samples, const MinimalHPolyhedron* poly_array,
    const float* starting_points, const int32_t num_samples_per_hpolyhedron,
    const int32_t dim, const int32_t num_polytopes, const int32_t mixing_steps,
    u_int64_t seed) {
  int threads_per_block = 128;
  int num_blocks =
      (num_polytopes * num_samples_per_hpolyhedron + threads_per_block - 1) /
      threads_per_block;

  uniformSampleInHPolyhedronKernel<<<num_blocks, threads_per_block>>>(
      samples, poly_array, starting_points, num_samples_per_hpolyhedron, dim,
      num_polytopes, mixing_steps, seed);
}

// all Hpolyhedra must live in the same dimension
const std::vector<Eigen::MatrixXf> UniformSampleInHPolyhedronCuda(
    const std::vector<HPolyhedron>& polyhedra,
    const Eigen::MatrixXf& starting_points,
    const int num_samples_per_hpolyhedron, const int mixing_steps,
    const int64_t seed) {
  MinimalHPolyhedron poly_array[polyhedra.size()];

  int dim = polyhedra.at(0).ambient_dimension();
  int num_poly = polyhedra.size();

  if (starting_points.rows() != dim) {
    throw std::runtime_error("starting points have wrong dimension!");
  }
  if (starting_points.cols() != num_poly) {
    throw std::runtime_error(
        fmt::format("Incorrect number of starting points supplied. A total "
                    "of {} polyhedra "
                    "and {} starting points were passed to the sampler. ",
                    num_poly, starting_points.cols()));
  }

  int idx = 0;
  for (auto p : polyhedra) {
    if (!p.PointInSet(starting_points.col(idx))) {
      throw std::runtime_error(
          fmt::format("starting point {} is not contained in polytope {}. The "
                      "initial point must lie inside of the polytope.",
                      idx, idx));
    }
    MinimalHPolyhedron mp = p.GetMyMinimalHPolyhedron();
    // perform deep copy of vector of polyhedra
    memcpy(poly_array[idx].A, mp.A, sizeof(float) * mp.dim * mp.num_faces);
    memcpy(poly_array[idx].b, mp.b, sizeof(float) * mp.num_faces);
    poly_array[idx].num_faces = mp.num_faces;
    poly_array[idx].dim = mp.dim;
    ++idx;
  }

  std::vector<float> samples_raw(num_poly * num_samples_per_hpolyhedron * dim);

  // wrangle GPU memory
  CudaPtr<const MinimalHPolyhedron> hpoly_ptr(poly_array, num_poly);
  CudaPtr<const float> starting_pts_ptr(starting_points.data(),
                                        starting_points.size());
  CudaPtr<float> samples_ptr(samples_raw.data(),
                             num_samples_per_hpolyhedron * dim * num_poly);

  hpoly_ptr.copyHostToDevice();
  starting_pts_ptr.copyHostToDevice();
  auto start = std::chrono::high_resolution_clock::now();

  executeUniformSampleInHPolyhedronKernel(
      samples_ptr.device, hpoly_ptr.device, starting_pts_ptr.device,
      num_samples_per_hpolyhedron, dim, num_poly, mixing_steps,
      (u_int64_t)seed);
  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();

  std::string tmp = fmt::format(
      "sampling duration (no copy): {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  samples_ptr.copyDeviceToHost();
  std::vector<Eigen::MatrixXf> samples;
  for (int idx = 0; idx < num_poly; ++idx) {
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        s(samples_ptr.host + idx * num_samples_per_hpolyhedron * dim, dim,
          num_samples_per_hpolyhedron);
    samples.push_back(s);
  }
  return samples;
}
}  // namespace csdecomp