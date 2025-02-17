#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <numeric>

#include "cuda_collision_checker.h"
#include "cuda_forward_kinematics.h"
#include "cuda_utilities.h"
#include "cuda_visibility_graph.h"

namespace csdecomp {
namespace {

// Converts flattened index of upper triangular part of square matrix of size n
// to index pair i,j. Both indices are 0-based. This assumes the flattened array
// does not contain the diagonal.
__device__ void get_indices_from_flat_index(uint32_t& i, uint32_t& j,
                                            const uint64_t& flat_index,
                                            const uint32_t& n) {
  // Calculate row (i) and column (j) from the position
  i = n - 2 - floorf(sqrtf(-8 * flat_index + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
  j = flat_index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
}

std::pair<uint32_t, uint32_t> get_indices_from_flat_index_cpp(
    const uint64_t& flat_index, const uint32_t& n) {
  uint32_t i =
      n - 2 -
      std::floor(std::sqrt(-8 * flat_index + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
  uint32_t j =
      flat_index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;

  return std::make_pair(i, j);
}

__global__ void spymemoryI(const uint8_t* array, const u_int32_t N) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("spymemory \n");
    printArrayN(array, N);
    printf("\n------------\n end spymemory \n");
  }
}
__global__ void spymemoryF(const float* array, const u_int32_t N) {
  u_int32_t cu_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_idx == 0) {
    printf("spymemory \n");
    printArrayN(array, N);
    printf("\n------------\n end spymemory \n");
  }
}

__global__ void numConfigsInEdgeKernel(uint32_t* num_configs_to_check_in_edge,
                                       const float* configurations,
                                       const int num_configurations,
                                       const int dim, const float step_size) {
  uint64_t flat_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_index >= (num_configurations * (num_configurations - 1)) / 2) return;
  uint32_t i, j;
  get_indices_from_flat_index(i, j, flat_index, num_configurations);
  //   printf("i %d, j %d \n", i,j);
  Eigen::Map<const Eigen::VectorXf> conf_a(configurations + i * dim, dim);
  Eigen::Map<const Eigen::VectorXf> conf_b(configurations + j * dim, dim);
  float dist = (conf_a - conf_b).norm();
  uint32_t num_configs_to_check = dist / step_size + 2;
  num_configs_to_check_in_edge[flat_index] = num_configs_to_check;
}
__global__ void poolEdgeCheckResultsKernel(
    uint8_t* edge_batch_col_free_buffer,
    const uint8_t* edge_batch_configs_col_free,
    const uint64_t* cumsum_num_configs_edge, const uint64_t starting_edge_index,
    const int num_edges_per_batch) {
  uint64_t edge_index_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index_local >= num_edges_per_batch) return;
  uint64_t edge_index = starting_edge_index + edge_index_local;

  uint32_t num_configs_to_pool;
  if (edge_index == 0) {
    num_configs_to_pool = cumsum_num_configs_edge[edge_index];
  } else {
    num_configs_to_pool = cumsum_num_configs_edge[edge_index] -
                          cumsum_num_configs_edge[edge_index - 1];
  }

  uint64_t upper = cumsum_num_configs_edge[edge_index] - num_configs_to_pool;
  uint64_t lower = (starting_edge_index == 0)
                       ? 0
                       : cumsum_num_configs_edge[starting_edge_index - 1];
  uint64_t pooling_start_index = upper - lower;

  //   if (edge_index_local == 4) {
  //     uint32_t i, j;
  //     get_indices_from_flat_index(i, j, edge_index, 5);

  //     printf("edge_index %d \n", edge_index);
  //     printf("i %d, j %d \n", i, j);
  //     printf("starting_edge_index %d \n", starting_edge_index);
  //     printf("edge config collision free \n");
  //     printf("num configs to pool %d \n", num_configs_to_pool);
  //     printf(" pooling start index %d\n", pooling_start_index);
  //     printArrayN(edge_batch_configs_col_free + pooling_start_index,
  //                 num_configs_to_pool);
  //   }

  for (int step = 0; step < num_configs_to_pool; ++step) {
    if (!edge_batch_configs_col_free[pooling_start_index + step]) {
      edge_batch_col_free_buffer[edge_index_local] = 0;
      //   if (edge_index_local == 3) printf("return false\n");
      return;
    }
  }
  edge_batch_col_free_buffer[edge_index_local] = 1;
  //   if (edge_index_local == 3) printf("return true\n");

  return;
}

__global__ void interpolateEdgesKernel(float* edge_configs_buffer,
                                       const float* graph_nodes,
                                       const uint64_t* cumsum_num_configs_edge,
                                       const uint64_t starting_edge_index,
                                       const int num_nodes,
                                       const uint64_t num_edges_per_batch,
                                       const int dim, const float step_size) {
  // This fills in the edge configs
  // |start|end|intermediate 0| intermediate 1|...intermediate N|
  uint64_t edge_index_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_index_local >= num_edges_per_batch) return;
  uint32_t i, j;
  const uint64_t edge_index = edge_index_local + starting_edge_index;
  get_indices_from_flat_index(i, j, edge_index, num_nodes);
  //   printf("interp kernel i %d, j %d \n", i, j);
  Eigen::Map<const Eigen::VectorXf> conf_a(graph_nodes + i * dim, dim);
  Eigen::Map<const Eigen::VectorXf> conf_b(graph_nodes + j * dim, dim);
  uint32_t config_writing_idx, num_intermediate_checks;
  if (edge_index == 0) {
    config_writing_idx = 0;
    num_intermediate_checks = cumsum_num_configs_edge[0] - 2;
  } else {
    uint64_t upper = cumsum_num_configs_edge[edge_index - 1];
    uint64_t lower = (starting_edge_index == 0)
                         ? 0
                         : cumsum_num_configs_edge[starting_edge_index - 1];
    config_writing_idx = (upper - lower) * dim;
    num_intermediate_checks = cumsum_num_configs_edge[edge_index] -
                              cumsum_num_configs_edge[edge_index - 1] - 2;
  }
  //   printf("ij2 i %d, j %d \n", i, j);
  //   printf("edge index %llu, i %u, j %u, num checks %u, config_writing_idx
  //   %u\n",
  //  edge_index, i, j, num_intermediate_checks, config_writing_idx);

  // write start and end configuration
  for (int i = 0; i < dim; ++i) {
    edge_configs_buffer[config_writing_idx + i] = conf_a[i];
    edge_configs_buffer[config_writing_idx + i + dim] = conf_b[i];
  }

  const float t_step = 1.0 / (num_intermediate_checks + 1);
  for (int idx1 = 0; idx1 < num_intermediate_checks; idx1++) {
    float t = t_step * (idx1 + 1);
    for (int idx2 = 0; idx2 < dim; ++idx2) {
      edge_configs_buffer[config_writing_idx + 2 * dim + idx1 * dim + idx2] =
          t * conf_b[idx2] + (1 - t) * conf_a[idx2];
    }
  }
}
}  // namespace

Eigen::SparseMatrix<uint8_t> VisibilityGraph(
    const Eigen::MatrixXf& configurations, const MinimalPlant& mplant,
    const float step_size, const uint64_t max_configs_per_batch) {
  const int dim = mplant.kin_tree.num_configuration_variables;
  const int num_configurations = configurations.cols();
  assert(dim == configurations.rows());
  assert(num_configurations >= 2);
  assert(max_configs_per_batch > 1);
  assert(max_configs_per_batch < 10000001);

  assert(step_size >= 1e-12);
  // figure out memory requirement
  const int num_edges_to_check =
      num_configurations * (num_configurations - 1) / 2;

  //   // check if nodes themselves are collision free
  //   std::vector<uint8_t> is_col_free_nodes =
  //       checkCollisionFreeCuda(&configurations, &mplant);

  // compute num_configs_to_check_per_edge
  std::vector<uint32_t> num_configs_to_check_in_edge(num_edges_to_check, 0);
  std::vector<uint64_t> cumsum_configs_to_check_in_edge(num_edges_to_check, 0);

  CudaPtr<const float> configurations_ptr(configurations.data(),
                                          configurations.size());

  CudaPtr<uint32_t> num_configs_to_check_in_edge_ptr(
      num_configs_to_check_in_edge.data(), num_configs_to_check_in_edge.size());

  configurations_ptr.copyHostToDevice();
  int threads_per_block = 128;
  int num_blocks =
      (num_edges_to_check + threads_per_block - 1) / threads_per_block;

  numConfigsInEdgeKernel<<<num_blocks, threads_per_block>>>(
      num_configs_to_check_in_edge_ptr.device, configurations_ptr.device,
      num_configurations, dim, step_size);

  num_configs_to_check_in_edge_ptr.copyDeviceToHost();

  std::partial_sum(num_configs_to_check_in_edge.begin(),
                   num_configs_to_check_in_edge.end(),
                   cumsum_configs_to_check_in_edge.begin());

  const uint64_t number_edge_configs = cumsum_configs_to_check_in_edge.back();
  //   std::cout << "num configs to check :";
  //   for (const auto& n : num_configs_to_check_in_edge) {
  //     std::cout << n << ", ";
  //   }
  //   std::cout << std::endl;
  //   std::cout << "cumsum :";
  //   for (const auto& n : cumsum_configs_to_check_in_edge) {
  //     std::cout << n << ", ";
  //   }
  //   std::cout << std::endl;

  // load balancing
  auto it = std::max_element(num_configs_to_check_in_edge.begin(),
                             num_configs_to_check_in_edge.end());
  const int max_num_configs_per_edge = *it;
  const int num_edges_per_batch =
      std::min(num_edges_to_check,
               (int)(max_configs_per_batch / max_num_configs_per_edge + 1));

  const int num_batches = num_edges_to_check / num_edges_per_batch;
  const int num_edges_in_last_batch =
      num_edges_to_check - num_batches * num_edges_per_batch;

  const int batch_size = max_num_configs_per_edge * num_edges_per_batch;

  CudaPtr<float> edge_configs_batch(nullptr, batch_size * dim);
  CudaPtr<uint8_t> edge_configs_batch_col_free(nullptr, batch_size);

  // collect overall adjacency information here
  std::vector<uint8_t> edge_col_free(num_edges_to_check, 0);
  std::vector<uint8_t> edges_batch_col_free(batch_size, 0);

  CudaPtr<uint8_t> edges_batch_col_free_ptr(edges_batch_col_free.data(),
                                            num_edges_per_batch);
  CudaPtr<const uint64_t> cumsum_configs_to_check_in_edge_ptr(
      cumsum_configs_to_check_in_edge.data(),
      cumsum_configs_to_check_in_edge.size());
  cumsum_configs_to_check_in_edge_ptr.copyHostToDevice();

  // collision checker memory
  CudaPtr<const MinimalPlant> plant_ptr(&mplant, 1);
  plant_ptr.copyHostToDevice();
  CudaPtr<float> transforms_ptr(nullptr,
                                16 * batch_size * mplant.kin_tree.num_links);
  CudaPtr<uint8_t> is_pair_col_free_ptr(
      nullptr, mplant.num_collision_pairs * batch_size);
  printf(
      "num batches %d \n num_edges_per_batch %d \n, num_edges_to_check %d \n",
      num_batches, num_edges_per_batch, num_edges_to_check);
  for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // printf("batch idx %d", batch_idx);
    threads_per_block = 128;
    num_blocks =
        (num_edges_per_batch + threads_per_block - 1) / threads_per_block;
    uint64_t starting_edge_index = batch_idx * num_edges_per_batch;

    interpolateEdgesKernel<<<num_blocks, threads_per_block>>>(
        edge_configs_batch.device, configurations_ptr.device,
        cumsum_configs_to_check_in_edge_ptr.device, starting_edge_index,
        num_configurations, num_edges_per_batch, dim, step_size);

    // printf("edge index 4 configs\n");
    // spymemoryF<<<1, 1>>>(edge_configs_batch.device, 100);
    // cudaDeviceSynchronize();

    uint64_t upper = cumsum_configs_to_check_in_edge[starting_edge_index - 1 +
                                                     num_edges_per_batch];
    uint64_t lower;
    if (starting_edge_index == 0) {
      lower = 0;
    } else {
      lower = cumsum_configs_to_check_in_edge[starting_edge_index - 1];
    }
    uint64_t num_configurations_to_check = upper - lower;

    executeForwardKinematicsKernel(
        transforms_ptr.device, edge_configs_batch.device,
        num_configurations_to_check, &(plant_ptr.device->kin_tree));

    executeCollisionFreeKernel(
        edge_configs_batch_col_free.device, is_pair_col_free_ptr.device,
        plant_ptr.device, transforms_ptr.device, num_configurations_to_check,
        mplant.num_collision_pairs);

    // make cuda index correspond to the local edge index
    poolEdgeCheckResultsKernel<<<num_blocks, threads_per_block>>>(
        edges_batch_col_free_ptr.device, edge_configs_batch_col_free.device,
        cumsum_configs_to_check_in_edge_ptr.device, starting_edge_index,
        num_edges_per_batch);

    // copy batch into global edge info array
    cudaMemcpy((void*)(edge_col_free.data() + batch_idx * num_edges_per_batch),
               edges_batch_col_free_ptr.device,
               num_edges_per_batch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  }

  // Handle remaining edges if any are left.
  if (num_edges_in_last_batch) {
    uint64_t starting_edge_index = num_batches * num_edges_per_batch;
    threads_per_block = 128;
    num_blocks =
        (num_edges_in_last_batch + threads_per_block - 1) / threads_per_block;
    interpolateEdgesKernel<<<num_blocks, threads_per_block>>>(
        edge_configs_batch.device, configurations_ptr.device,
        cumsum_configs_to_check_in_edge_ptr.device, starting_edge_index,
        num_configurations, num_edges_in_last_batch, dim, step_size);

    uint64_t num_configurations_to_check =
        cumsum_configs_to_check_in_edge[starting_edge_index - 1 +
                                        num_edges_in_last_batch] -
        cumsum_configs_to_check_in_edge[starting_edge_index - 1];

    executeForwardKinematicsKernel(
        transforms_ptr.device, edge_configs_batch.device,
        num_configurations_to_check, &(plant_ptr.device->kin_tree));

    executeCollisionFreeKernel(
        edge_configs_batch_col_free.device, is_pair_col_free_ptr.device,
        plant_ptr.device, transforms_ptr.device, num_configurations_to_check,
        mplant.num_collision_pairs);

    poolEdgeCheckResultsKernel<<<num_blocks, threads_per_block>>>(
        edges_batch_col_free_ptr.device, edge_configs_batch_col_free.device,
        cumsum_configs_to_check_in_edge_ptr.device, starting_edge_index,
        num_edges_in_last_batch);

    cudaMemcpy(
        (void*)(edge_col_free.data() + num_batches * num_edges_per_batch),
        edges_batch_col_free_ptr.device,
        num_edges_in_last_batch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<Eigen::Triplet<uint8_t>> triplets;
  triplets.reserve(num_edges_to_check);
  for (int i = 0; i < num_edges_to_check; ++i) {
    if (edge_col_free[i]) {
      auto [row, col] = get_indices_from_flat_index_cpp(i, num_configurations);
      triplets.emplace_back(row, col, 1);
      //   adjacency_matrix.coeffRef(row, col) = 1;
      //   adjacency_matrix.coeffRef(col, row) = 1;
    }
  }
  Eigen::SparseMatrix<uint8_t> adjacency_matrix(num_configurations,
                                                num_configurations);
  adjacency_matrix.setFromTriplets(triplets.begin(), triplets.end());

  auto stop = std::chrono::high_resolution_clock::now();
  std::string tmp = fmt::format(
      "time to insert results in matrix: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  return adjacency_matrix;
}
}  // namespace csdecomp