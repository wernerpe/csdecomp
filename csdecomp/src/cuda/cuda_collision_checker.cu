
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "cuda_collision_checker.h"
#include "cuda_forward_kinematics.h"
#include "cuda_geometry_utilities.h"
#include "cuda_utilities.h"
namespace csdecomp {
__global__ void pairCollisionFreeKernel(
    uint8_t *is_pair_col_free_device,
    const CollisionGeometry *scene_geometries_device,
    const GeometryIndex *collision_pair_matrix_device,
    const float *transforms_device, const MinimalKinematicTree *tree_device,
    const int num_configurations, const int num_collision_pairs) {
  // Cu index is indexing the uint8_t vector `is_pair_col_free_device` which
  // says if a collision pair is collision free. The num_collision_pairs x
  // num_configurations results are concatenated as follows:
  // |pair1 pair2 pair3 ... | |pair1 pair2 pair3 ... | |pair1 pair2 pair3 ... |
  //
  // The transforms X_W_L are flattened and concatenated by configuration as
  // follows (dimension is 16*num_links*num_configurations):
  // |X_W_L1 X_W_L2 ... X_W_LN | X_W_L1 ... X_W_LN | X_W_L1 X_W_L2 ... X_W_LN |

  int cu_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_index >= num_configurations * num_collision_pairs) {
    return;
  }
  uint8_t *result = is_pair_col_free_device + cu_index;
  const int collision_pair_index = cu_index % num_collision_pairs;
  const int geomA_idx =
      *(collision_pair_matrix_device + 2 * collision_pair_index);
  const int geomB_idx =
      *(collision_pair_matrix_device + 2 * collision_pair_index + 1);
  //   printf("collision_pair_matrix device \n");
  //   printArrayN(collision_pair_matrix_device, 4);
  const int configuration_idx = cu_index / num_collision_pairs;
  //   printf("config %d result location %d pair %d, geomA_idx %d, geomB_idx
  //   %d\n",
  //          configuration_idx, cu_index, collision_pair_index, geomA_idx,
  //          geomB_idx);

  // Get index of the transformations. They are concatenated first by links,
  // then by configurations.
  const CollisionGeometry *geomA = scene_geometries_device + geomA_idx;
  const CollisionGeometry *geomB = scene_geometries_device + geomB_idx;
  const int X_W_LA_idx =
      geomA->link_index + tree_device->num_links * configuration_idx;
  const int X_W_LB_idx =
      geomB->link_index + tree_device->num_links * configuration_idx;
  const Eigen::Matrix4f X_W_LA =
      Eigen::Map<const Eigen::Matrix4f>(transforms_device + 16 * X_W_LA_idx);
  const Eigen::Matrix4f X_W_LB =
      Eigen::Map<const Eigen::Matrix4f>(transforms_device + 16 * X_W_LB_idx);
  // do the collision check
  *result = isPairCollisionFree(geomA, &X_W_LA, geomB, &X_W_LB);
}

__global__ void poolPairCollisionFreeResultsKernel(
    uint8_t *is_config_collision_free_device,
    const uint8_t *is_pair_collision_free_device, const int num_configurations,
    const int num_collision_pairs) {
  int configuration_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (configuration_index >= num_configurations) {
    return;
  }

  for (int pair_idx = 0; pair_idx < num_collision_pairs; ++pair_idx) {
    if (!is_pair_collision_free_device[configuration_index *
                                           num_collision_pairs +
                                       pair_idx]) {
      is_config_collision_free_device[configuration_index] = false;
      return;
    }
  }
  is_config_collision_free_device[configuration_index] = true;
}

__global__ void poolGeomVoxelPairCollisionFreeResultsKernel(
    uint8_t *is_config_collision_free_device,
    const uint8_t *is_geom_to_vox_pair_col_free_device,
    const int num_configurations, const int num_geom_to_vox_pairs) {
  int configuration_index = blockIdx.x * blockDim.x + threadIdx.x;
  // catch if self collision already happened
  if (configuration_index >= num_configurations ||
      !is_config_collision_free_device[configuration_index]) {
    return;
  }

  for (int pair_idx = 0; pair_idx < num_geom_to_vox_pairs; ++pair_idx) {
    if (!is_geom_to_vox_pair_col_free_device[configuration_index *
                                                 num_geom_to_vox_pairs +
                                             pair_idx]) {
      is_config_collision_free_device[configuration_index] = false;
      return;
    }
  }
  is_config_collision_free_device[configuration_index] = true;
}

__global__ void poolVoxelGeomPairCollisionFreeResultsKernel(
    uint8_t *is_config_voxel_collision_free_device,
    const uint8_t *is_geom_to_vox_pair_col_free_device,
    const int num_configurations, const int num_robot_geometries,
    const int num_voxels) {
  int64_t configuration_index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t voxel_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t num_geom_to_vox_pairs = num_robot_geometries * num_voxels;
  if (voxel_index >= num_voxels) return;
  if (configuration_index >= num_configurations) return;

  for (int pair_idx = 0; pair_idx < num_robot_geometries; ++pair_idx) {
    if (!is_geom_to_vox_pair_col_free_device[configuration_index *
                                                 num_geom_to_vox_pairs +
                                             pair_idx * num_voxels +
                                             voxel_index]) {
      is_config_voxel_collision_free_device[configuration_index * num_voxels +
                                            voxel_index] = false;
      return;
    }
  }
  is_config_voxel_collision_free_device[configuration_index * num_voxels +
                                        voxel_index] = true;
}

__global__ void geomToVoxelPairCollisionFreeKernel(
    uint8_t *is_geom_to_vox_pair_col_free_device, const float *voxels,
    const float voxel_radius, const CollisionGeometry *scene_geometries_device,
    const GeometryIndex *robot_geometry_ids, const float *transforms_device,
    const MinimalKinematicTree *tree_device, const int num_configurations,
    const int num_robot_geometries, const int num_voxels) {
  // Cu index is indexing the uint8_t vector `is_pair_col_free_device` which
  // says if a collision pair is collision free. The num_collision_pairs x
  // num_configurations results are concatenated as follows:
  // |pair1 pair2 pair3 ... | |pair1 pair2 pair3 ... | |pair1 pair2 pair3 ... |
  //
  // The transforms X_W_L are flattened and concatenated by configuration as
  // follows (dimension is 16*num_links*num_configurations):
  // |X_W_L1 X_W_L2 ... X_W_LN | X_W_L1 ... X_W_LN | X_W_L1 X_W_L2 ... X_W_LN |

  int cu_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_index >= num_configurations * num_voxels * num_robot_geometries) {
    return;
  }
  const int configuration_idx = cu_index / (num_voxels * num_robot_geometries);
  uint8_t *result = is_geom_to_vox_pair_col_free_device + cu_index;
  const int voxel_index = cu_index % num_voxels;
  const int robot_geom_id_index =
      (cu_index / num_voxels) % num_robot_geometries;
  const int geom_idx = *(robot_geometry_ids + robot_geom_id_index);

  //   printf("collision_pair_matrix device \n");
  //   printArrayN(collision_pair_matrix_device, 4);

  //   printf("config %d result location %d pair %d, geomA_idx %d, geomB_idx
  //   %d\n",
  //          configuration_idx, cu_index, collision_pair_index, geomA_idx,
  //          geomB_idx);

  // Get index of the transformations. They are concatenated first by links,
  // then by configurations.
  const CollisionGeometry *geom = scene_geometries_device + geom_idx;
  const int X_W_L_idx =
      geom->link_index + tree_device->num_links * configuration_idx;
  const Eigen::Matrix4f X_W_L =
      Eigen::Map<const Eigen::Matrix4f>(transforms_device + 16 * X_W_L_idx);
  const Eigen::Vector3f p_W_Voxel =
      Eigen::Map<const Eigen::Vector3f>(voxels + 3 * voxel_index);
  // do the collision check
  *result = geomToVoxelCollisionFree(geom, &X_W_L, &p_W_Voxel, voxel_radius);
}

void executeCollisionFreeKernel(uint8_t *is_config_col_free_device,
                                uint8_t *is_pair_col_free_device,
                                const MinimalPlant *plant_device,
                                const float *transforms_device,
                                const int num_configurations,
                                const int num_collision_pairs) {
  int threads_per_block_pcc = 128;
  int threads_per_block_pooling = 128;
  // calculate the number of blocks for the pair wise collision checks
  int num_blocks_pcc =
      (num_configurations * num_collision_pairs + threads_per_block_pcc - 1) /
      threads_per_block_pcc;
  //   printf("num pairs %d \n num_configs %d \n num_blocks cckernel %d \n",
  //          num_collision_pairs, num_configurations, num_blocks);
  pairCollisionFreeKernel<<<num_blocks_pcc, threads_per_block_pcc>>>(
      is_pair_col_free_device, plant_device->scene_collision_geometries,
      plant_device->collision_pairs_flat, transforms_device,
      &(plant_device->kin_tree), num_configurations, num_collision_pairs);

  int num_blocks_pooling =
      (num_configurations + threads_per_block_pooling - 1) /
      threads_per_block_pooling;

  poolPairCollisionFreeResultsKernel<<<num_blocks_pooling,
                                       threads_per_block_pooling>>>(
      is_config_col_free_device, is_pair_col_free_device, num_configurations,
      num_collision_pairs);
}

void executeCollisionFreeVoxelsWithoutSelfCollisionKernel(
    uint8_t *is_config_voxel_col_free_device, uint8_t *is_pair_col_free_device,
    uint8_t *is_geom_to_vox_pair_col_free_device, const float *voxels,
    const float voxel_radius, const MinimalPlant *plant_device,
    const GeometryIndex *robot_geometry_ids, const float *transforms_device,
    const int num_configurations, const int num_collision_pairs,
    const int num_robot_geometries, const int num_voxels) {
  assert(num_voxels >= 0);
  // no idea what to set here nsight is not giving me a clear message
  int threads_per_block_g2v = 128;
  int threads_per_block_g2v_pooling = 128;

  if (num_voxels > 0) {
    // Check collision between each robot geometry and each voxel.
    int num_blocks_g2v =
        (num_configurations * num_robot_geometries * num_voxels +
         threads_per_block_g2v - 1) /
        threads_per_block_g2v;
    int highest_cuda_idx = num_blocks_g2v * threads_per_block_g2v;

    geomToVoxelPairCollisionFreeKernel<<<num_blocks_g2v,
                                         threads_per_block_g2v>>>(
        is_geom_to_vox_pair_col_free_device, voxels, voxel_radius,
        plant_device->scene_collision_geometries, robot_geometry_ids,
        transforms_device, &(plant_device->kin_tree), num_configurations,
        num_robot_geometries, num_voxels);

    // Pool the results of the collision checks between robot geometries and
    // voxels, such that if any of the checks fails, then the configuration is
    // marked as in collision.
    int num_blocks_g2v_pooling =
        (num_configurations + threads_per_block_g2v_pooling - 1) /
        threads_per_block_g2v_pooling;

    dim3 grid_size;
    dim3 block_size;
    GetGridAndBlockSize(num_configurations, num_voxels, &grid_size,
                        &block_size);

    poolVoxelGeomPairCollisionFreeResultsKernel<<<grid_size, block_size>>>(
        is_config_voxel_col_free_device, is_geom_to_vox_pair_col_free_device,
        num_configurations, num_robot_geometries, num_voxels);
  }
}

void executeCollisionFreeVoxelsKernel(
    uint8_t *is_config_col_free_device, uint8_t *is_pair_col_free_device,
    uint8_t *is_geom_to_vox_pair_col_free_device, const float *voxels,
    const float voxel_radius, const MinimalPlant *plant_device,
    const GeometryIndex *robot_geometry_ids, const float *transforms_device,
    const int num_configurations, const int num_collision_pairs,
    const int num_robot_geometries, const int num_voxels) {
  // no idea what to set here nsight is not giving me a clear message
  int threads_per_block_pcc = 128;
  int threads_per_block_pooling = 128;
  int threads_per_block_g2v = 128;
  int threads_per_block_g2v_pooling = 128;

  int num_blocks_pcc =
      (num_configurations * num_collision_pairs + threads_per_block_pcc - 1) /
      threads_per_block_pcc;
  pairCollisionFreeKernel<<<num_blocks_pcc, threads_per_block_pcc>>>(
      is_pair_col_free_device, plant_device->scene_collision_geometries,
      plant_device->collision_pairs_flat, transforms_device,
      &(plant_device->kin_tree), num_configurations, num_collision_pairs);

  int num_blocks_pooling =
      (num_configurations + threads_per_block_pooling - 1) /
      threads_per_block_pooling;

  poolPairCollisionFreeResultsKernel<<<num_blocks_pooling,
                                       threads_per_block_pooling>>>(
      is_config_col_free_device, is_pair_col_free_device, num_configurations,
      num_collision_pairs);
  if (num_voxels > 0) {
    int num_blocks_g2v =
        (num_configurations * num_robot_geometries * num_voxels +
         threads_per_block_g2v - 1) /
        threads_per_block_g2v;

    geomToVoxelPairCollisionFreeKernel<<<num_blocks_g2v,
                                         threads_per_block_g2v>>>(
        is_geom_to_vox_pair_col_free_device, voxels, voxel_radius,
        plant_device->scene_collision_geometries, robot_geometry_ids,
        transforms_device, &(plant_device->kin_tree), num_configurations,
        num_robot_geometries, num_voxels);

    int num_blocks_g2v_pooling =
        (num_configurations + threads_per_block_g2v_pooling - 1) /
        threads_per_block_g2v_pooling;

    poolGeomVoxelPairCollisionFreeResultsKernel<<<
        num_blocks_g2v_pooling, threads_per_block_g2v_pooling>>>(
        is_config_col_free_device, is_geom_to_vox_pair_col_free_device,
        num_configurations, num_robot_geometries * num_voxels);
  }
}

std::vector<uint8_t> checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
    const Eigen::MatrixXf *configurations, const Voxels *voxels,
    const float voxel_radius, const MinimalPlant *plant,
    const std::vector<GeometryIndex> &robot_geometry_ids) {
  int num_configs = configurations->cols();
  int num_collision_pairs = plant->num_collision_pairs;
  // makes no sense to call this without any voxels result is not defined
  assert(voxels->cols());

  assert(configurations->rows() ==
             plant->kin_tree.num_configuration_variables &&
         "incorrect number of configuration variables");

  // allocate memory for transforms, joint values and the kinematic tree
  CudaPtr<const float> configurations_ptr(configurations->data(),
                                          configurations->size());

  CudaPtr<float> transforms_ptr(nullptr,
                                16 * num_configs * plant->kin_tree.num_links);
  CudaPtr<uint8_t> is_pair_col_free_ptr(nullptr,
                                        num_collision_pairs * num_configs);
  CudaPtr<uint8_t> is_geom_to_vox_pair_col_free_ptr(
      nullptr, robot_geometry_ids.size() * voxels->size() * num_configs);

  CudaPtr<const MinimalPlant> plant_ptr(plant, 1);
  CudaPtr<const GeometryIndex> robot_geometry_ids_ptr(
      robot_geometry_ids.data(), robot_geometry_ids.size());
  CudaPtr<const float> voxels_ptr(voxels->data(), voxels->size());

  configurations_ptr.copyHostToDevice();
  voxels_ptr.copyHostToDevice();
  robot_geometry_ids_ptr.copyHostToDevice();
  plant_ptr.copyHostToDevice();

  std::vector<uint8_t> is_config_voxel_col_free(num_configs * voxels->cols());
  CudaPtr<uint8_t> is_config_voxel_col_free_ptr(is_config_voxel_col_free.data(),
                                                num_configs * voxels->cols());

  // call forward kinematics kernel
  // auto start = std::chrono::high_resolution_clock::now();
  executeForwardKinematicsKernel(transforms_ptr.device,
                                 configurations_ptr.device, num_configs,
                                 &(plant_ptr.device->kin_tree));
  // check all collision pairs
  executeCollisionFreeVoxelsWithoutSelfCollisionKernel(
      is_config_voxel_col_free_ptr.device, is_pair_col_free_ptr.device,
      is_geom_to_vox_pair_col_free_ptr.device, voxels_ptr.device, voxel_radius,
      plant_ptr.device, robot_geometry_ids_ptr.device, transforms_ptr.device,
      num_configs, num_collision_pairs, robot_geometry_ids.size(),
      voxels->cols());

  cudaDeviceSynchronize();
  // auto stop = std::chrono::high_resolution_clock::now();

  // std::string tmp = fmt::format(
  //     "execution time cuda no copy: {} ms",
  //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
  //         .count());
  // std::cout << tmp << std::endl;
  // retrieve answers and free memory
  is_config_voxel_col_free_ptr.copyDeviceToHost();

  return is_config_voxel_col_free;
}

std::vector<uint8_t> checkCollisionFreeVoxelsCuda(
    const Eigen::MatrixXf *configurations, const Voxels *voxels,
    const float voxel_radius, const MinimalPlant *plant,
    const std::vector<GeometryIndex> &robot_geometry_ids) {
  int num_configs = configurations->cols();
  int num_collision_pairs = plant->num_collision_pairs;

  assert(configurations->rows() ==
             plant->kin_tree.num_configuration_variables &&
         "incorrect number of configuration variables");

  // allocate memory for transforms, joint values and the kinematic tree
  CudaPtr<const float> configurations_ptr(configurations->data(),
                                          configurations->size());

  std::vector<uint8_t> is_config_col_free(num_configs);

  CudaPtr<float> transforms_ptr(nullptr,
                                16 * num_configs * plant->kin_tree.num_links);
  CudaPtr<uint8_t> is_pair_col_free_ptr(nullptr,
                                        num_collision_pairs * num_configs);
  CudaPtr<uint8_t> is_geom_to_vox_pair_col_free_ptr(
      nullptr, robot_geometry_ids.size() * voxels->size() * num_configs);

  CudaPtr<uint8_t> is_config_col_free_ptr(is_config_col_free.data(),
                                          num_configs);
  CudaPtr<const MinimalPlant> plant_ptr(plant, 1);
  CudaPtr<const GeometryIndex> robot_geometry_ids_ptr(
      robot_geometry_ids.data(), robot_geometry_ids.size());
  CudaPtr<const float> voxels_ptr(voxels->data(), voxels->size());

  configurations_ptr.copyHostToDevice();
  if (voxels->cols()) voxels_ptr.copyHostToDevice();
  robot_geometry_ids_ptr.copyHostToDevice();
  plant_ptr.copyHostToDevice();

  // call forward kinematics kernel
  auto start = std::chrono::high_resolution_clock::now();
  executeForwardKinematicsKernel(transforms_ptr.device,
                                 configurations_ptr.device, num_configs,
                                 &(plant_ptr.device->kin_tree));
  // check all collision pairs
  executeCollisionFreeVoxelsKernel(
      is_config_col_free_ptr.device, is_pair_col_free_ptr.device,
      is_geom_to_vox_pair_col_free_ptr.device, voxels_ptr.device, voxel_radius,
      plant_ptr.device, robot_geometry_ids_ptr.device, transforms_ptr.device,
      num_configs, num_collision_pairs, robot_geometry_ids.size(),
      voxels->cols());

  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();

  //   std::string tmp = fmt::format(
  //       "execution time cuda no copy: {} ms",
  //       std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
  //           .count());
  //   std::cout << tmp << std::endl;
  // retrieve answers and free memory
  is_config_col_free_ptr.copyDeviceToHost();

  return is_config_col_free;
}

std::vector<uint8_t> checkCollisionFreeCuda(
    const Eigen::MatrixXf *configurations, const MinimalPlant *plant) {
  int num_configs = configurations->cols();
  int num_collision_pairs = plant->num_collision_pairs;

  assert(configurations->rows() ==
             plant->kin_tree.num_configuration_variables &&
         "incorrect number of configuration variables");

  // allocate memory for transforms, joint values and the kinematic tree
  CudaPtr<const float> configurations_ptr(configurations->data(),
                                          configurations->size());

  std::vector<uint8_t> is_config_col_free(num_configs);

  CudaPtr<float> transforms_ptr(nullptr,
                                16 * num_configs * plant->kin_tree.num_links);
  CudaPtr<uint8_t> is_pair_col_free_ptr(nullptr,
                                        num_collision_pairs * num_configs);
  CudaPtr<uint8_t> is_config_col_free_ptr(is_config_col_free.data(),
                                          num_configs);
  CudaPtr<const MinimalPlant> plant_ptr(plant, 1);

  // copy kinematic tree and joint values
  configurations_ptr.copyHostToDevice();
  plant_ptr.copyHostToDevice();

  // call forward kinematics kernel
  auto start = std::chrono::high_resolution_clock::now();
  executeForwardKinematicsKernel(transforms_ptr.device,
                                 configurations_ptr.device, num_configs,
                                 &(plant_ptr.device->kin_tree));
  // check all collision pairs
  executeCollisionFreeKernel(is_config_col_free_ptr.device,
                             is_pair_col_free_ptr.device, plant_ptr.device,
                             transforms_ptr.device, num_configs,
                             num_collision_pairs);

  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();

  //   std::string tmp = fmt::format(
  //       "execution time cuda no copy: {} ms",
  //       std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
  //           .count());
  //   std::cout << tmp << std::endl;
  // retrieve answers and free memory
  is_config_col_free_ptr.copyDeviceToHost();

  return is_config_col_free;
}

__device__ bool isPairCollisionFree(const CollisionGeometry *geomA,
                                    const Eigen::Matrix4f *X_W_LA,
                                    const CollisionGeometry *geomB,
                                    const Eigen::Matrix4f *X_W_LB) {
  assert(geomA->type == ShapePrimitive::SPHERE ||
         geomA->type == ShapePrimitive::BOX);
  assert(geomB->type == ShapePrimitive::SPHERE ||
         geomB->type == ShapePrimitive::BOX);

  // Calculate the world transforms for both geometries
  Eigen::Matrix4f X_W_GEOMA = (*X_W_LA) * geomA->X_L_B;
  Eigen::Matrix4f X_W_GEOMB = (*X_W_LB) * geomB->X_L_B;

  // Sphere-Box collision
  if (geomA->type == SPHERE && geomB->type == BOX) {
    return sphereBox(geomA->dimensions[0], X_W_GEOMA.block<3, 1>(0, 3),
                     geomB->dimensions, X_W_GEOMB);
  } else if (geomA->type == BOX && geomB->type == SPHERE) {
    return sphereBox(geomB->dimensions[0], X_W_GEOMB.block<3, 1>(0, 3),
                     geomA->dimensions, X_W_GEOMA);
  }
  // Sphere-Sphere collision
  else if (geomA->type == SPHERE && geomB->type == SPHERE) {
    return sphereSphere(geomA->dimensions[0], X_W_GEOMA.block<3, 1>(0, 3),
                        geomB->dimensions[0], X_W_GEOMB.block<3, 1>(0, 3));
  }
  // Box-Box collision (using Separating Axis Theorem)
  else {
    return boxBox(geomA->dimensions, X_W_GEOMA, geomB->dimensions, X_W_GEOMB);
  }
}

__device__ bool geomToVoxelCollisionFree(const CollisionGeometry *geom,
                                         const Eigen::Matrix4f *X_W_L,
                                         const Eigen::Vector3f *p_W_Voxel,
                                         const float voxel_radius) {
  assert(geom->type == ShapePrimitive::SPHERE ||
         geom->type == ShapePrimitive::BOX);

  // Calculate the world transforms for both geometries
  Eigen::Matrix4f X_W_GEOM = (*X_W_L) * geom->X_L_B;

  // Sphere-Sphere collision
  if (geom->type == SPHERE) {
    return sphereSphere(voxel_radius, *p_W_Voxel, geom->dimensions[0],
                        X_W_GEOM.block<3, 1>(0, 3));
  }
  // Sphere-Box collision
  else {
    return sphereBox(voxel_radius, *p_W_Voxel, geom->dimensions, X_W_GEOM);
  }
}
}  // namespace csdecomp