#pragma once
#include <Eigen/Dense>
#include <vector>

#include "collision_geometry.h"
#include "cuda_utilities.h"
#include "minimal_kinematic_tree.h"
#include "minimal_plant.h"

namespace csdecomp {
/**
 * @brief Executes the collision checking kernel on the GPU.
 *
 * This function performs collision checking of the robot against itself and all
 * static geometries in the scene using CUDA.
 *
 * @param is_config_col_free_device Pointer to GPU memory for storing
 * configuration-wise collision results.
 * @param is_pair_col_free_device Pointer to GPU memory for storing pair-wise
 * collision results.
 * @param plant_device Pointer to GPU memory containing the MinimalPlant
 * structure.
 * @param transforms_device Pointer to GPU memory containing transformation
 * matrices.
 * @param num_configurations Number of configurations to check.
 * @param num_collision_pairs Number of collision pairs to check.
 *
 * @note Assumes pre-allocated GPU memory for all pointers.
 * @note is_config_col_free_device will indicate if each configuration is
 * collision-free.
 * @note is_pair_col_free_device will indicate collision state for each pair in
 * each configuration.
 */
void executeCollisionFreeKernel(uint8_t *is_config_col_free_device,
                                uint8_t *is_pair_col_free_device,
                                const MinimalPlant *plant_device,
                                const float *transforms_device,
                                const int num_configurations,
                                const int num_collision_pairs);

/**
 * @brief Executes collision checking kernel including voxelmap collisions on
 * the GPU.
 *
 * Performs collision checking of the robot against itself, all static
 * geometries, and a supplied voxelmap using CUDA.
 *
 * @param is_config_col_free_device GPU memory pointer for configuration-wise
 * collision results.
 * @param is_pair_col_free_device GPU memory pointer for pair-wise collision
 * results.
 * @param is_geom_to_vox_pair_col_free_device GPU memory pointer for robot
 * geometry to voxel collision results.
 * @param voxels Flattened 3xN array of voxel locations in world frame.
 * @param voxel_radius Radius of voxels (treated as spheres).
 * @param plant_device GPU memory pointer to MinimalPlant structure.
 * @param robot_geometry_ids Pointer to array of geometry IDs for voxel
 * collision checking.
 * @param transforms_device GPU memory pointer to transformation matrices.
 * @param num_configurations Number of configurations to check.
 * @param num_collision_pairs Number of collision pairs to check.
 * @param num_robot_geometries Number of robot geometries to check against
 * voxels.
 * @param num_voxels Number of voxels in the voxelmap.
 *
 * @note Assumes pre-allocated GPU memory for all pointers.
 * @note is_config_col_free_device indicates if each configuration is
 * collision-free.
 * @note is_pair_col_free_device indicates collision state for each static pair
 * per configuration.
 * @note is_geom_to_vox_pair_col_free_device indicates collision state for each
 * robot geometry to each voxel, ordered by configuration, robot geometry, then
 * voxel.
 */
void executeCollisionFreeVoxelsKernel(
    uint8_t *is_config_col_free_device, uint8_t *is_pair_col_free_device,
    uint8_t *is_geom_to_vox_pair_col_free_device, const float *voxels,
    const float voxel_radius, const MinimalPlant *plant_device,
    const GeometryIndex *robot_geometry_ids, const float *transforms_device,
    const int num_configurations, const int num_collision_pairs,
    const int num_robot_geometries, const int num_voxels);
/**
 * @brief High-level C++ wrapper for CUDA-based collision checking.
 *
 * Handles memory allocation and performs collision checking using CUDA.
 *
 * @param configurations Pointer to the matrix of configurations to check.
 * @param plant MinimalPlant object with kinematics and geometry information.
 * @return std::vector<uint8_t> Vector indicating collision-free status for each
 * configuration.
 */
std::vector<uint8_t> checkCollisionFreeCuda(
    const Eigen::MatrixXf *configurations, const MinimalPlant *plant);

/**
 * @brief High-level C++ wrapper for CUDA-based collision checking with voxel
 * maps, excluding self-collisions.
 *
 * Handles memory allocation and performs collision checking against voxel maps
 * and the static environment, but excludes self-collision checks.
 *
 * @param configurations Pointer to the matrix of configurations to check.
 * @param voxels Pointer to the Voxels object representing the voxel map.
 * @param voxel_radius Radius of each voxel.
 * @param plant MinimalPlant object with kinematics and geometry information.
 * @param robot_geometry_ids Vector of geometry indices to check against the
 * voxel map.
 * @return std::vector<uint8_t> Vector indicating collision-free status for each
 * configuration.
 */
std::vector<uint8_t> checkCollisionFreeVoxelsCuda(
    const Eigen::MatrixXf *configurations, const Voxels *voxels,
    const float voxel_radius, const MinimalPlant *plant,
    const std::vector<GeometryIndex> &robot_geometries);

std::vector<uint8_t> checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
    const Eigen::MatrixXf *configurations, const Voxels *voxels,
    const float voxel_radius, const MinimalPlant *plant,
    const std::vector<GeometryIndex> &robot_geometry_ids);

#ifdef __CUDACC__

/**
 * @brief Checks if a pair of collision geometries is intersecting.
 *
 * This device function determines whether two collision objects intersect.
 *
 * @param geomA Pointer to the first collision object.
 * @param X_W_LA Pointer to the transform matrix from link frame A to world
 * frame.
 * @param geomB Pointer to the second collision object.
 * @param X_W_LB Pointer to the transform matrix from link frame B to world
 * frame.
 * @return bool True if the pair is collision-free, false otherwise.
 */
__device__ bool isPairCollisionFree(const CollisionGeometry *geomA,
                                    const Eigen::Matrix4f *X_W_LA,
                                    const CollisionGeometry *geomB,
                                    const Eigen::Matrix4f *X_W_LB);
/**
 * @brief Checks if a geometry is in collision with a voxel.
 *
 * This device function determines whether a collision geometry intersects with
 * a voxel.
 *
 * @param geom Pointer to the collision geometry.
 * @param X_W_L Pointer to the transform matrix from link frame to world frame.
 * @param p_W_Voxel Pointer to the voxel's position in world frame.
 * @param voxel_radius Radius of the voxel.
 * @return bool True if the geometry and voxel are collision-free, false
 * otherwise.
 */
__device__ bool geomToVoxelCollisionFree(const CollisionGeometry *geom,
                                         const Eigen::Matrix4f *X_W_L,
                                         const Eigen::Vector3f *p_W_Voxel,
                                         const float voxel_radius);

#endif
}  // namespace csdecomp