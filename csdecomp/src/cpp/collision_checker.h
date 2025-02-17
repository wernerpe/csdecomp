#pragma once
#include "collision_geometry.h"
#include "minimal_plant.h"

namespace csdecomp {
/**
 * @brief Check if a configuration is collision-free.
 *
 * @param configuration The joint configuration of the robot to check.
 * @param plant MinimalPlantObject containing all information about the system
 * required for collision checking (kinematics, geometries, and collision
 * pairs).
 * @return bool False if a collision is detected, True otherwise.
 */
bool checkCollisionFree(const Eigen::VectorXf &configuration,
                        const MinimalPlant &plant);

bool checkEdgeCollisionFree(const Eigen::VectorXf &configuration_1,
                            const Eigen::VectorXf &configuration_2,
                            const MinimalPlant &plant,
                            const float &step_size = 0.01);

/**
 * @brief Check if robot is colliding with the world, itself or voxels at a
 * configuration.
 *
 * @param configuration The joint configuration of the robot to check.
 * @param voxels Voxel matrix 3xN containing all the positions of the voxels in
 * world frame. N is the number of voxels.
 * @param voxel_radius radius of voxels -voxels are treated as spheres in the
 * collision checker.
 * @param plant MinimalPlantObject containing all information about the system
 * required for collision checking (kinematics, geometries, and collision
 * pairs).
 * @param robot_geometries vector of geometryindices to check for collisions
 * agains the voxels.
 * @return bool False if a collision is detected, True otherwise.
 */
bool checkCollisionFreeVoxels(
    const Eigen::VectorXf &configuration, const Voxels &voxels,
    const float voxel_radius, const MinimalPlant &plant,
    const std::vector<GeometryIndex> &robot_geometries);

/**
 * @brief Check if a pair of geometries is collision free.
 *
 * @param geomA The first CollisionGeometry.
 * @param X_W_LA The transformation matrix from local frame of geomA to world
 * frame.
 * @param geomB The second CollisionGeometry.
 * @param X_W_LB The transformation matrix from local frame of geomB to world
 * frame.
 * @return bool False if the objects are colliding, True otherwise.
 */
bool pairCollisionFree(const CollisionGeometry &geomA,
                       const Eigen::Matrix4f &X_W_LA,
                       const CollisionGeometry &geomB,
                       const Eigen::Matrix4f &X_W_LB);

/**
 * @brief Check if a pair of geometries is collision free.
 *
 * @param geom The first CollisionGeometry.
 * @param X_W_LA The transformation matrix from local frame of geomA to world
 * frame.
 * @param p_W_Voxel position of voxel in world frame.
 * @param voxel_radius The transformation matrix from local frame of geomB to
 * world frame.
 * @return bool False if the objects are colliding, True otherwise.
 */

bool geomToVoxelCollisionFree(const CollisionGeometry &geom,
                              const Eigen::Matrix4f &X_W_LA,
                              const Eigen::Vector3f &p_W_Voxel,
                              const float &voxel_radius);
}  // namespace csdecomp