#include "collision_checker.h"

#include <fmt/core.h>

#include <iostream>

#include "geometry_utilities.h"

namespace csdecomp {

bool checkCollisionFree(const Eigen::VectorXf &configuration,
                        const MinimalPlant &plant) {
  Eigen::MatrixXf transforms =
      Eigen::MatrixXf::Zero(plant.kin_tree.num_links * 4, 4);
  computeLinkFrameToWorldTransformsMinimal(configuration, plant.kin_tree,
                                           &transforms);
  const CollisionPairMatrix cpm = Eigen::Map<const CollisionPairMatrix>(
      plant.collision_pairs_flat, 2, (int)plant.num_collision_pairs);

  for (int pair = 0; pair < plant.num_collision_pairs; ++pair) {
    int idA = cpm(0, pair);
    int idB = cpm(1, pair);
    const CollisionGeometry *geomA = &(plant.scene_collision_geometries[idA]);
    const CollisionGeometry *geomB = &(plant.scene_collision_geometries[idB]);
    // Eigen::Matrix4f X_W_LA = &(transforms.block<4, 4>(4 * geomA->link_index,
    // 0)); Eigen::Matrix4f* X_W_LB = &(transforms.block<4, 4>(4 *
    // geomB->link_index, 0));
    bool collision_free = pairCollisionFree(
        *geomA, transforms.block<4, 4>(4 * geomA->link_index, 0), *geomB,
        transforms.block<4, 4>(4 * geomB->link_index, 0));
    if (!collision_free) return false;
  }
  return true;
}

bool checkEdgeCollisionFree(const Eigen::VectorXf &configuration_1,
                            const Eigen::VectorXf &configuration_2,
                            const MinimalPlant &plant, const float &step_size) {
  int t_steps = ceil((configuration_2 - configuration_1).norm() / step_size);
  for (int step_idx = 0; step_idx < t_steps; ++step_idx) {
    float t = step_idx / (1.0 * t_steps);
    if (!checkCollisionFree(t * configuration_2 + (1 - t) * configuration_1,
                            plant)) {
      return false;
    }
  }
  return true;
}

bool checkCollisionFreeVoxels(
    const Eigen::VectorXf &configuration, const Voxels &voxels,
    const float voxel_radius, const MinimalPlant &plant,
    const std::vector<GeometryIndex> &robot_geometries) {
  Eigen::MatrixXf transforms =
      Eigen::MatrixXf::Zero(plant.kin_tree.num_links * 4, 4);
  computeLinkFrameToWorldTransformsMinimal(configuration, plant.kin_tree,
                                           &transforms);
  const CollisionPairMatrix cpm = Eigen::Map<const CollisionPairMatrix>(
      plant.collision_pairs_flat, 2, (int)plant.num_collision_pairs);

  for (int pair = 0; pair < plant.num_collision_pairs; ++pair) {
    int idA = cpm(0, pair);
    int idB = cpm(1, pair);
    auto geomA = plant.scene_collision_geometries[idA];
    auto geomB = plant.scene_collision_geometries[idB];
    auto X_W_LA = transforms.block<4, 4>(4 * geomA.link_index, 0);
    auto X_W_LB = transforms.block<4, 4>(4 * geomB.link_index, 0);
    bool collision_free = pairCollisionFree(geomA, X_W_LA, geomB, X_W_LB);
    if (!collision_free) {
      return false;
    }
  }

  for (auto rgidx : robot_geometries) {
    for (int vox_idx = 0; vox_idx < voxels.cols(); ++vox_idx) {
      auto rob_geom = plant.scene_collision_geometries[rgidx];
      auto X_W_LA = transforms.block<4, 4>(4 * rob_geom.link_index, 0);
      bool col_free = geomToVoxelCollisionFree(
          rob_geom, X_W_LA, voxels.col(vox_idx), voxel_radius);
      if (!col_free) return false;
    }
  }
  return true;
}

// capsule, capsule -> compute closest distance between the linesegments, if
// less than radius, there is a collision. capsule, sphere -> compute closest
// distance to line segment compare to radius capsule, box
bool pairCollisionFree(const CollisionGeometry &geomA,
                       const Eigen::Matrix4f &X_W_LA,
                       const CollisionGeometry &geomB,
                       const Eigen::Matrix4f &X_W_LB) {
  // Calculate the world transforms for both geometries
  Eigen::Matrix4f X_W_GEOMA = X_W_LA * geomA.X_L_B;
  Eigen::Matrix4f X_W_GEOMB = X_W_LB * geomB.X_L_B;

  // Extract positions in world frame
  Eigen::Vector3f posA = X_W_GEOMA.block<3, 1>(0, 3);
  Eigen::Vector3f posB = X_W_GEOMB.block<3, 1>(0, 3);

  // Check for unsupported shape primitives
  if (geomA.type == CYLINDER || geomA.type == CAPSULE ||
      geomB.type == CYLINDER || geomB.type == CAPSULE) {
    throw std::runtime_error("Cylinders and capsules are not supported yet.");
  }

  // Sphere-Sphere collision
  if (geomA.type == SPHERE && geomB.type == SPHERE) {
    float radiusA = geomA.dimensions[0];
    float radiusB = geomB.dimensions[0];
    float distance = (posA - posB).norm();
    return distance > (radiusA + radiusB);
  }

  if (geomA.type == BOX && geomB.type == BOX) {
    return cppBoxBox(geomA.dimensions, X_W_GEOMA, geomB.dimensions, X_W_GEOMB);
  }

  // Sphere-Box collision
  if (geomA.type == SPHERE && geomB.type == BOX) {
    return cppSphereBox(geomA.dimensions[0], posA, geomB.dimensions, X_W_GEOMB);
  }
  if (geomA.type == BOX && geomB.type == SPHERE) {
    return cppSphereBox(geomB.dimensions[0], posB, geomA.dimensions, X_W_GEOMA);
  }

  if (geomA.type == SPHERE && geomB.type == CAPSULE) {
    return cppCapsuleSphere(X_W_GEOMB, geomB.dimensions[0], geomB.dimensions[1],
                            posA, geomA.dimensions[0]);
  }

  if (geomA.type == CAPSULE && geomB.type == SPHERE) {
    return cppCapsuleSphere(X_W_GEOMA, geomA.dimensions[0], geomA.dimensions[1],
                            posB, geomB.dimensions[0]);
  }

  // Should never reach here if all cases are handled
  throw std::runtime_error("Unexpected geometry types encountered.");
}

bool geomToVoxelCollisionFree(const CollisionGeometry &geom,
                              const Eigen::Matrix4f &X_W_LA,
                              const Eigen::Vector3f &p_W_Voxel,
                              const float &voxel_radius) {
  // Calculate the world transform for both geometries
  Eigen::Matrix4f X_W_GEOM = X_W_LA * geom.X_L_B;

  // Extract position in world frame
  Eigen::Vector3f pos = X_W_GEOM.block<3, 1>(0, 3);

  // Check for unsupported shape primitives
  if (geom.type == CYLINDER || geom.type == CAPSULE) {
    throw std::runtime_error("Cylinders and capsules are not supported yet.");
  }

  // Sphere-Sphere collision
  if (geom.type == SPHERE) {
    float radiusA = geom.dimensions[0];
    float distance = (pos - p_W_Voxel).norm();
    return distance > (radiusA + voxel_radius);
  }

  // Sphere-Box collision
  if (geom.type == BOX) {
    const CollisionGeometry &box = geom;
    const Eigen::Vector3f &spherePos = p_W_Voxel;
    const Eigen::Vector3f &boxPos = pos;
    const Eigen::Matrix3f boxRot = X_W_GEOM.block<3, 3>(0, 0);

    Eigen::Vector3f boxHalfExtents = box.dimensions * 0.5f;

    // Transform sphere center to box local space
    Eigen::Vector3f sphereLocalPos = boxRot.transpose() * (spherePos - boxPos);

    // Find the closest point on the box to the sphere center
    Eigen::Vector3f closestPoint;
    for (int i = 0; i < 3; ++i) {
      closestPoint[i] = std::max(
          -boxHalfExtents[i], std::min(sphereLocalPos[i], boxHalfExtents[i]));
    }

    // Check if the closest point is within the sphere's radius
    float distanceSquared = (closestPoint - sphereLocalPos).squaredNorm();
    return distanceSquared > (voxel_radius * voxel_radius);
  }

  // Should never reach here if all cases are handled
  throw std::runtime_error("Unexpected geometry types encountered.");
}
}  // namespace csdecomp