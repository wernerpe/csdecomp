#include "collision_checker.h"

#include <fmt/core.h>

#include <iostream>

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
    float maxlen =
        std::max((geomA.dimensions).maxCoeff(), (geomB.dimensions).maxCoeff());
    if ((posA - posB).norm() > 2 * sqrt(3) * maxlen) {
      return true;
    }
    Eigen::Matrix3f rotA = X_W_GEOMA.block<3, 3>(0, 0);
    Eigen::Matrix3f rotB = X_W_GEOMB.block<3, 3>(0, 0);
    Eigen::Vector3f geomA_ax1 = rotA.col(0);
    Eigen::Vector3f geomA_ax2 = rotA.col(1);
    Eigen::Vector3f geomA_ax3 = rotA.col(2);

    Eigen::Vector3f geomB_ax1 = rotB.col(0);
    Eigen::Vector3f geomB_ax2 = rotB.col(1);
    Eigen::Vector3f geomB_ax3 = rotB.col(2);
    std::vector<Eigen::Vector3f> sep_axes(15);
    sep_axes[0] = geomA_ax1;
    sep_axes[1] = geomA_ax2;
    sep_axes[2] = geomA_ax3;

    sep_axes[3] = geomB_ax1;
    sep_axes[4] = geomB_ax2;
    sep_axes[5] = geomB_ax3;

    sep_axes[6] = geomA_ax1.cross(geomB_ax1);
    sep_axes[7] = geomA_ax1.cross(geomB_ax2);
    sep_axes[8] = geomA_ax1.cross(geomB_ax3);

    sep_axes[9] = geomA_ax2.cross(geomB_ax1);
    sep_axes[10] = geomA_ax2.cross(geomB_ax2);
    sep_axes[11] = geomA_ax2.cross(geomB_ax3);

    sep_axes[12] = geomA_ax3.cross(geomB_ax1);
    sep_axes[13] = geomA_ax3.cross(geomB_ax2);
    sep_axes[14] = geomA_ax3.cross(geomB_ax3);

    const Eigen::Vector3f p_A_B = posA - posB;

    for (size_t i = 0; i < sep_axes.size(); ++i) {
      const double lhs = std::abs(p_A_B.dot(sep_axes[i]));
      const double rhs =
          std::abs((geomA.dimensions[0] / 2) * geomA_ax1.dot(sep_axes[i])) +
          std::abs((geomA.dimensions[1] / 2) * geomA_ax2.dot(sep_axes[i])) +
          std::abs((geomA.dimensions[2] / 2) * geomA_ax3.dot(sep_axes[i])) +
          std::abs((geomB.dimensions[0] / 2) * geomB_ax1.dot(sep_axes[i])) +
          std::abs((geomB.dimensions[1] / 2) * geomB_ax2.dot(sep_axes[i])) +
          std::abs((geomB.dimensions[2] / 2) * geomB_ax3.dot(sep_axes[i]));
      if (lhs > rhs) {
        // There exists a separating plane, so the pair is collision-free.
        return true;
      }
    }
    return false;
  }

  // Sphere-Box collision
  if ((geomA.type == SPHERE && geomB.type == BOX) ||
      (geomA.type == BOX && geomB.type == SPHERE)) {
    const CollisionGeometry &sphere = (geomA.type == SPHERE) ? geomA : geomB;
    const CollisionGeometry &box = (geomA.type == BOX) ? geomA : geomB;
    const Eigen::Vector3f &spherePos = (geomA.type == SPHERE) ? posA : posB;
    const Eigen::Vector3f &boxPos = (geomA.type == BOX) ? posA : posB;
    const Eigen::Matrix3f boxRot = (geomA.type == BOX)
                                       ? X_W_GEOMA.block<3, 3>(0, 0)
                                       : X_W_GEOMB.block<3, 3>(0, 0);

    float sphereRadius = sphere.dimensions[0];
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
    return distanceSquared > (sphereRadius * sphereRadius);
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