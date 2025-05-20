#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace csdecomp {
#define MAX_NUM_STATIC_COLLISION_GEOMETRIES 600

typedef int32_t GeometryIndex;
typedef Eigen::Matrix<GeometryIndex, 2, Eigen::Dynamic, Eigen::ColMajor>
    CollisionPairMatrix;

/**
 * @brief Enumeration of primitive shape types for collision objects.
 */
enum ShapePrimitive { BOX, SPHERE, CYLINDER, CAPSULE };

/**
 * @brief Enumeration of collision groups for categorizing objects.
 */
enum CollisionGroup { ROBOT, WORLD, VOXELS };

/**
 * @brief Represents a collision geometry with its properties.
 */
struct CollisionGeometry {
  ShapePrimitive type;         ///< The primitive shape type of the object
  Eigen::Vector3f dimensions;  ///< Dimensions of the object (interpretation
                               ///< depends on type)
  int link_index;              ///< Index of the link this object is attached to
  CollisionGroup group;        ///< The collision group this object belongs to
  Eigen::Matrix4f
      X_L_B;  ///< Transform from the collision geometry to its link frame
};

// voxels are always of type ShapePrimitive::SPHERE and are assumed to have the
// same radius
typedef Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::ColMajor> Voxels;

/**
 * @brief Holds information connecting collision objects to the kinematic tree.
 */
struct SceneInspector {
  /**
   * @brief Maps link indices to indices of collision geometries in the scene.
   *
   * Each vector in this vector represents the collision geometry indices
   * associated with a particular link.
   */
  std::vector<std::vector<int>> link_index_to_scene_collision_geometry_indices;

  std::unordered_map<std::string, int>
      scene_collision_geometry_name_to_geometry_index;
  /**
   * @brief Pairs of indices representing potential collisions between robot and
   * world objects.
   *
   * Each pair contains indices to world and robot collision objects.
   */
  std::vector<std::pair<int, int>> robot_to_world_collision_pairs;

  std::vector<GeometryIndex> robot_geometry_ids;
  /**
   * @brief Generates a collision pair matrix containing all collision pairs
   * between the robot and the static scene (no voxels).
   *
   * @return CollisionPairMatrix A 2xN matrix where N is the number of collision
   * pairs.
   */
  CollisionPairMatrix getCollisionPairMatrix() const {
    CollisionPairMatrix col_pair_mat =
        CollisionPairMatrix::Zero(2, robot_to_world_collision_pairs.size());
    int col = 0;
    for (auto p : robot_to_world_collision_pairs) {
      col_pair_mat(0, col) = (int32_t)p.first;
      col_pair_mat(1, col) = (int32_t)p.second;
      ++col;
    }

    return col_pair_mat;
  }
  SceneInspector() = default;
  SceneInspector(const SceneInspector& other)
      : link_index_to_scene_collision_geometry_indices(
            other.link_index_to_scene_collision_geometry_indices),
        scene_collision_geometry_name_to_geometry_index(
            other.scene_collision_geometry_name_to_geometry_index),
        robot_to_world_collision_pairs(other.robot_to_world_collision_pairs),
        robot_geometry_ids(other.robot_geometry_ids) {}

  SceneInspector& operator=(const SceneInspector& other) {
    if (this != &other) {
      link_index_to_scene_collision_geometry_indices =
          other.link_index_to_scene_collision_geometry_indices;
      scene_collision_geometry_name_to_geometry_index =
          other.scene_collision_geometry_name_to_geometry_index;
      robot_to_world_collision_pairs = other.robot_to_world_collision_pairs;
      robot_geometry_ids = other.robot_geometry_ids;
    }
    return *this;
  }
};
}  // namespace csdecomp