#pragma once
#include "collision_geometry.h"
#include "kinematic_tree.h"
#include "minimal_kinematic_tree.h"

namespace csdecomp {

/**
 * @struct MinimalPlant
 * @brief Represents a minimal plant structure containing kinematic and
 * collision information.
 *
 * This structure encapsulates the essential components of a plant, including
 * its kinematic tree, collision geometries, and collision pairs.
 */
struct MinimalPlant {
  /** @brief The kinematic tree representing the plant's structure and motion.
   */
  MinimalKinematicTree kin_tree;

  /**
   * @brief Array of collision geometries for static objects in the scene.
   * @note The maximum number of static collision geometries is defined by
   * MAX_NUM_STATIC_COLLISION_GEOMETRIES.
   */
  CollisionGeometry
      scene_collision_geometries[MAX_NUM_STATIC_COLLISION_GEOMETRIES];

  /**
   * @brief Flattened array of collision pair indices.
   * @note The size of this array is calculated based on the maximum number of
   * possible pairs given MAX_NUM_STATIC_COLLISION_GEOMETRIES.
   */
  GeometryIndex collision_pairs_flat[MAX_NUM_STATIC_COLLISION_GEOMETRIES *
                                     (MAX_NUM_STATIC_COLLISION_GEOMETRIES - 1) /
                                     2];

  /** @brief The number of collision geometries in the scene. */
  int32_t num_scene_geometries;

  /** @brief The number of collision pairs. */
  int32_t num_collision_pairs;
};

// wrapper class for kinematic tree and collision information
class Plant {
 public:
  Plant(const KinematicTree kin_tree, const SceneInspector inspector,
        const std::vector<CollisionGeometry> scene_collision_geometries)
      : kin_tree_(kin_tree),
        inspector_(inspector),
        scene_collision_geometries_(scene_collision_geometries) {}
  const int numLinks() const { return kin_tree_.numLinks(); }
  const int numJoints() const { return kin_tree_.numLinks(); }
  const Eigen::VectorXf getPositionLowerLimits() const {
    return kin_tree_.getPositionLowerLimits();
  }
  const Eigen::VectorXf getPositionUpperLimits() const {
    return kin_tree_.getPositionUpperLimits();
  }
  const int numConfigurationVariables() const {
    return kin_tree_.numConfigurationVariables();
  };

  const MinimalPlant getMinimalPlant() const {
    MinimalKinematicTree mtree = kin_tree_.getMyMinimalKinematicTreeStruct();
    CollisionPairMatrix cpm = inspector_.getCollisionPairMatrix();

    // Calculate the number of geometries to copy
    int num_geometries =
        std::min(static_cast<int>(scene_collision_geometries_.size()),
                 MAX_NUM_STATIC_COLLISION_GEOMETRIES);

    // Create a MinimalPlant instance
    MinimalPlant mplant;

    // Correctly copy the MinimalKinematicTree
    std::memcpy(mplant.kin_tree.joints, mtree.joints,
                sizeof(MinimalJoint) * MAX_JOINTS_PER_TREE);
    std::memcpy(mplant.kin_tree.links, mtree.links,
                sizeof(MinimalLink) * MAX_LINKS_PER_TREE);
    std::memcpy(mplant.kin_tree.joint_traversal_sequence,
                mtree.joint_traversal_sequence,
                sizeof(int) * MAX_JOINTS_PER_TREE);
    std::memcpy(mplant.kin_tree.joint_idx_to_config_idx,
                mtree.joint_idx_to_config_idx,
                sizeof(int) * MAX_JOINTS_PER_TREE);
    mplant.kin_tree.num_configuration_variables =
        mtree.num_configuration_variables;
    mplant.kin_tree.num_joints = mtree.num_joints;
    mplant.kin_tree.num_links = mtree.num_links;
    mplant.num_collision_pairs = cpm.cols();
    assert(mplant.num_collision_pairs <=
           MAX_NUM_STATIC_COLLISION_GEOMETRIES *
               (MAX_NUM_STATIC_COLLISION_GEOMETRIES - 1) / 2);

    std::copy(cpm.data(), cpm.data() + 2 * cpm.cols(),
              mplant.collision_pairs_flat);
    mplant.num_scene_geometries = num_geometries;

    // Copy the collision geometries
    std::copy(scene_collision_geometries_.begin(),
              scene_collision_geometries_.begin() + num_geometries,
              mplant.scene_collision_geometries);

    return mplant;
  };
  const KinematicTree& getKinematicTree() const { return kin_tree_; };
  const SceneInspector& getSceneInspector() const { return inspector_; };
  const std::vector<GeometryIndex>& getRobotGeometryIds() const {
    return inspector_.robot_geometry_ids;
  };
  const std::vector<CollisionGeometry>& getSceneCollisionGeometries() const {
    return scene_collision_geometries_;
  }
  const CollisionPairMatrix getCollisionPairMatrix() const {
    return inspector_.getCollisionPairMatrix();
  }

  const std::vector<std::string> getSceneCollisionGeometryNames() const {
    std::vector<std::string> keys(
        inspector_.scene_collision_geometry_name_to_geometry_index.size());

    for (const auto& pair :
         inspector_.scene_collision_geometry_name_to_geometry_index) {
      keys.at(pair.second) = pair.first;
    }
    return keys;
  };

  const int getSceneCollisionGeometryIndexByName(
      const std::string& name) const {
    auto it =
        inspector_.scene_collision_geometry_name_to_geometry_index.find(name);
    if (it !=
        inspector_.scene_collision_geometry_name_to_geometry_index.end()) {
      return it->second;
    }
    throw std::out_of_range("Geometry name not found: " + name);
  };

 private:
  KinematicTree kin_tree_;
  SceneInspector inspector_;
  std::vector<CollisionGeometry> scene_collision_geometries_;
};
}  // namespace csdecomp