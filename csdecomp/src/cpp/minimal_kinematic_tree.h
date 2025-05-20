#pragma once
#include <Eigen/Dense>

namespace csdecomp {

#define MAX_CHILD_JOINT_COUNT 15
#define MAX_LINKS_PER_TREE 1000
#define MAX_JOINTS_PER_TREE 1000

/**
 * @brief Enumeration of joint types in the kinematic tree.
 */
enum JointType { FIXED, REVOLUTE, PRISMATIC };

/**
 * @brief Represents a minimal joint structure in the kinematic tree.
 */
struct MinimalJoint {
  JointType type;  ///< Type of the joint
  Eigen::Matrix4f
      X_PL_J;  ///< Transform from the joint frame J to the parent link frame PL
  int index;   ///< Index of the joint
  int parent_link;       ///< Index of the parent link
  int child_link;        ///< Index of the child link
  Eigen::Vector3f axis;  ///< Axis of motion for revolute and prismatic joints
};

/**
 * @brief Represents a minimal link structure in the kinematic tree.
 */
struct MinimalLink {
  int index;               ///< Index of the link
  int parent_joint_index;  ///< Index of the parent joint
  int child_joint_index_array[MAX_CHILD_JOINT_COUNT];  ///< Array of child joint
                                                       ///< indices
  int num_child_joints;  ///< Number of child joints
};

/**
 * @brief Represents a minimal kinematic tree structure.
 */
struct MinimalKinematicTree {
  MinimalJoint joints[MAX_JOINTS_PER_TREE];  ///< Array of joints in the tree
  MinimalLink links[MAX_LINKS_PER_TREE];     ///< Array of links in the tree
  int joint_traversal_sequence[MAX_JOINTS_PER_TREE];  ///< Sequence for
                                                      ///< traversing joints
  int joint_idx_to_config_idx[MAX_JOINTS_PER_TREE];   ///< Mapping from joint
                                                     ///< index to configuration
                                                     ///< index
  int num_configuration_variables;  ///< Number of configuration variables
  int num_joints;                   ///< Total number of joints in the tree
  int num_links;                    ///< Total number of links in the tree
};

/**
 * @brief Computes the transforms from link frames to world frame for a minimal
 * kinematic tree.
 *
 * @param configuration The values of the joint variables.
 * @param minimal_tree The minimal kinematic tree structure.
 * @param transforms Pointer to an Eigen::MatrixXf to store the resulting
 * transforms. Each column represents a 4x4 transformation matrix in
 * column-major order.
 */
void computeLinkFrameToWorldTransformsMinimal(
    const Eigen::VectorXf &configuration,
    const MinimalKinematicTree &minimal_tree, Eigen::MatrixXf *transforms);

}  // namespace csdecomp