#pragma once
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

#include "minimal_kinematic_tree.h"

namespace csdecomp {
/**
 * @brief Represents a joint in the kinematic tree.
 */
struct Joint {
  std::string name;  ///< Name of the joint
  JointType type;    ///< Type of the joint (e.g., revolute, prismatic)
  Eigen::Matrix4f
      X_PL_J;  ///< Transform from the joint frame J to the parent link frame PL
  int parent_link;       ///< Index of the parent link
  int child_link;        ///< Index of the child link
  Eigen::Vector3f axis;  ///< Axis of motion for revolute and prismatic joints
  float position_lower_limit;
  float position_upper_limit;
};

/**
 * @brief Represents a link in the kinematic tree.
 */
struct Link {
  std::string name;  ///< Name of the link
  int parent_joint;  ///< Index of the parent joint (-1 for the root link)
  std::vector<int> child_joints;  ///< Indices of child joints
  // Eigen::Matrix4f X_L_PJ_collision; ///< Transform from this link frame L to
  // the parent joint PJ (commented out)
};

/**
 * @brief Represents a kinematic tree structure for robotic systems.
 */
class KinematicTree {
 public:
  /**
   * @brief Constructs a KinematicTree object.
   */
  KinematicTree();

  // Copy constructor
  KinematicTree(const KinematicTree &other)
      : is_finalized(other.is_finalized),
        num_leaf_nodes(other.num_leaf_nodes),
        num_configuration_variables(other.num_configuration_variables),
        position_lower_limits(other.position_lower_limits),
        position_upper_limits(other.position_upper_limits),
        leaf_link_indices(other.leaf_link_indices),
        joint_idx_to_config_idx(other.joint_idx_to_config_idx),
        joint_traversal_sequence(other.joint_traversal_sequence),
        link_name_to_index_(other.link_name_to_index_),
        joint_name_to_index_(other.joint_name_to_index_) {
    // Deep copy links
    links_.reserve(other.links_.size());
    for (const auto &link : other.links_) {
      links_.push_back(Link{link.name, link.parent_joint, link.child_joints});
    }

    // Deep copy joints
    joints_.reserve(other.joints_.size());
    for (const auto &joint : other.joints_) {
      joints_.push_back(Joint{joint.name, joint.type, joint.X_PL_J,
                              joint.parent_link, joint.child_link, joint.axis,
                              joint.position_lower_limit,
                              joint.position_upper_limit});
    }
  }

  /**
   * @brief Adds a link to the kinematic tree.
   * @param link The link to be added.
   */
  void addLink(const Link &link);

  /**
   * @brief Adds a joint to the kinematic tree.
   * @param joint The joint to be added.
   */
  void addJoint(const Joint &joint);

  /**
   * @brief Finalizes the kinematic tree structure.
   */
  void finalize();

  /**
   * @brief Gets the index of a link by its name.
   * @param name The name of the link.
   * @return The index of the link.
   */
  int getLinkIndexByName(const std::string &name) const;

  /**
   * @brief Gets the name of a link by its index.
   * @param index The index of the link.
   * @return The index of the link.
   */
  std::string getLinkNameByIndex(int index) const;

  /**
   * @brief Gets a link by its index.
   * @param index The index of the link.
   * @return A reference to the link.
   */
  const Link &getLink(const int index) const;

  /**
   * @brief Gets the index of a joint by its name.
   * @param name The name of the joint.
   * @return The index of the joint.
   */
  int getJointIndex(const std::string &name) const;

  /**
   * @brief Gets a joint by its index.
   * @param index The index of the joint.
   * @return A reference to the joint.
   */
  const Joint &getJoint(const int index) const;

  /**
   * @brief Checks if the kinematic tree is finalized.
   * @return True if finalized, false otherwise.
   */
  bool isFinalized() const;

  /**
   * @brief Gets the indices of leaf links.
   * @return A vector of leaf link indices.
   */
  const std::vector<int> &getLeafLinkIndices() const;

  /**
   * @brief Gets all links in the kinematic tree.
   * @return A vector of all links.
   */
  const std::vector<Link> &getLinks() const;

  /**
   * @brief Gets all joints in the kinematic tree.
   * @return A vector of all joints.
   */
  const std::vector<Joint> &getJoints() const;

  const Eigen::VectorXf getPositionLowerLimits() const;

  const Eigen::VectorXf getPositionUpperLimits() const;

  /**
   * @brief Computes the transforms from link frames to world frame.
   * @param joint_values The values of the joint variables.
   * @return A vector of transformation matrices.
   */
  std::vector<Eigen::Matrix4f> computeLinkFrameToWorldTransforms(
      const Eigen::VectorXf &joint_values) const;

  /**
   * @brief Gets the minimal kinematic tree structure for GPU evaluation.
   * @return A MinimalKinematicTree object.
   */
  const MinimalKinematicTree getMyMinimalKinematicTreeStruct() const;

  /**
   * @brief Gets the number of links in the kinematic tree.
   * @return The number of links.
   */
  const int numLinks() const;

  /**
   * @brief Gets the number of joints in the kinematic tree.
   * @return The number of joints.
   */
  const int numJoints() const;

  /**
   * @brief Gets the number of configuration variables in the kinematic tree.
   * @return The number of configuration variables.
   */
  const int numConfigurationVariables() const;

 private:
  std::vector<Link> links_;
  std::vector<Joint> joints_;
  std::unordered_map<std::string, int> link_name_to_index_;
  std::unordered_map<std::string, int> joint_name_to_index_;
  bool is_finalized;
  int num_leaf_nodes;
  int num_configuration_variables;
  std::vector<float> position_lower_limits;
  std::vector<float> position_upper_limits;
  std::vector<int> leaf_link_indices;
  std::vector<int> joint_idx_to_config_idx;
  std::vector<int> joint_traversal_sequence;
};
}  // namespace csdecomp