#include "kinematic_tree.h"

#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <stack>
#include <stdexcept>

namespace csdecomp {

KinematicTree::KinematicTree() { is_finalized = false; }

void KinematicTree::addLink(const Link &link) {
  for (auto l : links_) {
    if (l.name == link.name) {
      std::string error = fmt::format(
          "Duplicate link name detected in URDF. Name : {}", l.name);
      throw std::runtime_error(error);
    }
  }
  if (is_finalized)
    throw std::runtime_error("Cannot add link to finalized tree.");

  int index = links_.size();
  links_.push_back(link);
  link_name_to_index_[link.name] = index;
}

void KinematicTree::addJoint(const Joint &joint) {
  for (auto j : joints_) {
    if (j.name == joint.name) {
      throw std::runtime_error("Duplicate joint name detected in URDF.");
    }
  }
  if (is_finalized)
    throw std::runtime_error("Cannot add joint to finalized tree.");

  int parent_index = joint.parent_link;
  int child_index = joint.child_link;
  if (parent_index == -1 || child_index == -1) {
    throw std::runtime_error("Parent or child link not found for joint: " +
                             joint.name);
  }

  int joint_index = joints_.size();
  joints_.push_back(joint);
  joint_name_to_index_[joint.name] = joint_index;

  links_[parent_index].child_joints.push_back(joint_index);
  links_[child_index].parent_joint = joint_index;
}

int KinematicTree::getLinkIndexByName(const std::string &name) const {
  auto it = link_name_to_index_.find(name);
  int result = (it != link_name_to_index_.end()) ? it->second : -1;
  if (result == -1) {
    std::string all_keys;
    for (const auto &pair : link_name_to_index_) {
      all_keys += pair.first + ", \n";
    }
    throw std::runtime_error(
        fmt::format("[KinematicTree] Invalid link name {}, valid names are: {}",
                    name, all_keys));
  }
  return result;
}

std::string KinematicTree::getLinkNameByIndex(int index) const {
  for (const auto &pair : link_name_to_index_) {
    if (pair.second == index) {
      return pair.first;
    }
  }
  throw std::runtime_error("link index not found");
}

const Link &KinematicTree::getLink(const int index) const {
  return links_.at(index);
}

void KinematicTree::finalize() {
  is_finalized = true;
  num_leaf_nodes = 0;
  num_configuration_variables = 0;
  leaf_link_indices.clear();
  joint_idx_to_config_idx.clear();
  joint_idx_to_config_idx.reserve(joints_.size());
  for (int i = 0; i < int(joints_.size()); ++i) {
    joint_idx_to_config_idx.emplace_back(0);
  }

  for (int i = 0; i < int(links_.size()); ++i) {
    if (links_[i].child_joints.empty()) {
      leaf_link_indices.push_back(i);
      num_leaf_nodes++;
    }
  }

  int joint_idx = 0;

  for (auto j : joints_) {
    if (j.type != JointType::FIXED) {
      joint_idx_to_config_idx.at(joint_idx) = num_configuration_variables;
      ++num_configuration_variables;
    } else {
      joint_idx_to_config_idx.at(joint_idx) = -1;
    }
    ++joint_idx;
  }

  // precompute DFS sequence of joints to traverse kinematic tree in
  std::stack<int> child_joints_to_process;

  // start at root to ensure correctness
  Link root = links_.at(0);
  for (auto cj : root.child_joints) {
    child_joints_to_process.push(cj);
  }
  while (!child_joints_to_process.empty()) {
    int joint_idx_to_process = child_joints_to_process.top();
    child_joints_to_process.pop();

    joint_traversal_sequence.push_back(joint_idx_to_process);

    int child_link_index = joints_[joint_idx_to_process].child_link;
    // push child link child joints onto stack
    for (int cj : links_[child_link_index].child_joints) {
      child_joints_to_process.push(cj);
    }
  }

  // extract joint limits
  for (auto j : joints_) {
    if (j.type != JointType::FIXED) {
      position_lower_limits.push_back(j.position_lower_limit);
      position_upper_limits.push_back(j.position_upper_limit);
    }
  }
  if (position_lower_limits.size() != num_configuration_variables) {
    throw std::runtime_error(
        "[CSDECOMP:KinematicTree] The number of joint limits does not match "
        "the "
        "number of configuraion variables. Double check your URDF.");
  }
}

int KinematicTree::getJointIndex(const std::string &name) const {
  auto it = joint_name_to_index_.find(name);
  return (it != joint_name_to_index_.end()) ? it->second : -1;
}

const Joint &KinematicTree::getJoint(const int index) const {
  return joints_.at(index);
}

const std::vector<Link> &KinematicTree::getLinks() const { return links_; }

const std::vector<Joint> &KinematicTree::getJoints() const { return joints_; }

bool KinematicTree::isFinalized() const { return is_finalized; }

const std::vector<int> &KinematicTree::getLeafLinkIndices() const {
  return leaf_link_indices;
}

const Eigen::VectorXf KinematicTree::getPositionLowerLimits() const {
  Eigen::VectorXf lims(num_configuration_variables);
  int config_idx = 0;
  for (float l : position_lower_limits) {
    lims(config_idx) = l;
    ++config_idx;
  }
  return lims;
}

const Eigen::VectorXf KinematicTree::getPositionUpperLimits() const {
  Eigen::VectorXf lims(num_configuration_variables);
  int config_idx = 0;
  for (float l : position_upper_limits) {
    lims(config_idx) = l;
    ++config_idx;
  }
  return lims;
}

const int KinematicTree::numLinks() const { return links_.size(); }

const int KinematicTree::numJoints() const { return joints_.size(); }

const int KinematicTree::numConfigurationVariables() const {
  return num_configuration_variables;
}

std::vector<Eigen::Matrix4f> KinematicTree::computeLinkFrameToWorldTransforms(
    const Eigen::VectorXf &joint_values) const {
  if (!is_finalized) {
    throw std::runtime_error("KinematicTree is not finalized");
  }

  if (joint_values.size() != num_configuration_variables) {
    throw std::invalid_argument(fmt::format(
        "Incorrect number of configuration varialbes added, there "
        "are {} active joints and {} configuration values were provided",
        num_configuration_variables, joint_values.size()));
  }

  // Transforms from the link frames to the world frame (post joint angle
  // transform of the joint frame)
  std::vector<Eigen::Matrix4f> link_to_world_transforms(
      links_.size(), Eigen::Matrix4f::Identity());
  //   // these are the transforms from the the joint frame to the world frame.
  //   These
  //   // transforms are always before transofrmation by the joint angle.
  //   std::vector<Eigen::MatrixXf> joint_to_world_transforms(
  //       joints_.size(), Eigen::Matrix4f::Identity());

  for (int joint_idx_to_process : joint_traversal_sequence) {
    // get the parent link transform
    int parent_link_index = joints_[joint_idx_to_process].parent_link;
    Eigen::Matrix4f X_W_PL = link_to_world_transforms[parent_link_index];
    // get the child link index we are computing the transform for
    int child_link_index = joints_[joint_idx_to_process].child_link;

    // extract X_PL_J
    Eigen::Matrix4f X_PL_J = joints_[joint_idx_to_process].X_PL_J;
    float config_value = 0;
    // if the joint is not fixed we extract the config value
    if (joints_[joint_idx_to_process].type != JointType::FIXED) {
      config_value =
          joint_values[joint_idx_to_config_idx[joint_idx_to_process]];
    }

    // Compute the active transform from joint
    Eigen::Matrix4f X_J_L = Eigen::Matrix4f::Identity();
    // compute X_PL_L -> apply joint transform
    if (joints_[joint_idx_to_process].type == JointType::REVOLUTE) {
      Eigen::Matrix3f rotation =
          Eigen::AngleAxisf(config_value, joints_[joint_idx_to_process].axis)
              .toRotationMatrix();
      X_J_L.block<3, 3>(0, 0) = rotation;
    } else if (joints_[joint_idx_to_process].type == JointType::PRISMATIC) {
      X_J_L.block<3, 1>(0, 3) +=
          joints_[joint_idx_to_process].axis * config_value;
    }

    // combine X_W_PL * X_PL_L to get the link transform
    Eigen::Matrix4f X_W_L = X_W_PL * X_PL_J * X_J_L;
    link_to_world_transforms.at(child_link_index) = X_W_L;
  }
  return link_to_world_transforms;
}

const MinimalKinematicTree KinematicTree::getMyMinimalKinematicTreeStruct()
    const {
  MinimalKinematicTree minimal_tree;
  minimal_tree.num_configuration_variables = num_configuration_variables;
  minimal_tree.num_joints = int(joints_.size());
  minimal_tree.num_links = int(links_.size());

  int curr_step = 0;
  for (int i : joint_traversal_sequence) {
    minimal_tree.joint_traversal_sequence[curr_step] = i;
    ++curr_step;
  }

  int curr_link = 0;
  for (Link l : links_) {
    MinimalLink min_link;
    min_link.index = curr_link;
    min_link.parent_joint_index = l.parent_joint;
    min_link.num_child_joints = int(l.child_joints.size());

    if (l.child_joints.size() > MAX_CHILD_JOINT_COUNT) {
      throw std::runtime_error(fmt::format(
          "Error attempting to construct a minimal kinematic tree that has "
          "more than {} child joints for a single link. Consider either "
          "modifying your urdf or "
          "recompiling and modifying 'MAX_CHILD_JOINT_COUNT' in "
          "minimal_kinematic_tree.h. The violating link was {}.",
          MAX_CHILD_JOINT_COUNT, l.name));
    }
    int cj_idx = 0;
    for (int cj : l.child_joints) {
      min_link.child_joint_index_array[cj_idx] = cj;
      ++cj_idx;
    }

    minimal_tree.links[curr_link] = min_link;
    ++curr_link;
  }

  int curr_joint = 0;
  for (Joint j : joints_) {
    MinimalJoint min_joint;
    min_joint.index = curr_joint;
    min_joint.type = j.type;
    min_joint.X_PL_J = j.X_PL_J;
    min_joint.parent_link = j.parent_link;
    min_joint.child_link = j.child_link;
    min_joint.axis = j.axis;
    minimal_tree.joint_idx_to_config_idx[curr_joint] =
        joint_idx_to_config_idx[curr_joint];
    minimal_tree.joints[curr_joint] = min_joint;
    ++curr_joint;
  }

  return minimal_tree;
}
}  // namespace csdecomp