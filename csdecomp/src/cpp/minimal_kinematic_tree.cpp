#include "minimal_kinematic_tree.h"

#include <fmt/core.h>

#include <Eigen/Dense>
#include <stdexcept>

namespace csdecomp {

void computeLinkFrameToWorldTransformsMinimal(
    const Eigen::VectorXf &joint_values,
    const MinimalKinematicTree &minimal_tree, Eigen::MatrixXf *transforms) {
  // check that the dimensions are correct
  if (transforms->rows() != 4 * minimal_tree.num_links) {
    throw std::runtime_error(fmt::format(
        "[CSDECOMP:MINIMALKINEMATICTREE] Error the transform matrix is "
        "{}x{} but {}x4 is requested",
        transforms->rows(), transforms->cols(), 4 * minimal_tree.num_links));
  }
  // root transform is always identity
  transforms->topLeftCorner(4, 4).setIdentity();

  for (int traversal_idx = 0; traversal_idx < minimal_tree.num_joints;
       ++traversal_idx) {
    int current_joint_index =
        minimal_tree.joint_traversal_sequence[traversal_idx];
    int parent_link_index =
        minimal_tree.joints[current_joint_index].parent_link;
    int child_link_index = minimal_tree.joints[current_joint_index].child_link;
    Eigen::Matrix4f X_W_PL = transforms->block(4 * parent_link_index, 0, 4, 4);
    Eigen::Matrix4f X_PL_J = minimal_tree.joints[current_joint_index].X_PL_J;
    float config_value = 0;
    // if the joint is not fixed we extract the config value
    if (minimal_tree.joints[current_joint_index].type != JointType::FIXED) {
      config_value =
          joint_values[minimal_tree
                           .joint_idx_to_config_idx[current_joint_index]];
    }
    // Compute the active transform from joint
    Eigen::Matrix4f X_J_L = Eigen::Matrix4f::Identity();
    // compute X_PL_L -> apply joint transform
    if (minimal_tree.joints[current_joint_index].type == JointType::REVOLUTE) {
      Eigen::Matrix3f rotation =
          Eigen::AngleAxisf(config_value,
                            minimal_tree.joints[current_joint_index].axis)
              .toRotationMatrix();
      X_J_L.block<3, 3>(0, 0) = rotation;
    } else if (minimal_tree.joints[current_joint_index].type ==
               JointType::PRISMATIC) {
      X_J_L.block<3, 1>(0, 3) +=
          minimal_tree.joints[current_joint_index].axis * config_value;
    }

    // combine X_W_PL * X_PL_L to get the link transform
    Eigen::Matrix4f X_W_L = X_W_PL * X_PL_J * X_J_L;
    transforms->block(4 * child_link_index, 0, 4, 4) = X_W_L;
  }
}
}  // namespace csdecomp