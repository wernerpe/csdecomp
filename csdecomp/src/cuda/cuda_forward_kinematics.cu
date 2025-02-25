#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "cuda_forward_kinematics.h"
#include "cuda_utilities.h"

namespace csdecomp {

__device__ void singleConfigForwardKinematics(
    float *transforms_flat, const float *configuration_device,
    const MinimalKinematicTree *tree_device) {
  // make sure the root transform is identity ->0 offset from pointer
  set4x4BlockIdentityOf4xNMatrix(transforms_flat);

  // now loop over the traversal order and compute transforms
  for (int traversal_idx = 0; traversal_idx < tree_device->num_joints;
       ++traversal_idx) {
    int current_joint_index =
        tree_device->joint_traversal_sequence[traversal_idx];
    int parent_link_index =
        tree_device->joints[current_joint_index].parent_link;
    int child_link_index = tree_device->joints[current_joint_index].child_link;

    const Eigen::Matrix4f X_W_PL =
        Eigen::Map<Eigen::Matrix4f>(transforms_flat + 16 * parent_link_index);
    const Eigen::Matrix4f X_PL_J =
        tree_device->joints[current_joint_index].X_PL_J;
    float config_value = 0;
    if (tree_device->joints[current_joint_index].type != JointType::FIXED) {
      config_value = configuration_device
          [tree_device->joint_idx_to_config_idx[current_joint_index]];
    }

    // Compute the active transform from joint
    Eigen::Matrix4f X_J_L = Eigen::Matrix4f::Identity();
    if (tree_device->joints[current_joint_index].type == JointType::REVOLUTE) {
      Eigen::Matrix3f rotation =
          Eigen::AngleAxisf(config_value,
                            tree_device->joints[current_joint_index].axis)
              .toRotationMatrix();
      // setTopLeft3x3Blockof4x4(X_J_L, rotation.data());
      X_J_L.block<3, 3>(0, 0) = rotation;
    } else if (tree_device->joints[current_joint_index].type ==
               JointType::PRISMATIC) {
      X_J_L.block<3, 1>(0, 3) +=
          tree_device->joints[current_joint_index].axis * config_value;
    }
    Eigen::Matrix4f X_W_L = X_W_PL * X_PL_J * X_J_L;
    Eigen::Map<Eigen::Matrix4f>(transforms_flat + 16 * child_link_index) =
        X_W_L;
  }
}

__global__ void forwardKinematicsKernel(
    float *transforms_flat, const float *configurations,
    const int num_parallel_evals, const MinimalKinematicTree *tree_device) {
  // extract thread info
  int cu_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (cu_index >= num_parallel_evals) {
    return;
  }
  // get pointer to configuration to evaluate
  const float *configuration =
      configurations + cu_index * tree_device->num_configuration_variables;
  // get pointer to destination where the transforms need to be written
  float *transforms_dest =
      transforms_flat + 16 * cu_index * tree_device->num_links;
  // evaluate forward kinematics for the configuration, and write transforms
  // to destination.
  singleConfigForwardKinematics(transforms_dest, configuration, tree_device);
}

void executeForwardKinematicsKernel(float *transforms_flat_device,
                                    const float *configurations_device,
                                    const int num_configurations,
                                    const MinimalKinematicTree *tree_device) {
  int threads_per_block = 128;
  int num_blocks =
      (num_configurations + threads_per_block - 1) / threads_per_block;
  forwardKinematicsKernel<<<num_blocks, threads_per_block>>>(
      transforms_flat_device, configurations_device, num_configurations,
      tree_device);
}

void computeForwardKinematicsCuda(Eigen::MatrixXf *transforms,
                                  const Eigen::MatrixXf *configurations,
                                  const MinimalKinematicTree *tree) {
  // check that transforms matrix has the correct dimensions. Should be
  // 4*num_configs*num_links.
  int num_configs = configurations->cols();
  int num_transforms_tot = transforms->cols() / 4;
  assert(num_configs * tree->num_links == num_transforms_tot &&
         "Transforms matrix has incorrect size");
  assert(configurations->rows() == tree->num_configuration_variables &&
         "incorrect number of configuration variables");

  // allocate memory for transforms, joint values and the kinematic tree
  CudaPtr<const float> configurations_ptr(configurations->data(),
                                          configurations->size());
  CudaPtr<float> transforms_ptr(transforms->data(), transforms->size());
  CudaPtr<const MinimalKinematicTree> tree_ptr(tree, 1);

  // copy kinematic tree and joint values
  configurations_ptr.copyHostToDevice();
  tree_ptr.copyHostToDevice();

  // call forward kinematics kernel
  auto start = std::chrono::high_resolution_clock::now();
  executeForwardKinematicsKernel(transforms_ptr.device,
                                 configurations_ptr.device,
                                 configurations->cols(), tree_ptr.device);
  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();

  std::string tmp = fmt::format(
      "execution time cuda no copy: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  // retrieve answers and free memory
  transforms_ptr.copyDeviceToHost();
}
}  // namespace csdecomp