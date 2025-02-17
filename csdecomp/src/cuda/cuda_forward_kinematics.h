#pragma once
#include <Eigen/Dense>

#include "minimal_kinematic_tree.h"

namespace csdecomp {
/**
 * @brief Low-level C++ wrapper for evaluating forward kinematics on GPU.
 *
 * This function executes the forward kinematics computation on the GPU. It
 * expects all data to already be present in GPU memory.
 *
 * @param transforms_flat_device Pointer to GPU memory where the resulting
 * transforms will be stored. This should be an array of 4x4 matrices stored in
 * column-major order. The size should be 16 * num_configurations * num_joints
 * floats.
 * @param configurations_device Pointer to GPU memory containing joint
 * configurations. Size should be num_configurations * num_joints floats.
 * @param num_configurations The number of configurations to compute.
 * @param tree_device Pointer to the MinimalKinematicTree structure in GPU
 * memory.
 *
 * @note All pointers must be pointing to memory on the GPU.
 * @note The caller is responsible for memory management (allocation and
 * freeing) of GPU resources.
 */
void executeForwardKinematicsKernel(float *transforms_flat_device,
                                    const float *configurations_device,
                                    const int num_configurations,
                                    const MinimalKinematicTree *tree_device);

/**
 * @brief High-level C++ wrapper for evaluating forward kinematics on GPU.
 *
 * This function handles memory allocation and transfer between CPU and GPU,
 * executes the forward kinematics computation, and returns the results.
 * It is designed to be exposed to Python.
 *
 * @param transforms Pointer to an Eigen::MatrixXf where the resulting
 * transforms will be stored. On output, this matrix will contain
 * num_configurations * num_links 4x4 matrices stored sequentially. The size
 * should be (16 * num_links) rows by num_configurations columns.
 * @param configurations Pointer to an Eigen::MatrixXf containing the input
 * joint configurations. Each column represents one configuration. The size
 * should be num_joints rows by num_configurations columns.
 * @param tree Pointer to the MinimalKinematicTree structure
 * describing the kinematic tree.
 *
 * @note This function allocates and frees GPU memory internally.
 * @note The transforms matrix will be resized if necessary to accommodate the
 * output.
 */
void computeForwardKinematicsCuda(Eigen::MatrixXf *transforms,
                                  const Eigen::MatrixXf *configurations,
                                  const MinimalKinematicTree *tree);

#ifdef __CUDACC__

// Evaluates the forward kinematics of a single configuration and fills all
// transforms into transforms_flat. Transforms flat is a 4xN matrix stored in
// columnmajor, where N is the number of transforms
__device__ void singleConfigForwardKinematics(
    float *transforms_flat_device, const float *configuration_device,
    const MinimalKinematicTree *tree_device);

#endif
}  // namespace csdecomp