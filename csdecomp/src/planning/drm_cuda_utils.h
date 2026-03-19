#pragma once

#include <cuda_runtime_api.h>

namespace csdecomp {
/**
 * @brief Computes distances between a current joint configuration and a set of
 * target configurations.
 *
 * This CUDA function calculates the Euclidean distance between a single current
 * joint configuration and multiple target joint configurations in parallel.
 *
 * @param target_joints Pointer to an array of target joint configurations.
 *                      Shape: [num_configurations, num_dofs]
 * @param current_joint Pointer to the current joint configuration.
 *                      Shape: [num_dofs]
 * @param num_dofs Number of degrees of freedom (DOFs) in each joint
 * configuration.
 * @param num_configurations Number of target configurations to compare against.
 * @param distance Pointer to the output array where computed distances will be
 * stored. Shape: [num_configurations]
 * @param stream CUDA stream to use for asynchronous execution (optional,
 * default: nullptr).
 *
 * @note This function is designed to be called from host code and executed on a
 * CUDA-capable GPU.
 * @note The function assumes that all input and output arrays are pre-allocated
 * in GPU memory.
 * @note When stream is nullptr, the function will use the default CUDA stream.
 */
void ComputeJointDistances(const float* target_joints,
                           const float* current_joint, int num_dofs,
                           int num_configurations, float* distance,
                           cudaStream_t stream = nullptr);
}  // namespace csdecomp