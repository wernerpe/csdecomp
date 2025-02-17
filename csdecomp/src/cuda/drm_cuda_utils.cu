#include <cassert>

#include "csdecomp/src/cuda/drm_cuda_utils.h"
namespace csdecomp {
namespace {
__global__ void ComputeJointDistancesKernel(const float* target_joints,
                                            const float* current_joint,
                                            int num_dofs,
                                            int num_configurations,
                                            float* distance) {
  for (int out_index = blockIdx.x * blockDim.x + threadIdx.x;
       out_index < num_configurations; out_index += blockDim.x * gridDim.x) {
    const int node_index = out_index * num_dofs;
    float joints_distance = 0.0;
    for (int ii = 0; ii < num_dofs; ++ii) {
      joints_distance += (target_joints[node_index + ii] - current_joint[ii]) *
                         (target_joints[node_index + ii] - current_joint[ii]);
    }
    distance[out_index] = joints_distance;
  }
}
}  // namespace

void ComputeJointDistances(const float* target_joints,
                           const float* current_joint, int num_dofs,
                           int num_configurations, float* distance,
                           cudaStream_t stream) {
  assert(target_joints != nullptr);
  assert(current_joint != nullptr);
  assert(num_dofs > 0);
  assert(num_configurations > 0);
  assert(distance != nullptr);

  int threads_per_block = 128;
  int num_blocks =
      (num_configurations + threads_per_block - 1) / threads_per_block;
  ComputeJointDistancesKernel<<<num_blocks, threads_per_block, 0, stream>>>(
      target_joints, current_joint, num_dofs, num_configurations, distance);
}
}  // namespace csdecomp