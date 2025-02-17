#pragma once
#include <Eigen/Dense>
#include <vector>

#include "hpolyhedron.h"

namespace csdecomp {

/**
 * @brief Executes uniform sampling within H-polyhedrons using a CUDA kernel.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param samples Output array for the sampled points
 * @param poly_array Array of MinimalHPolyhedron structures
 * @param starting_points Array of starting points for the sampling process
 * @param num_samples_per_hpolyhedron Number of samples to generate per
 * H-polyhedron
 * @param dim Dimensionality of the space
 * @param num_polytopes Number of H-polyhedrons to sample from
 * @param mixing_steps Number of mixing steps in the sampling process
 * @param seed random seed for curandstate
 */
void executeUniformSampleInHPolyhedronKernel(
    float* samples, const MinimalHPolyhedron* poly_array,
    const float* starting_points, const int32_t num_samples_per_hpolyhedron,
    const int32_t dim, const int32_t num_polytopes, const int32_t mixing_steps,
    const u_int64_t seed = 1337);

/**
 * @brief Performs uniform sampling within multiple H-polyhedra using a CUDA
 * implementation of hit-and-run sampling. This function is a high-level C++
 * wrapper that handles memory allocation and calls the CUDA kernel for
 *sampling. For details see Lovász, László. "Hit-and-run mixesfast."
 *Mathematical programming 86 (1999): 443-461.
 *
 * @param polyhedra Vector of H-polyhedrons to sample from (must all be in the
 *same dimension)
 * @param starting_points Matrix of starting points for the sampling process
 * @param num_samples_per_hpolyhedron Number of samples to generate per
 *H-polyhedron
 * @param mixing_steps Number of mixing steps in the sampling process
 * @param seed random seed for curandstate
 * @return std::vector<Eigen::MatrixXf> Vector of matrices, each containing
 *samples for one H-polyhedron
 */
const std::vector<Eigen::MatrixXf> UniformSampleInHPolyhedronCuda(
    const std::vector<HPolyhedron>& polyhedra,
    const Eigen::MatrixXf& starting_points,
    const int num_samples_per_hpolyhedron, const int mixing_steps,
    const int64_t seed = 1337);
}  // namespace csdecomp