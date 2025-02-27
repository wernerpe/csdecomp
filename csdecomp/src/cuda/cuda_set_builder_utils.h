#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "cuda_utilities.h"
#include "hpolyhedron.h"

namespace csdecomp {
/**
 * @brief Projects samples onto line segments using CUDA.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param distances Output array for distances of projections
 * @param projections Output array for projected points
 * @param samples Input array of sample points
 * @param line_start_points Array of start points of line segments
 * @param line_end_points Array of end points of line segments
 * @param line_segment_idxs Array of indices for line segments
 * @param num_samples Number of sample points
 * @param dimension Dimensionality of the space
 */
void executeProjectSamplesOntoLineSegmentsKernel(
    float* distances, float* projections, const float* samples,
    const float* line_start_points, const float* line_end_points,
    const u_int32_t* line_segment_idxs, const u_int32_t num_samples,
    const u_int32_t dimension);

/**
 * @brief Computes the midpoint between start and end points using CUDA.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param results Output array for midpoints
 * @param start_points Array of start points
 * @param end_points Array of end points
 * @param num_points Number of point pairs
 * @param dimension Dimensionality of the space
 */
void executeStepConfigToMiddleKernel(float* results, const float* start_points,
                                     const float* end_points,
                                     const u_int32_t num_points,
                                     const u_int32_t dimension);

/**
 * @brief Updates bisection bounds based on collision checks using CUDA.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param bisection_lower_bound_device Lower bounds for bisection
 * @param bisection_upper_bound_device Upper bounds for bisection
 * @param currently_checked_configs Configurations being checked
 * @param is_config_col_free Array indicating if configurations are
 * collision-free
 * @param num_configs Number of configurations
 * @param dimension Dimensionality of the space
 */
void executeUpdateBisectionBoundsKernel(float* bisection_lower_bound_device,
                                        float* bisection_upper_bound_device,
                                        const float* currently_checked_configs,
                                        uint8_t* is_config_col_free,
                                        u_int32_t num_configs,
                                        u_int32_t dimension);

/**
 * @brief Computes distances between pairs of points using CUDA.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param distances Output array for computed distances
 * @param pointsA First set of points
 * @param pointsB Second set of points
 * @param num_configs Number of point pairs
 * @param dimension Dimensionality of the space
 */
void executePointToPointDistanceKernel(float* distances, const float* pointsA,
                                       const float* pointsB,
                                       u_int32_t num_configs,
                                       u_int32_t dimension);

/**
 * @brief Stores configurations from the current_configs buffer that are in
 * collision in the configs_in_collision buffer using CUDA.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param configs_in_collision Output array for configurations in collision
 * @param current_configs Current configurations being checked
 * @param is_current_config_col_free Array indicating if current configs are
 * collision-free
 * @param num_configs Number of configurations
 * @param dimension Dimensionality of the space
 */
void executeStoreIfCollisionKernel(float* configs_in_collision,
                                   const float* current_configs,
                                   uint8_t* is_current_config_col_free,
                                   u_int32_t num_configs, u_int32_t dimension);

/**
 * @brief Adds a face to the polyhedron and updates the distances of particles
 * to MAX_F that are made redundant by the new face.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param p_device Pointer to the minimal H-polyhedron on the device
 * @param distances Array of distances - these are updated to MAX_F for
 * particles outside of the set
 * @param points_in_collision Array of points in collision
 * @param projections Array of projections
 * @param line_start Start point of the line segment
 * @param line_end End point of the line segment
 * @param particle_idx_to_add Index of the particle to add
 * @param num_configs Number of configurations
 * @param dim Dimensionality of the space
 * @param stepback Stepback parameter
 * @param success_flag Pointer to flag indicating success
 * @return uint8_t Returns 1 if done (element to add has distance equal to
 * MAX_F), 0 otherwise
 */
uint8_t executeAddFaceAndCheckOtherParticleRedundantKernel(
    MinimalHPolyhedron* p_device, float* distances, float* points_in_collision,
    const float* projections, const float* line_start, const float* line_end,
    const u_int32_t particle_idx_to_add, const u_int32_t num_configs,
    const u_int32_t dim, const float stepback, uint8_t* success_flag);

/**
 * @brief Projects samples onto line segments using CUDA. This function
 * expects all inputs to be on the host. (It takes care of memory allocation)
 *
 * @param line_start_points Matrix of line segment start points
 * @param line_end_points Matrix of line segment end points
 * @param samples Vector of sample matrices
 * @return std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>>
 *         Pair of vectors containing distances and projections
 */
const std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>>
projectSamplesOntoLineSegmentsCuda(const Eigen::MatrixXf& line_start_points,
                                   const Eigen::MatrixXf& line_end_points,
                                   const std::vector<Eigen::MatrixXf>& samples);
/**
 * @brief This function goes over checked_configurations_device and copies up to
 * max_num_update_particles to configurations_in_collsion_device. This function
 * is used to consolidate the memory of the particles that are in collision for
 * the bisection updates in the EI-ZO algorithm.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param configurations_in_collision_device Output array for configurations in
 * collision
 * @param checked_configs_device Array of checked configurations
 * @param is_config_free_device Array indicating if configurations are free
 * @param max_num_update_particles Maximum number of update particles
 * @param num_configs Number of configurations
 * @param dimension Dimensionality of the space
 * @param collision_counter Pointer to collision counter
 * @return u_int32_t Number of configurations in collision
 */
u_int32_t executeCollisionConsolidationKernel(
    float* configurations_in_collision_device,
    const float* checked_configs_device, const uint8_t* is_config_free_device,
    const u_int32_t max_num_update_particles, const u_int32_t num_configs,
    const u_int32_t dimension, u_int32_t* collision_counter);
/**
 * @brief This function counts the collisions in the first M entries of
 * is_config_free_device.
 *
 * This function is a C++ wrapper that calls the corresponding CUDA kernel.
 * It assumes that memory is pre-allocated on the GPU.
 *
 * @param is_config_free_device Array indicating if configurations are free
 * @param M Portion of array to consider
 * @param num_configs Number of configurations, (length of @param
 * is_config_free_device)
 * @param dimension Dimensionality of the space
 * @param collision_counter Pointer to collision counter
 * @return u_int32_t Number of configurations in collision in first @param M
 * entries
 */
u_int32_t executeCountCollisionsInFirstMEntriesKernel(
    const uint8_t* is_config_free_device, const u_int32_t M,
    const u_int32_t num_configs, u_int32_t* first_m_collision_counter);
/**
 * @brief Calculates the number of samples for the undadaptive test.
 *
 * @param epsilon Admissible fraction in collision
 * @param delta Allowed uncertainty
 * @param tau Acceptance threshold -> 0.5 is a good value.
 * @return u_int32_t Number of samples
 */
constexpr u_int32_t inline numSamplesUnadaptive(const float& epsilon,
                                                const float& delta,
                                                const float& tau) {
  return static_cast<u_int32_t>(
      std::ceil(2 * std::log(1 / delta) / (epsilon * tau * tau)));
}
}  // namespace csdecomp