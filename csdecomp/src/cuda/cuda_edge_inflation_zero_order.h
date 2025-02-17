#pragma once
#include <fmt/core.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "cuda_set_builder_utils.h"
#include "hpolyhedron.h"
#include "minimal_plant.h"
namespace csdecomp {
#define MAX_NUM_PARTICLES 10000
#define MAX_NUM_VOXELS 10000

/**
 * @brief Options for the EI-ZO algorithm.
 */
struct EizoOptions {
  float configuration_margin{0.01};  ///< Margin for configuration space
  float tau{0.5};                 ///< Descision threshold 0.5 is a good choice
  float delta{0.05};              ///< Admissible uncertainty
  float epsilon{0.01};            ///< Admissible fraction in collision
  u_int32_t bisection_steps{9};   ///< Number of bisection steps
  u_int32_t num_particles{1000};  ///< Number of particles to use
  u_int32_t max_hyperplanes_per_iteration{
      10};                                 ///< Max hyperplanes per iteration
  u_int32_t max_iterations{500};           ///< Maximum number of iterations
  u_int32_t mixing_steps{40};              ///< Number of mixing steps
  bool verbose{false};                     ///< Verbose output flag
  bool track_iteration_information{true};  ///< Track iteration info flag
};

/**
 * @brief Information about a single iteration of EI-ZO.
 */
struct EizoIterationInfo {
  int iteration_index;            ///< Iteration number
  int num_samples_drawn;          ///< Number of samples drawn
  int collisions_threshold;       ///< Maximum admissible number of collisions
  int num_collisions_unadaptive;  ///< Number of collisions in unadaptive test
  int num_collisions_updated;  ///< Number of collisions used to optimize faces
  int num_faces_placed;        ///< Number of faces placed
  float iteration_time_ms;     ///< Iteration time in milliseconds
  Eigen::MatrixXf
      points_post_bisection;  ///< Points after bisection (for 2D/3D plotting)
};

/**
 * @brief Aggregate information about a single FastEdgeInflation call.
 */
struct EizoInfo {
  std::vector<EizoIterationInfo> iteration_info;
};

/**
 * @brief GPU memory structure for Edge Inflation Zero-Order algorithm.
 */
struct EizoGpuMemory {
  EizoGpuMemory(const MinimalPlant& plant, const EizoOptions& options,
                const std::vector<GeometryIndex>& robot_geometry_ids,
                const int max_num_voxels);

  const float delta_min;
  const uint32_t max_num_configurations;
  const int dim;

  CudaPtr<float> voxel_buffer;

  // buffers for collision checking in unadaptive test
  CudaPtr<float> transforms_buffer;
  CudaPtr<float> configs_to_check_buffer;

  CudaPtr<uint8_t> is_pair_col_free_buffer;
  CudaPtr<uint8_t> is_config_col_free_buffer;
  CudaPtr<uint8_t> is_geom_to_vox_pair_col_free_buffer;

  // buffers for the bisection updates
  CudaPtr<float> configs_in_collision_buffer;
  CudaPtr<float> updated_configs_buffer;

  CudaPtr<float> bisection_lower_bounds_buffer;
  CudaPtr<float> bisection_upper_bounds_buffer;

  // projection buffers
  CudaPtr<float> projections_buffer;
  CudaPtr<float> distances_buffer;

  // flag for adding faces
  CudaPtr<uint8_t> success_flag_ptr;

  // counters for tracking fraction in collision
  CudaPtr<u_int32_t> collision_counter_ptr;
  CudaPtr<u_int32_t> first_m_collision_counter_ptr;

  // debugging buffer
  std::vector<float> debugging_float_buffer_data;
  CudaPtr<float> debugging_float_buffer;

  // transferred between host and device during iterations
  u_int32_t collision_counter{0};
  uint8_t success_flag{0};
};

#ifdef __CUDACC__

/**
 * @brief Executes the Edge Inflation Zero Order (EIZO) algorithm on GPU. Assmes
 * pre allcated memory.
 *
 * @param P_device Device pointer to MinimalHPolyhedron
 * @param line_start_device Device pointer to line start
 * @param line_end_device Device pointer to line end
 * @param voxels_device Device pointer to voxels
 * @param voxel_radius Radius of voxels
 * @param num_voxels Number of voxels
 * @param robot_geometry_ids Device pointer to robot geometry IDs
 * @param num_robot_geometries Number of robot geometries
 * @param plant_device Device pointer to MinimalPlant
 * @param plant_host Host pointer to MinimalPlant
 * @param options_device Device pointer to EizoOptions
 * @param options_host Host pointer to EizoOptions
 * @param mem GPU memory structure
 * @param line_segment_idxs_buffer Device pointer to line segment indices
 * @param eizo_info EI-ZO info structure
 */
void executeEizo(MinimalHPolyhedron* P_device, const float* line_start_device,
                 const float* line_end_device, const float* voxels_device,
                 const float voxel_radius, const u_int32_t num_voxels,
                 const GeometryIndex* robot_geometry_ids,
                 const u_int32_t num_robot_geometries,
                 const MinimalPlant* plant_device,
                 const MinimalPlant* plant_host,
                 const EizoOptions* options_device,
                 const EizoOptions* options_host, EizoGpuMemory& mem,
                 const u_int32_t* line_segment_idxs_buffer,
                 EizoInfo& eizo_info);
#endif

/**
 * @brief High-level CPP function call that performs EI-ZO using
 * CUDA. This directly handles memory allocation. For faster calls consider
 * using the set builder class that pre-allocates the memory.
 *
 * @param line_start Start point of the line segment
 * @param line_end End point of the line segment
 * @param voxels Voxel representation of the environment
 * @param voxel_radius Radius of each voxel
 * @param domain Initial domain for inflation
 * @param plant Minimal plant structure
 * @param robot_geometry_ids Vector of robot geometry indices
 * @param options EI-ZO options
 * @param eizo_info Output structure for inflation information
 * @return MinimalHPolyhedron Resulting inflated polyhedron
 */
const MinimalHPolyhedron EizoCuda(
    const Eigen::VectorXf& line_start, const Eigen::VectorXf& line_end,
    const Voxels& voxels, const float voxel_radius, const HPolyhedron& domain,
    const MinimalPlant& plant,
    const std::vector<GeometryIndex>& robot_geometry_ids,
    const EizoOptions& options, EizoInfo& eizo_info);

/**
 * @brief Prints a summary of the EI-ZO process.
 *
 * @param info EI-ZO information structure
 */
void inline printEizoSummary(const EizoInfo& info) {
  std::cout << "EI-ZO Summary:" << std::endl;
  std::cout << "=============================" << std::endl;
  std::cout << "Total iterations: " << info.iteration_info.size() << std::endl
            << std::endl;

  int total_samples = 0;
  int total_collisions_unadaptive = 0;
  int total_collisions_updated = 0;
  int total_faces_placed = 0;
  float total_time_ms = 0.0f;

  // Updated column headers with shorter names
  std::cout << std::left << std::setw(8) << "Iter." << std::setw(10)
            << "Samples" << std::setw(12) << "Col. (UT)" << std::setw(10)
            << "Thresh." << std::setw(12) << "Col. (OPT)" << std::setw(8)
            << "Faces" << std::setw(12) << "Time (ms)" << std::endl;
  std::cout << std::string(72, '-') << std::endl;

  for (size_t i = 0; i < info.iteration_info.size(); ++i) {
    const auto& iter = info.iteration_info[i];
    total_samples += iter.num_samples_drawn;
    total_collisions_unadaptive += iter.num_collisions_unadaptive;
    total_collisions_updated += iter.num_collisions_updated;
    total_faces_placed += iter.num_faces_placed;
    total_time_ms += iter.iteration_time_ms;

    std::cout << std::left << std::setw(8) << iter.iteration_index
              << std::setw(10) << iter.num_samples_drawn << std::setw(12)
              << iter.num_collisions_unadaptive << std::setw(10)
              << iter.collisions_threshold << std::setw(12)
              << iter.num_collisions_updated << std::setw(8)
              << iter.num_faces_placed << std::setw(12) << std::fixed
              << std::setprecision(2) << iter.iteration_time_ms << std::endl;
  }

  std::cout << std::string(72, '-') << std::endl;
  std::cout << "Totals:" << std::endl;
  std::cout << "  Samples drawn: " << total_samples << std::endl;
  std::cout << "  Collisions (UT): " << total_collisions_unadaptive
            << std::endl;
  std::cout << "  Collisions (OPT): " << total_collisions_updated << std::endl;
  std::cout << "  Faces placed: " << total_faces_placed << std::endl;
  std::cout << "  Total time: " << std::fixed << std::setprecision(2)
            << total_time_ms << " ms" << std::endl;
}
}  // namespace csdecomp