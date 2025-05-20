#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_collision_checker.h"
#include "cuda_edit_regions.h"
#include "cuda_hit_and_run_sampling.h"
#include "cuda_polytope_builder.h"
#include "cuda_visibility_graph.h"
#include "voxel_wrapper.h"

namespace py = pybind11;
using namespace csdecomp;

void add_cuda_bindings(py::module &m) {
  py::class_<EizoOptions>(m, "EizoOptions")
      .def(py::init<>())
      .def_readwrite("configuration_margin", &EizoOptions::configuration_margin)
      .def_readwrite("tau", &EizoOptions::tau)
      .def_readwrite("delta", &EizoOptions::delta)
      .def_readwrite("epsilon", &EizoOptions::epsilon)
      .def_readwrite("bisection_steps", &EizoOptions::bisection_steps)
      .def_readwrite("num_particles", &EizoOptions::num_particles)
      .def_readwrite("max_hyperplanes_per_iteration",
                     &EizoOptions::max_hyperplanes_per_iteration)
      .def_readwrite("max_iterations", &EizoOptions::max_iterations)
      .def_readwrite("mixing_steps", &EizoOptions::mixing_steps)
      .def_readwrite("verbose", &EizoOptions::verbose)
      .def_readwrite("track_iteration_information",
                     &EizoOptions::track_iteration_information);

  py::class_<CudaEdgeInflator>(m, "CudaEdgeInflator")
      .def(py::init<const MinimalPlant &, const std::vector<GeometryIndex> &,
                    const EizoOptions &, const HPolyhedron &>(),
           py::arg("plant"), py::arg("robot_geometry_ids"), py::arg("options"),
           py::arg("domain"))
      .def(
          "inflateEdge",
          [](CudaEdgeInflator &self, const Eigen::VectorXf &line_start,
             const Eigen::VectorXf &line_end, const VoxelsWrapper &vox,
             const float voxel_radius, bool verbose = true) {
            return self.inflateEdge(line_start, line_end, vox.matrix,
                                    voxel_radius, verbose);
          },
          py::arg("line_start"), py::arg("line_end"), py::arg("voxels"),
          py::arg("voxel_radius"), py::arg("verbose") = true);

  m.def("UniformSampleInHPolyhedraCuda", &UniformSampleInHPolyhedronCuda,
        py::arg("polyhedra"), py::arg("starting_points"),
        py::arg("num_samples_per_hpolyhedron"), py::arg("mixing_steps"),
        py::arg("seed") = 1337,
        R"pbdoc(
    Uniform sampling in multiple HPolyhedra using CUDA.

    Args:
        polyhedra (List[HPolyhedron]): List of HPolyhedron objects to sample from.
        starting_points (numpy.ndarray): Matrix of starting points for sampling.
        num_samples_per_hpolyhedron (int): Number of samples to generate per HPolyhedron.
        mixing_steps (int): Number of mixing steps in the sampling process.
        seed (int): Seed for the sampling process.

    Returns:
        List[numpy.ndarray]: List of matrices, each containing samples for one HPolyhedron.
    )pbdoc");

  m.def(
      "CheckCollisionFreeVoxelsCuda",
      [](const Eigen::MatrixXf &configurations, const VoxelsWrapper &voxels,
         float voxel_radius, const MinimalPlant &plant,
         const std::vector<GeometryIndex> &robot_geometries) {
        return checkCollisionFreeVoxelsCuda(&configurations, &(voxels.matrix),
                                            voxel_radius, &plant,
                                            robot_geometries);
      },
      py::arg("configurations"), py::arg("voxels"), py::arg("voxel_radius"),
      py::arg("plant"), py::arg("robot_geometries"),
      R"pbdoc(
    High-level C++ wrapper for CUDA-based collision checking against voxel maps.

    This function handles memory allocation and performs collision checking using CUDA.
    It includes self-collision checks and checks against the static environment.

    Args:
        configurations (numpy.ndarray): Matrix of configurations to check.
        voxels (Voxels): Voxel map representing the environment.
        voxel_radius (float): Radius of each voxel.
        plant (MinimalPlant): Object containing kinematics and geometry information for collision checking.
        robot_geometries (List[GeometryIndex]): List of geometry indices for the robot.

    Returns:
        numpy.ndarray: An array of boolean values indicating collision-free status for each configuration.
    )pbdoc");

  m.def(
      "CheckCollisionFreeCuda",
      [](const Eigen::MatrixXf &configurations, const MinimalPlant &plant) {
        return checkCollisionFreeCuda(&configurations, &plant);
      },
      py::arg("configurations"), py::arg("plant"),
      R"pbdoc(
    High-level C++ wrapper for CUDA-based collision checking.

    This function handles memory allocation and performs collision checking using CUDA.

    Args:
        configurations (numpy.ndarray): Matrix of configurations to check.
        plant (MinimalPlant): Object containing kinematics and geometry information for collision checking.

    Returns:
        numpy.ndarray: An array of boolean values indicating collision-free status for each configuration.
    )pbdoc");

  m.def("VisibilityGraph", &VisibilityGraph, py::arg("configurations"),
        py::arg("plant"), py::arg("step_size"),
        py::arg("max_configs_per_batch") = 10000000,
        R"pbdoc(
    Computes a visibility graph between a collection of passed configurations using CUDA. 
    It automatically handles loadbalancing based on the value passed for max_configs_per_batch.

    Args:
        configurations (numpy.ndarray): Dim x num_configurateions matrix of configurations to check.
        plant (MinimalPlant): Object containing kinematics and geometry information for collision checking.
        step_size (float): Step size for collision checks along edges.
        max_configs_per_batch (int): Max number of collision checks per batch. This is used for splitting up the edges into batches.  

    Returns:
        numpy.ndarray: An array of boolean values indicating collision-free status for each configuration.
    )pbdoc");

  py::class_<EditRegionsOptions>(m, "EditRegionsOptions")
      .def(py::init<>())
      .def_readwrite("configuration_margin",
                     &EditRegionsOptions::configuration_margin)
      .def_readwrite("bisection_steps", &EditRegionsOptions::bisection_steps)
      .def_readwrite("verbose", &EditRegionsOptions::verbose);

  m.def(
      "EditRegionsCuda",
      [](const Eigen::MatrixXf &collisions,
         const std::vector<u_int32_t> &line_segment_idxs,
         const Eigen::MatrixXf &line_start_points,
         const Eigen::MatrixXf &line_end_points,
         const std::vector<HPolyhedron> &regions, const MinimalPlant &plant,
         const std::vector<GeometryIndex> &robot_geometry_ids,
         const VoxelsWrapper &voxels, float voxel_radius,
         const EditRegionsOptions &options) {
        return EditRegionsCuda(collisions, line_segment_idxs, line_start_points,
                               line_end_points, regions, plant,
                               robot_geometry_ids, voxels.matrix, voxel_radius,
                               options);
      },
      py::arg("collisions"), py::arg("line_segment_idxs"),
      py::arg("line_start_points"), py::arg("line_end_points"),
      py::arg("regions"), py::arg("plant"), py::arg("robot_geometry_ids"),
      py::arg("voxels"), py::arg("voxel_radius"), py::arg("options"),
      R"pbdoc(
    Refines convex regions by removing collisions detected in trajectories.

    This function implements the recovery mechanism from "Superfast Configuration-Space Convex Set 
    Computation on GPUs for Online Motion Planning". It takes colliding configurations found in 
    trajectories and modifies the corresponding convex sets to exclude these collisions while 
    ensuring that line segments remain contained in the sets.

    Args:
        collisions (numpy.ndarray): Matrix where each column represents a colliding configuration.
        line_segment_idxs (List[int]): List of line segment indices which indicate which line segment to use for the optimization of each collision.
        line_start_points (numpy.ndarray): Matrix where each column is the start point of a line segment.
        line_end_points (numpy.ndarray): Matrix where each column is the end point of a line segment.
        regions (List[HPolyhedron]): Vector of convex regions (polytopes) to be modified.
        plant (MinimalPlant): Kinematics and collision model of the robot.
        robot_geometry_ids (List[GeometryIndex]): Indices of robot geometries to check for collisions.
        voxels (Voxels): Voxel representation of obstacles in the environment.
        voxel_radius (float): Radius of spheres associated with voxels for collision checking.
        options (EditRegionsOptions): Configuration options for the region editing process.

    Returns:
        tuple: A tuple containing:
            - List[HPolyhedron]: Refined convex regions with collisions removed.
            - tuple: A pair of matrices containing any line segments that needed to be re-inflated.
    )pbdoc");
}