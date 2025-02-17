#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuda_collision_checker.h"
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
}