#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "collision_checker.h"
#include "voxel_wrapper.h"

namespace py = pybind11;
using namespace csdecomp;

void add_collision_checker_bindings(py::module &m) {
  m.def("PairCollisionFree", &pairCollisionFree, py::arg("geom_a"),
        py::arg("x_w_la"), py::arg("geom_b"), py::arg("x_w_lb"),
        "Check if two collision geometries are collision-free given their "
        "world transforms");

  m.def("CheckCollisionFree", &checkCollisionFree, py::arg("configuration"),
        py::arg("plant"), "Check if a configuration is collision-free.");

  m.def(
      "CheckCollisionFreeVoxels",
      [](const Eigen::VectorXf &configuration, const VoxelsWrapper &vox,
         const float &voxel_radius, const MinimalPlant &mp,
         const std::vector<int> &robot_geometry_indices) {
        std::vector<GeometryIndex> rob_geom_ids(robot_geometry_indices.size());
        for (auto id : robot_geometry_indices) {
          if (id >= mp.num_scene_geometries) {
            throw std::runtime_error(
                "Invalid robot geometry ID, check the SceneInspector for vaild "
                "ids.");
          }
          rob_geom_ids.emplace_back((GeometryIndex)id);
        }

        return checkCollisionFreeVoxels(configuration, vox.getMatrix(),
                                        voxel_radius, mp, rob_geom_ids);
      },
      py::arg("configuration"), py::arg("voxels"), py::arg("voxel_radius"),
      py::arg("plant"), py::arg("robot_geometry_index"),
      "Check if a configuration is collision-free against a voxel map.");
}