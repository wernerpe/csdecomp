#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "linesegment_aabb_checker.h"

namespace py = pybind11;
using namespace csdecomp;

void add_cpp_utils_bindings(py::module &m) {
  // Add the LinesegmentAABBIntersecting function for VectorXd
  m.def("LinesegmentAABBIntersecting", &csdecomp::LinesegmentAABBIntersecting,
        py::arg("p1"), py::arg("p2"), py::arg("box_min"), py::arg("box_max"),
        "Check if a line segment intersects with an axis-aligned bounding box "
        "(AABB).\n\n"
        "Args:\n"
        "    p1 (numpy.ndarray): First endpoint of the line segment.\n"
        "    p2 (numpy.ndarray): Second endpoint of the line segment.\n"
        "    box_min (numpy.ndarray): Minimum corner of the AABB.\n"
        "    box_max (numpy.ndarray): Maximum corner of the AABB.\n\n"
        "Returns:\n"
        "    bool: True if the line segment intersects with the AABB, False "
        "otherwise.\n\n"
        "Note:\n"
        "    All input vectors must have the same dimension (2 or 3).");
}