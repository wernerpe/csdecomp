#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "distance_aabb_linesegment.h"
#include "linesegment_aabb_checker.h"

namespace py = pybind11;
using namespace csdecomp;

void add_cpp_utils_bindings(py::module& m) {
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
  m.def(
      "LinesegmentAABBsIntersecting", &csdecomp::LinesegmentAABBsIntersecting,
      py::arg("p1"), py::arg("p2"), py::arg("boxes_min"), py::arg("boxes_max"),
      "Check if a line segment intersects with multiple axis-aligned "
      "bounding boxes "
      "(AABBs).\n\n"
      "Args:\n"
      "    p1 (numpy.ndarray): First endpoint of the line segment.\n"
      "    p2 (numpy.ndarray): Second endpoint of the line segment.\n"
      "    boxes_min (numpy.ndarray): Minimum corners of the AABBs as a matrix "
      "of shape (dim, N).\n"
      "    boxes_max (numpy.ndarray): Maximum corners of the AABBs as a matrix "
      "of shape (dim, N).\n\n"
      "Returns:\n"
      "    list: A list of boolean values, where True indicates the line "
      "segment intersects\n"
      "          with the corresponding AABB, False otherwise.\n\n"
      "Note:\n"
      "    The dimension of p1 and p2 must match the number of rows in "
      "boxes_min and boxes_max.\n"
      "    box_min and box_max must have the same shape.");

  m.def(
      "PwlPathAABBsIntersecting", &csdecomp::PwlPathAABBsIntersecting,
      py::arg("p1"), py::arg("p2"), py::arg("boxes_min"), py::arg("boxes_max"),
      "Check if a line segment intersects with multiple axis-aligned "
      "bounding boxes "
      "(AABBs).\n\n"
      "Args:\n"
      "    p1 (numpy.ndarray): (dim, N) First endpoints of the line segments.\n"
      "    p2 (numpy.ndarray): (dim, N) Second endpoints of the line "
      "segments.\n"
      "    boxes_min (numpy.ndarray): Minimum corners of the AABBs as a matrix "
      "of shape (dim, N).\n"
      "    boxes_max (numpy.ndarray): Maximum corners of the AABBs as a matrix "
      "of shape (dim, N).\n\n"
      "Returns:\n"
      "    list: A list of boolean values, where True indicates the line "
      "segment intersects\n"
      "          with the corresponding AABB, False otherwise.\n\n"
      "Note:\n"
      "    The dimension of p1 and p2 must match the number of rows in "
      "boxes_min and boxes_max.\n"
      "    box_min and box_max must have the same shape.");

  m.def(
      "DistanceLinesegmentAABB",
      [](const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
         const Eigen::VectorXd& box_min, const Eigen::VectorXd& box_max,
         const int maxit, const double tol) {
        // Call the C++ function
        auto result = csdecomp::DistanceLinesegmentAABB(p1, p2, box_min,
                                                        box_max, maxit, tol);

        // Convert the tuple to a Python dictionary for better usability
        py::dict output;
        output["projected_point"] = std::get<0>(result);  // p_proj
        output["optimal_point"] = std::get<1>(result);    // p_optimal
        output["distance"] = std::get<2>(result);         // distance

        return output;
      },
      py::arg("p1"), py::arg("p2"), py::arg("box_min"), py::arg("box_max"),
      py::arg("maxit") = 100, py::arg("tol") = 1e-9,
      "Calculate the closest point and distance between a line segment and an "
      "AABB.\n\n"
      "Args:\n"
      "    p1 (numpy.ndarray): First endpoint of the line segment.\n"
      "    p2 (numpy.ndarray): Second endpoint of the line segment.\n"
      "    box_min (numpy.ndarray): Minimum corner of the AABB.\n"
      "    box_max (numpy.ndarray): Maximum corner of the AABB.\n"
      "    maxit (int, optional): Maximum number of iterations for the golden "
      "section search. Default is 100.\n"
      "    tol (float, optional): Tolerance for convergence. Default is "
      "1e-9.\n\n"
      "Returns:\n"
      "    dict: A dictionary containing:\n"
      "        'projected_point': The closest point on the AABB to the line "
      "segment.\n"
      "        'optimal_point': The closest point on the line segment to the "
      "AABB.\n"
      "        'distance': The distance between these two points.\n\n");
}