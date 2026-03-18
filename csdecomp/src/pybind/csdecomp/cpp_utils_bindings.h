#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geometry/distance_aabb_linesegment.h"
#include "geometry/linesegment_aabb_checker.h"
#include "planning/bezier_curve.h"

namespace py = pybind11;
using namespace csdecomp;

void add_cpp_utils_bindings(py::module& m) {
  // Add the LinesegmentAABBIntersecting function for VectorXd
  m.def("LinesegmentAABBIntersecting", &LinesegmentAABBIntersecting,
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
      "LinesegmentAABBsIntersecting", &LinesegmentAABBsIntersecting,
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
      "PwlPathAABBsIntersecting", &PwlPathAABBsIntersecting, py::arg("p1"),
      py::arg("p2"), py::arg("boxes_min"), py::arg("boxes_max"),
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
      "DistanceLinesegmentAABB", &DistanceLinesegmentAABB, py::arg("p1"),
      py::arg("p2"), py::arg("box_min"), py::arg("box_max"),
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

  // Add the PointsInAABBs function
  m.def(
      "PointsInAABBs", &PointsInAABBs, py::arg("points"), py::arg("boxes_min"),
      py::arg("boxes_max"), py::arg("parallelize") = false,
      "Check if multiple points are inside any of the axis-aligned bounding "
      "boxes "
      "(AABBs).\n\n"
      "Args:\n"
      "    points (numpy.ndarray): Points as a matrix of shape (dim, "
      "N_points).\n"
      "    boxes_min (numpy.ndarray): Minimum corners of the AABBs as a matrix "
      "of shape (dim, N_boxes).\n"
      "    boxes_max (numpy.ndarray): Maximum corners of the AABBs as a matrix "
      "of shape (dim, N_boxes).\n"
      "    parallelize (bool, optional): Whether to use OpenMP "
      "parallelization. "
      "Default is True.\n\n"
      "Returns:\n"
      "    list: A list of integers (0 or 1), where 1 indicates the "
      "corresponding "
      "point\n"
      "          is inside at least one AABB, 0 otherwise.\n\n"
      "Note:\n"
      "    Each point is checked against all AABBs with early exit on first "
      "collision.\n"
      "    When parallelize=True, computation is distributed across available "
      "threads.\n"
      "    The dimension of points must match the number of rows in "
      "boxes_min and boxes_max.\n"
      "    boxes_min and boxes_max must have the same shape.");

  // Bind BezierCurve class
  py::class_<csdecomp::BezierCurve>(m, "BezierCurve")
      // Constructor
      .def(py::init<const Eigen::MatrixXd&, double, double>(),
           "Create a Bezier curve", py::arg("points"),
           py::arg("initial_time") = 0.0, py::arg("final_time") = 1.0)

      // Copy constructor
      .def(py::init<const csdecomp::BezierCurve&>())

      // Properties
      .def_property_readonly("degree", &csdecomp::BezierCurve::degree,
                             "Get the degree of the curve")
      .def_property_readonly("dimension", &csdecomp::BezierCurve::dimension,
                             "Get the dimension of the curve")
      .def_property_readonly("initial_time",
                             &csdecomp::BezierCurve::initial_time,
                             "Get the initial time")
      .def_property_readonly("final_time", &csdecomp::BezierCurve::final_time,
                             "Get the final time")
      .def_property_readonly("duration", &csdecomp::BezierCurve::duration,
                             "Get the duration")
      .def_property_readonly("points", &csdecomp::BezierCurve::points,
                             "Get the control points",
                             py::return_value_policy::reference_internal)

      // Shape property for Python compatibility
      .def_property_readonly("shape", &csdecomp::BezierCurve::shape,
                             "Get shape as (degree+1, dimension) tuple")

      // Methods
      .def("initial_point", &csdecomp::BezierCurve::initial_point,
           "Get the initial point of the curve")
      .def("final_point", &csdecomp::BezierCurve::final_point,
           "Get the final point of the curve")

      .def("bernstein", &csdecomp::BezierCurve::bernstein,
           "Compute Bernstein basis function value", py::arg("time"),
           py::arg("n"))

      // Python compatibility for Bernstein (with underscore)
      .def("_berstein", &csdecomp::BezierCurve::_berstein,
           "Compute Bernstein basis function value (Python compatibility)",
           py::arg("time"), py::arg("n"))

      // Call operator for evaluation
      .def(
          "__call__",
          static_cast<Eigen::VectorXd (csdecomp::BezierCurve::*)(double) const>(
              &csdecomp::BezierCurve::operator()),
          "Evaluate the curve at a given time", py::arg("time"))

      // Vectorized evaluation
      .def("evaluate_at_times", &csdecomp::BezierCurve::evaluate_at_times,
           "Evaluate the curve at multiple time points", py::arg("times"))

      // Support for numpy array evaluation (Python compatibility)
      .def(
          "__call__",
          [](const csdecomp::BezierCurve& self,
             const std::vector<double>& times) {
            Eigen::MatrixXd result = self.evaluate_at_times(times);
            // Return transposed to match Python convention (times x dimension)
            return result.transpose();
          },
          "Evaluate the curve at multiple time points (numpy array input)",
          py::arg("times"))

      // Arithmetic operators
      .def("__mul__",
           static_cast<csdecomp::BezierCurve (csdecomp::BezierCurve::*)(
               const csdecomp::BezierCurve&) const>(
               &csdecomp::BezierCurve::operator*),
           "Multiply two Bezier curves", py::arg("other"))
      .def("__mul__",
           static_cast<csdecomp::BezierCurve (csdecomp::BezierCurve::*)(double)
                           const>(&csdecomp::BezierCurve::operator*),
           "Multiply curve by scalar", py::arg("scalar"))
      .def(
          "__rmul__",
          [](const csdecomp::BezierCurve& self, double scalar) {
            return scalar * self;
          },
          "Multiply scalar by curve", py::arg("scalar"))
      .def("__imul__", &csdecomp::BezierCurve::operator*=,
           "In-place multiplication by scalar", py::arg("scalar"))

      .def("__add__",
           static_cast<csdecomp::BezierCurve (csdecomp::BezierCurve::*)(
               const csdecomp::BezierCurve&) const>(
               &csdecomp::BezierCurve::operator+),
           "Add two Bezier curves", py::arg("other"))
      .def("__add__",
           static_cast<csdecomp::BezierCurve (csdecomp::BezierCurve::*)(double)
                           const>(&csdecomp::BezierCurve::operator+),
           "Add scalar to curve", py::arg("scalar"))
      .def(
          "__radd__",
          [](const csdecomp::BezierCurve& self, double scalar) {
            return scalar + self;
          },
          "Add curve to scalar", py::arg("scalar"))
      .def("__iadd__", &csdecomp::BezierCurve::operator+=, "In-place addition",
           py::arg("other"))

      .def("__sub__",
           static_cast<csdecomp::BezierCurve (csdecomp::BezierCurve::*)(
               const csdecomp::BezierCurve&) const>(
               &csdecomp::BezierCurve::operator-),
           "Subtract two Bezier curves", py::arg("other"))
      .def("__sub__",
           static_cast<csdecomp::BezierCurve (csdecomp::BezierCurve::*)(double)
                           const>(&csdecomp::BezierCurve::operator-),
           "Subtract scalar from curve", py::arg("scalar"))
      //   .def(
      //       "__rsub__",
      //       [](const csdecomp::BezierCurve& self, double scalar) {
      //         return scalar - self;
      //       },
      //       "Subtract curve from scalar", py::arg("scalar"))
      //   .def("__isub__", &csdecomp::BezierCurve::operator-=,
      //        "In-place subtraction", py::arg("other"))

      .def(
          "__neg__", [](const csdecomp::BezierCurve& self) { return -self; },
          "Unary minus operator")

      // Python compatibility for scalar product (@ operator equivalent)
      .def("__matmul__", &csdecomp::BezierCurve::scalar_product,
           "Scalar product of two Bezier curves", py::arg("other"))

      //   // Conversion methods
      //   .def(
      //       "_number_to_curve",
      //       [](const csdecomp::BezierCurve& self, double scalar) {
      //         Eigen::MatrixXd points(1, 1);
      //         points(0, 0) = scalar;
      //         return csdecomp::BezierCurve(points, self.initial_time(),
      //                                      self.final_time());
      //       },
      //       "Convert number to constant curve (Python compatibility)",
      //       py::arg("scalar"))

      // Curve operations
      .def("elevate_degree", &csdecomp::BezierCurve::elevate_degree,
           "Elevate the degree of the curve", py::arg("new_degree"))

      .def("derivative", &csdecomp::BezierCurve::derivative,
           "Compute the derivative of the curve")

      .def("integral", &csdecomp::BezierCurve::integral,
           "Compute the integral of the curve",
           py::arg("initial_condition") = Eigen::VectorXd())

      .def("domain_split", &csdecomp::BezierCurve::domain_split,
           "Split the curve at a given time", py::arg("split_time"))

      .def("shift_domain", &csdecomp::BezierCurve::shift_domain,
           "Shift the time domain of the curve", py::arg("time_shift"))

      // Norm calculations
      .def("l2_squared", &csdecomp::BezierCurve::l2_squared,
           "Compute the L2 squared norm of the curve")

      .def("squared_l2_norm", &csdecomp::BezierCurve::squared_l2_norm,
           "Compute the squared L2 norm of the curve")

      //   // Generic convex function integration
      //   .def(
      //       "integral_of_convex_function",
      //       [](const csdecomp::BezierCurve& self, py::function f) {
      //         return self.integral_of_convex_function(
      //             [f](const Eigen::VectorXd& point) -> double {
      //               return f(point).cast<double>();
      //             });
      //       },
      //       "Compute integral of a convex function over the curve",
      //       py::arg("f"))

      // Python compatibility methods
      .def("_assert_same_times", &csdecomp::BezierCurve::_check_same_times,
           "Check if curves have compatible time domains", py::arg("other"))

      // String representation
      .def("__repr__", [](const csdecomp::BezierCurve& curve) {
        return "<BezierCurve degree=" + std::to_string(curve.degree()) +
               " dimension=" + std::to_string(curve.dimension()) + " time=[" +
               std::to_string(curve.initial_time()) + ", " +
               std::to_string(curve.final_time()) + "]>";
      });

  py::class_<csdecomp::CompositeBezierCurve>(
      m, "CompositeBezierCurve",
      "Composite Bezier curve implementation - represents a piecewise Bezier "
      "curve composed of multiple connected segments")
      // Constructor
      .def(py::init<const std::vector<csdecomp::BezierCurve>&>(),
           "Constructor from vector of BezierCurve segments", py::arg("curves"))

      // Getters
      .def("dimension", &csdecomp::CompositeBezierCurve::dimension,
           "Get the dimension of the curve")
      .def("initial_time", &csdecomp::CompositeBezierCurve::initial_time,
           "Get the initial time parameter")
      .def("final_time", &csdecomp::CompositeBezierCurve::final_time,
           "Get the final time parameter")
      .def("duration", &csdecomp::CompositeBezierCurve::duration,
           "Get the total duration of the curve")
      .def("num_segments", &csdecomp::CompositeBezierCurve::num_segments,
           "Get the number of curve segments")
      .def("knot_times", &csdecomp::CompositeBezierCurve::knot_times,
           "Get the knot times vector",
           py::return_value_policy::reference_internal)
      .def("curves", &csdecomp::CompositeBezierCurve::curves,
           "Get the vector of BezierCurve segments",
           py::return_value_policy::reference_internal)

      // Methods
      .def("curve_segment", &csdecomp::CompositeBezierCurve::curve_segment,
           "Find which curve segment contains the given time", py::arg("time"))

      // Evaluation operators
      .def("__call__", &csdecomp::CompositeBezierCurve::operator(),
           "Evaluate the composite curve at a given time", py::arg("time"))
      .def(
          "evaluate_at_times",
          &csdecomp::CompositeBezierCurve::evaluate_at_times,
          "Vectorized evaluation at multiple time points. Returns matrix where "
          "each column is a point on the curve (shape: dimension x num_times)",
          py::arg("times"))
      .def("curve_segments", &csdecomp::CompositeBezierCurve::curve_segments,
           "Get segment indices for multiple time points", py::arg("times"))

      // Index operator
      .def("__getitem__", &csdecomp::CompositeBezierCurve::operator[],
           "Access individual curve segments", py::arg("index"),
           py::return_value_policy::reference_internal)

      // Optional: Add __len__ for Python-like behavior
      .def("__len__", &csdecomp::CompositeBezierCurve::num_segments,
           "Get the number of curve segments (Python len() support)");

  // Bind collision detection function
  m.def("BezierCurveHPolyhedronCollisionFree",
        &csdecomp::BezierCurveHPolyhedronCollisionFree,
        "Check if Bezier curve is collision-free with H-polyhedron",
        py::arg("curve"), py::arg("A"), py::arg("b"), py::arg("tol") = 1e-2);

  m.def("IntersectCompositeBezierCurveWithHPolyhedra",
        &csdecomp::IntersectCompositeBezierCurveWithHPolyhedra,
        R"doc(
      Find intersections between composite Bezier curve segments and H-polyhedron obstacles.

      For each segment of the composite Bezier curve, determines which H-polyhedron
      obstacles it potentially intersects with. Uses the BezierCurveHPolyhedronCollisionFree 
      function to check collision status - segments that are NOT collision-free are 
      considered to intersect.

      Parameters
      ----------
      c : CompositeBezierCurve
          The composite Bezier curve to analyze
      As : list of numpy.ndarray
          Vector of constraint matrices, one for each H-polyhedron 
          (each A_i has shape num_constraints_i x dimension)
      bs : list of numpy.ndarray  
          Vector of constraint vectors, one for each H-polyhedron
          (each b_i has shape num_constraints_i x 1)
      hpoly_to_ignore : dict
          Dictionary mapping segment indices to lists of obstacle indices to ignore.
          If an obstacle index is present in the list for a segment, that obstacle
          is not checked for collision with that segment.
      tol : float, optional
          Tolerance for collision detection algorithm (default: 1e-2)
      parallelize : bool, optional
          Whether to use parallel processing over segments (default: True)

      Returns
      -------
      dict
          A dictionary where result[seg_idx] contains the indices of all H-polyhedrons
          that segment seg_idx intersects with. Segments with no collisions are not
          included in the dictionary.

      Notes
      -----
      - An intersection is detected when BezierCurveHPolyhedronCollisionFree returns 
        false, meaning the segment is not proven to be collision-free.
      - Parallelizes over curve segments (recommended when num_obstacles >> num_segments).
      
      Preconditions
      -------------
      - len(As) must equal len(bs)
      - For each i: As[i].shape[1] must equal c.dimension()
      - For each i: As[i].shape[0] must equal bs[i].size
      - tol must be positive
      )doc",
        py::arg("c"), py::arg("As"), py::arg("bs"), py::arg("hpoly_to_ignore"),
        py::arg("tol") = 1e-2, py::arg("parallelize") = true);
}