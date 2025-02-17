#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hpolyhedron.h"

namespace py = pybind11;
using namespace csdecomp;

void add_hpolyhedron_bindings(py::module &m) {
  py::class_<MinimalHPolyhedron>(m, "MinimalHPolyhedron")
      .def(py::init<>())
      .def("A",
           [](const MinimalHPolyhedron &self) {
             Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                 Amin(self.A, self.num_faces, self.dim);
             // create copy and return
             Eigen::MatrixXf A = Amin;
             return A;
           })
      .def("b",
           [](const MinimalHPolyhedron &self) {
             Eigen::Map<const Eigen::VectorXf> bmin(self.b, self.num_faces);

             // create copy and return
             Eigen::VectorXf b = bmin;
             return b;
           })
      .def_property_readonly(
          "num_faces",
          [](const MinimalHPolyhedron &self) { return self.num_faces; })
      .def_property_readonly(
          "dim", [](const MinimalHPolyhedron &self) { return self.dim; });

  py::class_<HPolyhedron>(m, "HPolyhedron")
      .def(py::init<>())
      .def(py::init<const MinimalHPolyhedron &>())
      .def(py::init<const Eigen::MatrixXf &, Eigen::VectorXf &>(), py::arg("A"),
           py::arg("b"))
      .def("A", &HPolyhedron::A)
      .def("b", &HPolyhedron::b)
      .def("MakeBox", &HPolyhedron::MakeBox)
      .def("GetMyMinimalHPolyhedron", &HPolyhedron::GetMyMinimalHPolyhedron)
      .def("PointInSet", &HPolyhedron::PointInSet)
      .def("UniformSample", &HPolyhedron::UniformSample,
           py::arg("previous_sample"), py::arg("mixing_steps"))
      .def("GetFeasiblePoint", &HPolyhedron::GetFeasiblePoint)
      .def("ChebyshevCenter", &HPolyhedron::ChebyshevCenter)
      .def("ambient_dimension", &HPolyhedron::ambient_dimension)
      .def(
          "UniformSample",
          [](HPolyhedron &self, const Eigen::VectorXf &previous_sample,
             int mixing_steps) {
            return self.UniformSample(previous_sample, mixing_steps);
          },
          py::arg("previous_sample"), py::arg("mixing_steps"));
}