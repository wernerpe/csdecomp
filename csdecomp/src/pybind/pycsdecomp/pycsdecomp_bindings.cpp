#include <pybind11/pybind11.h>

#include "collision_checker_bindings.h"
#include "cpp_utils_bindings.h"
#include "cuda_bindings.h"
#include "drm_bindings.h"
#include "hpolyhedron_bindings.h"
#include "plant_bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(pycsdecomp_bindings, m) {
  m.doc() = R"pbdoc(
    PyCSDecomp Python Bindings
    --------------------------
    
    This module provides Python bindings for the CSDecomp library.

  )pbdoc";
  add_plant_bindings(m);
  add_hpolyhedron_bindings(m);
  add_collision_checker_bindings(m);
  add_cuda_bindings(m);
  add_drm_bindings(m);
  add_cpp_utils_bindings(m);
}