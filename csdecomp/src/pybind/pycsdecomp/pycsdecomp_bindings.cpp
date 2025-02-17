#include <pybind11/pybind11.h>

#include "collision_checker_bindings.h"
#include "cuda_bindings.h"
#include "drm_bindings.h"
#include "hpolyhedron_bindings.h"
#include "plant_bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(pycsdecomp_bindings, m) {
  add_plant_bindings(m);
  add_hpolyhedron_bindings(m);
  add_collision_checker_bindings(m);
  add_cuda_bindings(m);
  add_drm_bindings(m);
}