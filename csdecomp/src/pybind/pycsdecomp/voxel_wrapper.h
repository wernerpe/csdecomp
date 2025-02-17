#pragma once
#include "collision_geometry.h"
using namespace csdecomp;
namespace {
class VoxelsWrapper {
 public:
  Voxels matrix;
  VoxelsWrapper(){};
  VoxelsWrapper(Voxels vox) { matrix = vox; }

  void setMatrix(const Voxels& m) { matrix = m; }

  Voxels getMatrix() const { return matrix; }
};
}  // namespace