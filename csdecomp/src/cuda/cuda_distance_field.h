/*
Author: Cao Thanh Tung, Zheng Jiaqi
Date: 21/01/2010, 25/08/2019

File Name: pba3D.h

===============================================================================

Copyright (c) 2019, School of Computing, National University of Singapore.
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#pragma once

#include <cuda_runtime.h>

#include <Eigen/Geometry>
#include <vector>

#define MARKER -2147483648
#define MAX_INT 2147483647

// Sites 	 : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODE(x, y, z, a, b) \
  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))
#define DECODE(value, x, y, z) \
  x = ((value) >> 20) & 0x3ff; \
  y = ((value) >> 10) & 0x3ff; \
  z = (value)&0x3ff

#define TOID(x, y, z, w) ((z) * (w) * (w) + (y) * (w) + (x))

#define NOTSITE(value) (((value) >> 31) & 1)
#define HASNEXT(value) (((value) >> 30) & 1)

#define GET_X(value) (((value) >> 20) & 0x3ff)
#define GET_Y(value) (((value) >> 10) & 0x3ff)
#define GET_Z(value) ((NOTSITE((value))) ? MAX_INT : ((value)&0x3ff))

__device__ constexpr int ravelIdx(const int3 coord, const int gridSize) {
  return (coord.z * gridSize + coord.y) * gridSize + coord.x;
}

__device__ constexpr int encode(const int3 coord) {
  return (coord.x << 20) | (coord.y << 10) | coord.z;
}

__device__ constexpr int3 decode(const int id) {
  return int3{(id >> 20) & 0x3ff, (id >> 10) & 0x3ff, id & 0x3ff};
}

__device__ constexpr float3 operator-(const float3& A, const float3& B) {
  return float3{A.x - B.x, A.y - B.y, A.z - B.z};
}

struct VoronoiGrid {
  int* grid;
  const float3 origin;
  const float voxelSize;
  const int gridSize;

  __device__ constexpr float getDistance(const float3& P_W) const {
    const float3 P_G = P_W - origin;

    const int3 coord = int3{static_cast<int>(round(P_G.x / voxelSize)),
                            static_cast<int>(round(P_G.y / voxelSize)),
                            static_cast<int>(round(P_G.z / voxelSize))};

    if (coord.x < 0 || coord.y < 0 || coord.z < 0 || coord.x >= gridSize ||
        coord.y >= gridSize || coord.z >= gridSize) {
      return 0.0f;
    }

    const int3 nearest = decode(grid[ravelIdx(coord, gridSize)]);

    const int3 delta =
        int3{coord.x - nearest.x, coord.y - nearest.y, coord.z - nearest.z};

    return sqrtf((delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)) *
           voxelSize;
  }
};

struct OccupancyGrid {
  int* grid;
  const float3 origin;
  const float voxelSize;
  const int gridSize;

  __device__ void setOccupied(const float3& P_W) const {
    const float3 P_G = P_W - origin;

    const int3 coord = int3{static_cast<int>(round(P_G.x / voxelSize)),
                            static_cast<int>(round(P_G.y / voxelSize)),
                            static_cast<int>(round(P_G.z / voxelSize))};

    if (coord.x < 0 || coord.y < 0 || coord.z < 0 || coord.x >= gridSize ||
        coord.y >= gridSize || coord.z >= gridSize) {
      return;
    }

    grid[ravelIdx(coord, gridSize)] = encode(coord);
  }
};

class DistanceField {
  void colorZAxis(int m1);

  void computeProximatePointsYAxis(int m2);

  // Phase 3 of PBA. m3 must divides texture size
  // This method color along the Y axis
  void colorYAxis(int m3);

 public:
  OccupancyGrid occupancy;
  VoronoiGrid voronoi;

  const int gridSize;
  const size_t numVoxels;

  // Initialize necessary memory for 3D Voronoi Diagram computation
  // - textureSize: The size of the Discrete Voronoi Diagram (width = height)
  DistanceField(const Eigen::Vector3f& origin, std::size_t gridSize,
                const float voxelSize);

  // Deallocate all allocated memory
  virtual ~DistanceField();

  // Compute 3D Voronoi diagram
  // Input: a 3D texture. Each pixel is an integer encoding 3 coordinates.
  // 		For each site at (x, y, z), the pixel at coordinate (x, y, z)
  // should contain 		the encoded coordinate (x, y, z). Pixels that
  // are not sites should contain 		the integer MARKER. Use ENCODE
  // (and DECODE) macro to encode (and decode). See our website for the effect
  // of the three parameters: 		phase1Band, phase2Band, phase3Band
  // Parameters must divide textureSize
  void computeEDT(int phase1Band = 1, int phase2Band = 1, int phase3Band = 2);

  void setOccupancy(const Eigen::Matrix3Xf& P_VG);

  void resetOccupancy();

  std::vector<float> getDistance(const Eigen::Matrix3Xf& P_VG);
};
