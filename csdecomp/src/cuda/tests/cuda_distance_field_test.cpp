#include "cuda_distance_field.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

using namespace std::chrono;

GTEST_TEST(DistanceField, RandomVoxelGrid) {
  std::vector<Eigen::Vector3f> lines;

  std::ifstream f("csdecomp/tests/test_assets/random_grid.voxels.txt");

  float tx, ty, tz;

  while (f >> tx >> ty >> tz) {
    lines.emplace_back(tx, ty, tz);
  }

  Eigen::Matrix3Xf points(3, lines.size());

  for (std::size_t i = 0; i < lines.size(); ++i) {
    points.col(i) = lines[i];
  }

  const auto origin = Eigen::Vector3f{-0.75, -0.75, 0.0};
  const float voxelSize = 0.02f;
  const int gridSize = 128;

  DistanceField field(origin, gridSize, voxelSize);

  field.setOccupancy(points);

  auto start = high_resolution_clock::now();
  field.computeEDT();
  auto stop = high_resolution_clock::now();

  std::cout << "execution time: "
            << duration_cast<milliseconds>(stop - start).count() << " ms\n";

  const int skip = 4;

  Eigen::Matrix3Xf testPoints(
      3, (gridSize / skip) * (gridSize / skip) * (gridSize / skip));

  std::vector<float> actual_dists;
  actual_dists.reserve(testPoints.cols());

  for (int i = 0; i < gridSize; i += skip) {
    for (int j = 0; j < gridSize; j += skip) {
      for (int k = 0; k < gridSize; k += skip) {
        auto coord = Eigen::Vector3i{i, j, k};
        Eigen::Vector3f P_W = coord.cast<float>() * voxelSize + origin;

        testPoints.col(actual_dists.size()) = P_W;

        // Scan through every inputPoint (i.e occupied voxel) and find the
        // nearest one to this voxel
        actual_dists.push_back(
            (points.colwise() - P_W).colwise().norm().minCoeff());
      }
    }
  }

  std::vector<float> expected_dists = field.getDistance(testPoints);

  for (std::size_t i = 0; i < actual_dists.size(); ++i) {
    // Assert that the nearest voxel has the same distance as the one
    // listed in the output voronoi map
    ASSERT_NEAR(expected_dists[i], actual_dists[i], 1e-3);
  }
}
