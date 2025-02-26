
#include "cuda_hit_and_run_sampling.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <matplotlibcpp.h>

#include <chrono>
#include <iostream>
#include <random>

#include "hpolyhedron.h"
#include "plotting_utils.h"

namespace plt = matplotlibcpp;
using namespace PlottingUtils;
using namespace csdecomp;

class HNRSampleTest : public ::testing::Test {
 protected:
  HPolyhedron cube, cube2d, poly, poly2, skinnypoly;

  void SetUp() override {
    Eigen::VectorXf lower(3);
    lower << -1, -1, -1;
    Eigen::VectorXf upper(3);
    upper << 1, 1, 1;
    cube.MakeBox(lower, upper);
    cube2d.MakeBox(lower.head(2), upper.head(2));

    Eigen::MatrixXf A(6, 2);
    Eigen::VectorXf b(6);
    // clang-format off
      A << 0,  1,  // Top face
           1,  1,   // Upper-right face
           1, -1,  // Lower-right face
           0, -1,  // Bottom face
          -1,  0,  // Left face
          -1,  1;  // Upper-left face
  
    // Define the b vector (6x1)
      b << 2,   // Top face
           3,   // Upper-right face
           3,   // Lower-right face
           2,   // Bottom face
           2,   // Left face
           2;   // Upper-left face
    // clang-format on
    Eigen::MatrixXf A2(4, 2);
    Eigen::VectorXf b2(4);
    // clang-format off
      A2 << -1,  0,  // Left face
           0,  -1,  // Low face
           1,   1,  // Upper-right face
           -2,  1;  // Upper-left face
  
    // Define the b vector (6x1)
      b2 << -5,   // Left face
            -2,   // Low face 
            16,   // Upper-right face
            -6;   // Upper-left face
    // clang-format on
    // https://www.desmos.com/cahttps://www.desmos.com/calculator/v8wzjxpplzlculator/v8wzjxpplz
    Eigen::MatrixXf A3(4, 2);
    Eigen::VectorXf b3(4);
    // clang-format off
      A3 <<  0,   -1,  // Left face
             1,    1,  // Low face
             1, -0.5,  // Upper-right face
            -2,    1;  // Upper-left face
  
    // Define the b vector 
      b3 << -2,   // Left face
            12,   // Low face 
           3.15,   // Upper-right face
            -6;   // Upper-left face
    // clang-format on
    Eigen::VectorXf shift(2);
    shift << 10, 3;
    b = b + A * shift;
    //   std::cout<<b<<std::endl;
    poly = HPolyhedron(A, b);
    poly2 = HPolyhedron(A2, b2);
    skinnypoly = HPolyhedron(A3, b3);
  }
};

TEST_F(HNRSampleTest, TestCudaUniformSample) {
  int num_samples = 3;
  std::vector<HPolyhedron> poly_vec;
  poly_vec.push_back(poly);
  poly_vec.push_back(poly);
  poly_vec.push_back(poly);
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  int idx = 0;
  for (auto p : poly_vec) {
    interior_points.col(idx) = p.ChebyshevCenter();
    ++idx;
  }
  std::cout << poly_vec.at(0).A() << std::endl;
  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_samples,
                                     10);
  idx = 0;
  for (auto s : samples_poly) {
    for (int col = 0; col < s.cols(); ++col) {
      EXPECT_TRUE(poly_vec.at(idx).PointInSet(samples_poly.at(idx).col(col)));
    }
    std::cout << fmt::format("samples of poly {} \n", idx);
    std::cout << samples_poly.at(idx) << std::endl;
    ++idx;
  }
}

TEST_F(HNRSampleTest, TestCudaManyUniformSample) {
  int num_samples = 1000;
  std::vector<HPolyhedron> poly_vec;
  poly_vec.push_back(poly);
  poly_vec.push_back(poly2);
  poly_vec.push_back(poly);
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  int idx = 0;
  for (auto p : poly_vec) {
    interior_points.col(idx) = p.ChebyshevCenter();
    ++idx;
  }
  std::cout << poly_vec.at(0).A() << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_samples,
                                     20);
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << fmt::format(
                   "execution time cuda: {} ms",
                   std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                         start)
                       .count())
            << std::endl;
  idx = 0;
  for (auto s : samples_poly) {
    for (int col = 0; col < num_samples; ++col) {
      EXPECT_TRUE(poly_vec.at(idx).PointInSet(samples_poly.at(idx).col(col)));
    }
    EXPECT_EQ(s.cols(), num_samples);
    EXPECT_FALSE(s.col(num_samples - 1).isApprox(samples_poly.at(0).col(0)));

    ++idx;
  }
  std::cout << samples_poly.at(0).block(0, 0, 2, 5);
  // check that samples are diverse
  EXPECT_FALSE(samples_poly.at(0).col(0).isApprox(samples_poly.at(2).col(0)));
}

TEST_F(HNRSampleTest, PlotCudaSamples) {
  int num_samples = 100000;
  std::vector<HPolyhedron> poly_vec;
  poly_vec.push_back(poly);
  poly_vec.push_back(poly2);
  poly_vec.push_back(skinnypoly);
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  int idx = 0;
  for (auto p : poly_vec) {
    interior_points.col(idx) = p.ChebyshevCenter();
    ++idx;
  }
  auto start = std::chrono::high_resolution_clock::now();

  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_samples,
                                     50);
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << fmt::format(
                   "execution time cuda: {} ms",
                   std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                         start)
                       .count())
            << std::endl;
  idx = 0;
  Eigen::Vector2f adversarial_point{5.75, 5.5};
  bool found_sample_close_to_adversarial_point = false;
  for (auto s : samples_poly) {
    for (int ii = 0; ii < num_samples; ++ii) {
      assert(poly_vec.at(idx).PointInSet(s.col(ii)));
      if (idx == 2) {
        float dist = (s.col(ii) - adversarial_point).norm();
        if (dist < 0.05) {
          found_sample_close_to_adversarial_point = true;
        }
      }
    }
    ++idx;
  }
  EXPECT_TRUE(found_sample_close_to_adversarial_point);

  if (true || std::getenv("BAZEL_TEST") == nullptr) {
    idx = 0;
    for (auto s : samples_poly) {
      setupTestPlottingEnv();
      plt::figure_size(1000, 1000);

      plot2DHPolyhedron(poly_vec.at(idx), "r", 1000);

      // disgustang but it works
      Eigen::VectorXd row = s.row(0).cast<double>();
      std::vector<double> samp_x(row.data(), row.data() + num_samples);
      row = s.row(1).cast<double>();
      std::vector<double> samp_y(row.data(), row.data() + num_samples);
      plt::scatter(samp_x, samp_y, 1);
      cleanupPlotAndPauseForUser();
      std::cout << fmt::format(
          "All points are containted in Polytope {}. Note that the plotting "
          "of the polytope boundary is inaccurate and therefore some of the "
          "points on the plot can lie outside of the boundary!\n",
          idx);
      ++idx;
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}