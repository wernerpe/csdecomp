
#include "hpolyhedron.h"

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>

using namespace csdecomp;

class HPolyhedronTest : public ::testing::Test {
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

TEST_F(HPolyhedronTest, Constructor) {
  Eigen::MatrixXf A(2, 2);
  A << 1, 0, 0, 1;
  Eigen::VectorXf b(2);
  b << 1, 1;

  HPolyhedron poly2(A, b);

  EXPECT_EQ(poly2.A(), A);
  EXPECT_EQ(poly2.b(), b);
  EXPECT_EQ(poly2.ambient_dimension(), 2);
}

TEST_F(HPolyhedronTest, MakeBox) {
  EXPECT_EQ(cube.A().rows(), 6);
  EXPECT_EQ(cube.A().cols(), 3);
  EXPECT_EQ(cube.b().size(), 6);
  EXPECT_EQ(cube.ambient_dimension(), 3);
}

TEST_F(HPolyhedronTest, GetMyMinimalHPolyhedron) {
  MinimalHPolyhedron minimal = poly.GetMyMinimalHPolyhedron();

  EXPECT_EQ(minimal.num_faces, 6);
  EXPECT_EQ(minimal.dim, 2);

  for (int j = 0; j < minimal.num_faces; ++j) {
    for (int i = 0; i < minimal.dim; ++i) {
      EXPECT_EQ(minimal.A[i + j * minimal.dim], poly.A()(j, i));
    }
    EXPECT_EQ(minimal.b[j], poly.b()(j));
  }
}

TEST_F(HPolyhedronTest, TestMinPolyHpolyConstructor) {
  MinimalHPolyhedron minimal = poly.GetMyMinimalHPolyhedron();
  HPolyhedron replica(minimal);
  EXPECT_EQ(replica.ambient_dimension(), poly.ambient_dimension());
  EXPECT_TRUE(replica.A().isApprox(poly.A()));
  EXPECT_TRUE(replica.b().isApprox(poly.b()));
}

TEST_F(HPolyhedronTest, UniformSample) {
  Eigen::VectorXf initial_point = Eigen::VectorXf::Zero(3);
  Eigen::VectorXf sample = cube.UniformSample(initial_point, 1000);

  EXPECT_EQ(sample.size(), 3);
  EXPECT_TRUE(cube.PointInSet(sample));

  Eigen::VectorXf initial_point2 = Eigen::VectorXf::Zero(2);
  EXPECT_THROW(poly.UniformSample(initial_point2, 100), std::invalid_argument);
  initial_point2 = poly.ChebyshevCenter();
  std::cout << initial_point2 << std::endl << std::endl;

  EXPECT_TRUE(poly.PointInSet(initial_point2));
  EXPECT_TRUE(poly.PointInSet(poly.UniformSample(initial_point2, 100)));
}

TEST_F(HPolyhedronTest, ChebyshevCenter) {
  Eigen::VectorXf center = poly.ChebyshevCenter();
  std::cout << "chebyshev center \n" << center << std::endl;
  EXPECT_TRUE(poly.PointInSet(center));
}

TEST_F(HPolyhedronTest, GetFeasiblePoint) {
  Eigen::VectorXf point = cube.GetFeasiblePoint();
  Eigen::VectorXf point2 = poly.GetFeasiblePoint();

  std::cout << point << std::endl;
  std::cout << point2 << std::endl;
  EXPECT_EQ(point.size(), 3);
  EXPECT_TRUE(cube.PointInSet(point));
  EXPECT_TRUE(poly.PointInSet(point2));
}

TEST_F(HPolyhedronTest, PointInSet) {
  // Test points inside the cube
  Eigen::VectorXf inside_point(3);
  inside_point << 0, 0, 0;
  EXPECT_TRUE(cube.PointInSet(inside_point));

  inside_point << 0.5, -0.5, 0.5;
  EXPECT_TRUE(cube.PointInSet(inside_point));

  // Test points on the boundary of the cube
  Eigen::VectorXf boundary_point(3);
  boundary_point << 1, 0, 0;
  EXPECT_TRUE(cube.PointInSet(boundary_point));

  boundary_point << -1, -1, -1;
  EXPECT_TRUE(cube.PointInSet(boundary_point));

  // Test points outside the cube
  Eigen::VectorXf outside_point(3);
  outside_point << 1.1, 0, 0;
  EXPECT_FALSE(cube.PointInSet(outside_point));

  outside_point << 0, -2, 0;
  EXPECT_FALSE(cube.PointInSet(outside_point));

  // Test point with incorrect dimension
  Eigen::VectorXf wrong_dim_point(2);
  wrong_dim_point << 0, 0;
  EXPECT_THROW(cube.PointInSet(wrong_dim_point), std::invalid_argument);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}