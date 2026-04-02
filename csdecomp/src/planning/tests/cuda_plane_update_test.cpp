#include "planning/cuda_plane_update.h"

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include "plant/cuda_collision_checker.h"
#include "plant/urdf_parser.h"

using namespace csdecomp;

// 2D environment: movable point sphere + 4 fixed boxes in corners
const std::string boxes_in_corners_urdf = R"(
<robot name="boxes">
  <link name="fixed">
    <collision name="top_left">
      <origin rpy="0 0 0" xyz="-1 1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="top_right">
      <origin rpy="0 0 0" xyz="1 1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="bottom_left">
      <origin rpy="0 0 0" xyz="-1 -1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="bottom_right">
      <origin rpy="0 0 0" xyz="1 -1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="movable">
    <collision name="sphere">
      <geometry><sphere radius="0.01"/></geometry>
    </collision>
  </link>
  <link name="for_joint"/>
  <joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="for_joint"/>
  </joint>
  <joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="for_joint"/>
    <child link="movable"/>
  </joint>
</robot>
)";

GTEST_TEST(PlaneUpdateTest, BasicSeparation) {
  URDFParser parser;
  parser.parseURDFString(boxes_in_corners_urdf);
  MinimalPlant plant = parser.getMinimalPlant();
  SceneInspector inspector = parser.getSceneInspector();

  // Feasible point at origin, collision points inside the 4 corner boxes
  Eigen::MatrixXf feasible(2, 4);
  Eigen::MatrixXf collision(2, 4);
  // clang-format off
  feasible << 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0;
  collision << -1.0,  1.0,  1.0, -1.0,
                1.0,  1.0, -1.0, -1.0;
  // clang-format on

  // Verify that feasible points are indeed free and collision points collide
  auto feas_results = checkCollisionFreeCuda(&feasible, &plant);
  auto col_results = checkCollisionFreeCuda(&collision, &plant);
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(feas_results[i])
        << "Feasible point " << i << " is in collision";
    ASSERT_FALSE(col_results[i]) << "Collision point " << i << " is free";
  }

  PlaneUpdateOptions options;
  options.bisection_steps = 10;
  options.configuration_margin = 0.01;

  Voxels vox(3, 0);  // no voxels
  auto result = ComputeSeparatingPlanesCuda(feasible, collision, plant,
                                            inspector.robot_geometry_ids, vox,
                                            0.0f, options);

  ASSERT_EQ(result.a.cols(), 4);
  ASSERT_EQ(result.b.size(), 4);
  ASSERT_EQ(result.boundary_points.cols(), 4);
  ASSERT_EQ(result.boundary_dists.size(), 4);

  // Each plane should separate: a^T q_feas + b < 0 (feasible side)
  for (int i = 0; i < 4; ++i) {
    float val_feas = result.a.col(i).dot(feasible.col(i)) + result.b(i);
    float val_col = result.a.col(i).dot(collision.col(i)) + result.b(i);
    EXPECT_LT(val_feas, 0.0f)
        << "Plane " << i << " does not separate feasible point";
    EXPECT_GT(val_col, 0.0f)
        << "Plane " << i << " does not separate collision point";
  }

  // Normals should be unit vectors
  for (int i = 0; i < 4; ++i) {
    float norm = result.a.col(i).norm();
    EXPECT_NEAR(norm, 1.0f, 1e-5) << "Plane " << i << " normal not unit";
  }

  // Boundary points should be near the actual obstacle boundary
  for (int i = 0; i < 4; ++i) {
    EXPECT_GT(result.boundary_dists(i), 0.1f)
        << "Boundary dist " << i << " suspiciously small";
    EXPECT_LT(result.boundary_dists(i), 1.5f)
        << "Boundary dist " << i << " suspiciously large";
  }
}

GTEST_TEST(PlaneUpdateTest, WithVoxels) {
  URDFParser parser;
  parser.parseURDFString(boxes_in_corners_urdf);
  MinimalPlant plant = parser.getMinimalPlant();
  SceneInspector inspector = parser.getSceneInspector();

  // Place a voxel obstacle at (0, 0.5) — directly between feasible and a
  // collision point
  Voxels vox(3, 1);
  vox << 0.0, 0.5, 0.0;
  float voxel_radius = 0.2;

  Eigen::MatrixXf feasible(2, 1);
  Eigen::MatrixXf collision(2, 1);
  feasible << 0.0, 0.0;
  collision << 0.0, 1.0;  // would be free without voxel, collides with voxel

  PlaneUpdateOptions options;
  options.bisection_steps = 10;
  options.configuration_margin = 0.01;

  auto result = ComputeSeparatingPlanesCuda(feasible, collision, plant,
                                            inspector.robot_geometry_ids, vox,
                                            voxel_radius, options);

  ASSERT_EQ(result.a.cols(), 1);

  // The plane should separate the feasible from the collision
  float val_feas = result.a.col(0).dot(feasible.col(0)) + result.b(0);
  float val_col = result.a.col(0).dot(collision.col(0)) + result.b(0);
  EXPECT_LT(val_feas, 0.0f) << "Plane does not separate feasible point";
  EXPECT_GT(val_col, 0.0f) << "Plane does not separate collision point";

  // Boundary should be closer than without voxel (voxel at 0.5 with radius 0.2)
  EXPECT_LT(result.boundary_dists(0), 0.5f)
      << "Boundary should be close due to voxel";
}

GTEST_TEST(PlaneUpdateTest, Timing) {
  URDFParser parser;
  parser.parseURDFString(boxes_in_corners_urdf);
  MinimalPlant plant = parser.getMinimalPlant();
  SceneInspector inspector = parser.getSceneInspector();

  const int M = 1000;
  Eigen::MatrixXf feasible = Eigen::MatrixXf::Zero(2, M);
  Eigen::MatrixXf collision(2, M);

  for (int i = 0; i < M; ++i) {
    float angle = 2.0f * M_PI * i / M;
    collision(0, i) = 1.2f * cos(angle);
    collision(1, i) = 1.2f * sin(angle);
  }

  PlaneUpdateOptions options;
  options.bisection_steps = 10;
  options.configuration_margin = 0.01;
  Voxels vox(3, 0);

  // Warmup
  ComputeSeparatingPlanesCuda(feasible, collision, plant,
                              inspector.robot_geometry_ids, vox, 0.0f, options);

  auto start = std::chrono::high_resolution_clock::now();
  const int N_RUNS = 100;
  for (int r = 0; r < N_RUNS; ++r) {
    ComputeSeparatingPlanesCuda(feasible, collision, plant,
                                inspector.robot_geometry_ids, vox, 0.0f,
                                options);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(end - start).count() / N_RUNS;
  std::cout << "[Timing] ComputeSeparatingPlanesCuda: " << ms << " ms (" << M
            << " points, " << options.bisection_steps << " bisection steps)"
            << std::endl;

  EXPECT_LT(ms, 50.0) << "Too slow";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
