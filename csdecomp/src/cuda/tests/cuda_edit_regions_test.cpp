#include <fmt/core.h>
#include <gtest/gtest.h>

#include <chrono>
#include <string>

// #include "collision_checker.h"
// #include "cuda_collision_checker.h"
// #include "cuda_hit_and_run_sampling.h"
// #include "cuda_polytope_builder.h"
#include "cuda_edit_regions.h"
#include "hpolyhedron.h"
#include "minimal_plant.h"
#include "plotting_utils.h"
#include "urdf_parser.h"

namespace plt = matplotlibcpp;
using namespace PlottingUtils;
using namespace csdecomp;
#define DO_PLOT 1

/* A movable sphere with fixed boxes in all corners.
┌───────────────┐
│┌────┐   ┌────┐│
││    │   │    ││
│└────┘   └────┘│
│       o       │
│┌────┐   ┌────┐│
││    │   │    ││
│└────┘   └────┘│
└───────────────┘ */
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

GTEST_TEST(PolytopeBuilderTest, TWODEnvTestWithPlotting0) {
  URDFParser parser;
  parser.parseURDFString(boxes_in_corners_urdf);
  KinematicTree tree = parser.getKinematicTree();
  MinimalPlant plant = parser.getMinimalPlant();
  SceneInspector inspector = parser.getSceneInspector();
  HPolyhedron domain;
  domain.MakeBox(tree.getPositionLowerLimits(), tree.getPositionUpperLimits());

  EditRegionsOptions options;
  options.bisection_steps = 9;
  options.configuration_margin = 0.01;
  Voxels vox(3, 3);
  // clang-format off
    vox << -0.55, -1.50, 1.90,  
            2.02,  1.80, 1.90,
            0.00,  0.00, 0.00;
  // clang-format on
  float voxel_radius = 0.05;

  Eigen::MatrixXf line_start(2, 3);
  Eigen::MatrixXf line_end(2, 3);
  // clang-format off
  line_start << -1.90, 0.00, 0.00,
                 0.00, 0.00, 1.80;

  line_end  <<   0.00, 0.00, 1.90,
                 0.00, 1.80, 1.80;
  // clang-format on
  std::vector<HPolyhedron> regions;
  regions.push_back(domain);
  regions.push_back(domain);
  regions.push_back(domain);

  Eigen::MatrixXf collisions(2, 4);
  // clang-format off
  collisions << -0.9, 0.6, 1.1, 1.9,
                 1.0, 0.5,-1.0, 1.9;
  // clang-format on
  Eigen::MatrixXf repeated_collisions(2, 4 * 3);

  // Fill the new matrix by copying the original matrix 3 times horizontally
  for (int i = 0; i < 3; ++i) {
    repeated_collisions.block(0, i * 4, 2, 4) = collisions;
  }
  std::vector<u_int32_t> line_segment_idxs(
      line_start.cols() * collisions.cols(), 0);
  // Set values for index 1 and 2
  for (int i = 0; i < 4; ++i) {
    line_segment_idxs[4 + i] = 1;  // Set the second group to 1
    line_segment_idxs[8 + i] = 2;  // Set the third group to 2
  }

  auto result = EditRegionsCuda(
      repeated_collisions, line_segment_idxs, line_start, line_end, regions,
      plant, inspector.robot_geometry_ids, vox, voxel_radius, options);
  Eigen::MatrixXf projections = result.second.first;
  Eigen::MatrixXf opt = result.second.second;

  auto regions_edited = result.first;
  int region_idx = 0;
  for (const auto r : regions_edited) {
    EXPECT_TRUE(r.PointInSet(line_start.col(region_idx)));
    EXPECT_TRUE(r.PointInSet(line_end.col(region_idx)));
    for (int col_id = 0; col_id < opt.cols(); col_id++) {
      EXPECT_FALSE(r.PointInSet(opt.col(col_id)));
    }
    for (int col_id = 0; col_id < collisions.cols(); col_id++) {
      EXPECT_FALSE(r.PointInSet(collisions.col(col_id)));
    }
    region_idx++;
  }

  if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
    Eigen::Matrix2Xf obs_points;
    Eigen::Matrix2Xf centers;
    Eigen::Matrix2Xf env_points;

    env_points.resize(2, 5);
    // clang-format off
      env_points << -2, 2,  2, -2, -2,
                     2, 2, -2, -2,  2;
    // clang-format on

    double c = 1.0;
    centers.resize(2, 4);
    // clang-format off
      centers << -c, c,  c, -c,
                  c, c, -c, -c;
    // clang-format on

    // approximating offset due to sphere radius with fixed offset
    double s = 0.7 + 0.01;
    obs_points.resize(2, 5);
    // clang-format off
      obs_points << -s, s,  s, -s, -s,
                     s, s, -s, -s,  s;
    // clang-format on

    setupTestPlottingEnv();
    plt::figure_size(1000, 1000);
    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    for (const auto r : regions_edited) {
      PlottingUtils::plot2DHPolyhedron(r, "m", 1000);
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});
    PlottingUtils::scatter(collisions.row(0), collisions.row(1), 100,
                           {{"color", "r"}});

    PlottingUtils::scatter(projections.row(0), projections.row(1), 50,
                           {{"color", "b"}});

    PlottingUtils::scatter(opt.row(0), opt.row(1), 100, {{"color", "m"}});

    matplotlibcpp::plot(
        {line_start(0, 0), line_start(0, 1), line_start(0, 2), line_end(0, 2)},
        {line_start(1, 0), line_start(1, 1), line_start(1, 2), line_end(1, 2)},
        {{"color", "g"}, {"linewidth", "4.0"}});

    for (size_t vox_id = 0; vox_id < vox.cols(); ++vox_id) {
      PlottingUtils::draw_circle(vox(0, vox_id), vox(1, vox_id), voxel_radius,
                                 "k");
    }

    cleanupPlotAndPauseForUser();
  }
  // add extra test to check if the optimized collisions are actually set to the
  // initial collision if the bisection never hits any obstacles
  options.bisection_steps = 1;

  auto result2 = EditRegionsCuda(
      repeated_collisions, line_segment_idxs, line_start, line_end, regions,
      plant, inspector.robot_geometry_ids, vox, voxel_radius, options);
  Eigen::MatrixXf optimized_collisions = result2.second.second;
  float diff = (optimized_collisions.col(1) - collisions.col(1)).norm();
  EXPECT_LE(diff, 1e-6);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}