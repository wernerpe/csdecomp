#include <fmt/core.h>
#include <gtest/gtest.h>

#include <chrono>
#include <string>

#include "collision_checker.h"
#include "cuda_collision_checker.h"
#include "cuda_hit_and_run_sampling.h"
#include "cuda_polytope_builder.h"
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

  EizoOptions fei_options;
  fei_options.track_iteration_information = true;
  Voxels vox(3, 3);
  // clang-format off
    vox<< -0.55, -1.50, 1.90,  
           2.02,  1.80, 1.90,
           0.00,  0.00, 0.00;
  // clang-format on
  float voxel_radius = 0.05;

  Eigen::MatrixXf line_start(2, 6);
  Eigen::MatrixXf line_end(2, 6);
  Eigen::Vector2f l_start;
  l_start << -0.5, 1.75;
  Eigen::Vector2f l_end;
  l_end << 0.5, 1.75;
  std::vector<HPolyhedron> regions;
  EizoOptions options;
  EizoInfo eizo_info, eizo_info1, eizo_info2;

  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain, plant,
               inspector.robot_geometry_ids, options, eizo_info);
  printEizoSummary(eizo_info);
  regions.push_back(HPolyhedron(mpoly));

  CudaEdgeInflator set_builder(plant, inspector.robot_geometry_ids, fei_options,
                               domain);
  regions.push_back(
      set_builder.inflateEdge(l_start, l_end, vox, voxel_radius, true));

  for (int vox_id = 0; vox_id < vox.cols(); ++vox_id) {
    for (auto r : regions) {
      EXPECT_FALSE(r.PointInSet(vox.col(vox_id).topRows(2)));
    }
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
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});

    matplotlibcpp::plot({l_start[0], l_end[0]}, {l_start[1], l_end[1]},
                        {{"color", "g"}, {"linewidth", "2.0"}});
    for (size_t vox_id = 0; vox_id < vox.cols(); ++vox_id) {
      PlottingUtils::draw_circle(vox(0, vox_id), vox(1, vox_id), voxel_radius,
                                 "k");
    }

    cleanupPlotAndPauseForUser();
  }
}

GTEST_TEST(PolytopeBuilderTest, TWODEnvTestWithPlotting) {
  URDFParser parser;
  parser.parseURDFString(boxes_in_corners_urdf);
  KinematicTree tree = parser.getKinematicTree();
  MinimalPlant plant = parser.getMinimalPlant();
  SceneInspector inspector = parser.getSceneInspector();
  HPolyhedron domain;
  domain.MakeBox(tree.getPositionLowerLimits(), tree.getPositionUpperLimits());

  EizoOptions fei_options;
  fei_options.track_iteration_information = true;

  CudaEdgeInflator set_builder(plant, inspector.robot_geometry_ids, fei_options,
                               domain);
  Voxels vox(3, 6);
  // clang-format off
    vox<< -0.55, -1.50, 0.20, 1.90, -1.90,  1.90,  
           2.05,  1.80, 0.20, 0.00, 0.00,  1.90,
           0.00,  0.00, 0.00, 0.00, 0.00,  0.00;
  //clang-format on
  float voxel_radius = 0.05;

  Eigen::MatrixXf line_start(2,6);
  Eigen::MatrixXf line_end(2,6);
  // clang-format off
  line_start<< -0.55, -0.2 , 1.75,-1.75,  0.50,  0.50,  
               -0.20,  1.00, 1.00, 1.00,  1.75, -1.75;

  line_end  <<  0.55, -0.20, 1.75,-1.75, -0.50, -0.50,  
               -0.20, -1.00,-1.00,-1.00,  1.75, -1.75;
  // clang-format on
  std::vector<HPolyhedron> regions;

  for (size_t lineidx = 0; lineidx < line_start.cols(); ++lineidx) {
    regions.push_back(set_builder.inflateEdge(line_start.col(lineidx),
                                              line_end.col(lineidx), vox,
                                              voxel_radius, true));
  }
  for (int vox_id = 0; vox_id < vox.cols(); ++vox_id) {
    for (auto r : regions) {
      EXPECT_FALSE(r.PointInSet(vox.col(vox_id).topRows(2)));
    }
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
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});

    for (size_t lineidx = 0; lineidx < line_start.cols(); ++lineidx) {
      matplotlibcpp::plot({line_start(0, lineidx), line_end(0, lineidx)},
                          {line_start(1, lineidx), line_end(1, lineidx)},
                          {{"color", "g"}, {"linewidth", "2.0"}});
    }

    for (size_t vox_id = 0; vox_id < vox.cols(); ++vox_id) {
      PlottingUtils::draw_circle(vox(0, vox_id), vox(1, vox_id), voxel_radius,
                                 "k");
    }
    for (auto r : regions) {
      PlottingUtils::plot2DHPolyhedron(r, "r", 1000);
    }
    cleanupPlotAndPauseForUser();
  }
}

GTEST_TEST(PolytopeBuilderTest, KinovaVoxels) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  URDFParser parser;
  parser.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser.parseDirectives(tmp));
  KinematicTree kt = parser.getKinematicTree();
  MinimalPlant kinova_plant = parser.getMinimalPlant();
  SceneInspector insp = parser.getSceneInspector();
  Eigen::VectorXf l_start(7);
  l_start << -0.59, 1.1554, 1.01, 0., 0., 0., 0.;
  Eigen::VectorXf l_end(7);
  l_end << -0.59, 0.8644, 0.666, -0.4353, -0.525, -0.1871, 0.;
  HPolyhedron domain;
  domain.MakeBox(kt.getPositionLowerLimits(), kt.getPositionUpperLimits());
  EizoOptions options;

  options.num_particles = 10000;
  options.max_iterations = 100;
  options.verbose = false;
  options.track_iteration_information = true;

  CudaEdgeInflator set_builder(kinova_plant, insp.robot_geometry_ids, options,
                               domain);
  Voxels vox(3, 4);

  // clang-format off
    vox << 0.3, 0.3, 0.3, 0.3,
           0.0, 0.4, 0.4,-0.4,
           0.5, 0.5, 0.6, 0.5;
  // clang-format on

  float voxel_radius = 0.01;

  options.track_iteration_information = true;

  auto start = std::chrono::high_resolution_clock::now();
  HPolyhedron P =
      set_builder.inflateEdge(l_start, l_end, vox, voxel_radius, true);
  auto stop = std::chrono::high_resolution_clock::now();
  tmp = fmt::format(
      "execution time FastEdgeInlfationCuda with malloc and copy: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));

  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 10000;
  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_coll_checks,
                                     50);

  std::vector<uint8_t> results =
      checkCollisionFreeCuda(&samples_poly.at(0), &kinova_plant);

  int num_collisions = std::count(results.begin(), results.end(), 0);
  float frac_in_col = num_collisions / (1.0 * num_coll_checks);

  EXPECT_LT(frac_in_col, options.epsilon * 1.2);
}

GTEST_TEST(PolytopeBuilderTest, KinovaManyVoxels) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  URDFParser parser;
  parser.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser.parseDirectives(tmp));
  KinematicTree kt = parser.getKinematicTree();
  MinimalPlant kinova_plant = parser.getMinimalPlant();
  SceneInspector insp = parser.getSceneInspector();
  Eigen::VectorXf l_start(7);
  l_start << -0.59, 1.1554, 1.01, 0., 0., 0., 0.;
  Eigen::VectorXf l_end(7);
  l_end << -0.59, 0.8644, 0.666, -0.4353, -0.525, -0.1871, 0.;
  HPolyhedron domain;
  domain.MakeBox(kt.getPositionLowerLimits(), kt.getPositionUpperLimits());
  EizoOptions options;

  options.num_particles = 10000;
  options.max_iterations = 100;
  options.max_hyperplanes_per_iteration = 20;
  options.verbose = false;
  options.track_iteration_information = true;

  CudaEdgeInflator set_builder(kinova_plant, insp.robot_geometry_ids, options,
                               domain);

  int num_voxels = 1000;
  Voxels vox_old(3, 4);
  Voxels vox_new(3, num_voxels);

  // clang-format off
    vox_old << 0.3, 0.3, 0.3, 0.3,
           0.0, 0.4, 0.4,-0.4,
           0.5, 0.5, 0.6, 0.5;
  // clang-format on
  for (int i = 0; i < num_voxels; ++i) {
    vox_new.col(i) = vox_old.col(i % 3);
  }

  float voxel_radius = 0.01;

  options.track_iteration_information = true;

  auto start = std::chrono::high_resolution_clock::now();
  HPolyhedron P =
      set_builder.inflateEdge(l_start, l_end, vox_new, voxel_radius, true);
  auto stop = std::chrono::high_resolution_clock::now();
  tmp = fmt::format(
      "execution time FastEdgeInlfationCuda with malloc and copy: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));

  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 10000;
  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_coll_checks,
                                     50);

  std::vector<uint8_t> results =
      checkCollisionFreeCuda(&samples_poly.at(0), &kinova_plant);

  int num_collisions = std::count(results.begin(), results.end(), 0);
  float frac_in_col = num_collisions / (1.0 * num_coll_checks);

  EXPECT_LT(frac_in_col, options.epsilon * 1.2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}