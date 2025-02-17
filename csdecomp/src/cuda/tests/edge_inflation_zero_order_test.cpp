#include <fmt/core.h>
#include <gtest/gtest.h>
#include <matplotlibcpp.h>

#include <chrono>
#include <string>

#include "cuda_collision_checker.h"
#include "cuda_edge_inflation_zero_order.h"
#include "cuda_hit_and_run_sampling.h"
#include "hpolyhedron.h"
#include "plotting_utils.h"
#include "urdf_parser.h"

namespace plt = matplotlibcpp;
using namespace PlottingUtils;
using namespace csdecomp;
#define DO_PLOT 1

class FCITest : public ::testing::Test {
 protected:
  void SetUp() override {
    parser.parseURDFString(boxes_in_corners_urdf);
    tree = parser.getKinematicTree();
    plant = parser.getMinimalPlant();
    inspector = parser.getSceneInspector();
    domain.MakeBox(tree.getPositionLowerLimits(),
                   tree.getPositionUpperLimits());

    // Draw the true cspace.
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
                      s, s, -s, -s, s;
    // clang-format on
  }
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
  HPolyhedron domain;
  URDFParser parser;
  KinematicTree tree;
  MinimalPlant plant;
  SceneInspector inspector;
  const int kMixingSteps = 50;
  Eigen::Matrix2Xf obs_points;
  Eigen::Matrix2Xf centers;
  Eigen::Matrix2Xf env_points;

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
};

TEST_F(FCITest, FCItest1) {
  Eigen::Vector2f l_start;
  l_start << 0, 0;
  Eigen::Vector2f l_end;
  l_end << 1, 0;
  EizoOptions options;
  EizoInfo ei_info;
  options.num_particles = 1000;
  options.track_iteration_information = true;
  Voxels vox;
  float voxel_radius = 0.01;
  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain, plant,
               inspector.robot_geometry_ids, options, ei_info);
  printEizoSummary(ei_info);
  HPolyhedron P(mpoly);

  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));
  Eigen::Matrix2Xf test_points(2, 2);
  // clang-format off
    test_points << 1.8, 1.8,
                   0.2, 0.4;
  // clang-format on
  EXPECT_TRUE(P.PointInSet(test_points.col(0)));
  EXPECT_FALSE(P.PointInSet(test_points.col(1)));

  if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
    setupTestPlottingEnv();
    plt::figure_size(1000, 1000);
    PlottingUtils::scatter(test_points.row(0), test_points.row(1), 50,
                           {{"color", "b"}});
    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});
    for (auto it_info : ei_info.iteration_info) {
      if (it_info.points_post_bisection.size()) {
        PlottingUtils::scatter(it_info.points_post_bisection.row(0),
                               it_info.points_post_bisection.row(1), 10.0,
                               {{"color", "r"}});
      }
    }

    matplotlibcpp::plot({l_start[0], l_end[0]}, {l_start[1], l_end[1]},
                        {{"color", "g"}, {"linewidth", "2.0"}});

    PlottingUtils::plot2DHPolyhedron(P, "r", 1000);
    cleanupPlotAndPauseForUser();
  }
}

TEST_F(FCITest, FCIThrows) {
  Eigen::Vector2f l_start;
  l_start << 0, 0;
  Eigen::Vector2f l_end;
  l_end << 1, 0;
  EizoOptions options;
  EizoInfo info;
  options.num_particles = 1000000;
  Voxels vox;
  float voxel_radius = 0.01;
  EXPECT_THROW(EizoCuda(l_start, l_end, vox, voxel_radius, domain, plant,
                        inspector.robot_geometry_ids, options, info),
               std::runtime_error);
  Eigen::VectorXf b2 = Eigen::VectorXf::Zero(MAX_NUM_FACES);
  Eigen::MatrixXf A2 =
      Eigen::MatrixXf::Zero(MAX_NUM_FACES, domain.ambient_dimension());
  HPolyhedron domain2(A2, b2);
  options.num_particles = 10;
  EXPECT_THROW(EizoCuda(l_start, l_end, vox, voxel_radius, domain2, plant,
                        inspector.robot_geometry_ids, options, info),
               std::runtime_error);
}

TEST_F(FCITest, FCItest2) {
  Eigen::Vector2f l_start;
  l_start << -0.5, 0;
  Eigen::Vector2f l_end;
  l_end << 0, 0.5;
  EizoOptions options;
  EizoInfo ei_info;
  options.num_particles = 1000;
  options.track_iteration_information = true;
  Voxels vox;
  float voxel_radius = 0.01;
  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain, plant,
               inspector.robot_geometry_ids, options, ei_info);
  printEizoSummary(ei_info);
  HPolyhedron P(mpoly);

  // sample in resulting poly and estimate fraciton in collision
  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 1000;
  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_coll_checks,
                                     50);

  std::vector<uint8_t> results =
      checkCollisionFreeCuda(&samples_poly.at(0), &plant);

  int num_collisions = std::count(results.begin(), results.end(), 0);
  float frac_in_col = num_collisions / (1.0 * num_coll_checks);

  EXPECT_LT(frac_in_col, options.epsilon * 1.1);

  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));
  Eigen::Matrix2Xf test_points(2, 2);
  // clang-format off
    test_points << 0.0, 0.0,
                   0.0, 0.8;
  // clang-format on
  EXPECT_TRUE(P.PointInSet(test_points.col(0)));
  EXPECT_FALSE(P.PointInSet(test_points.col(1)));

  if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
    setupTestPlottingEnv();
    plt::figure_size(1000, 1000);
    PlottingUtils::scatter(test_points.row(0), test_points.row(1), 50,
                           {{"color", "b"}});
    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});
    for (auto it_info : ei_info.iteration_info) {
      if (it_info.points_post_bisection.size()) {
        PlottingUtils::scatter(it_info.points_post_bisection.row(0),
                               it_info.points_post_bisection.row(1), 10.0,
                               {{"color", "r"}});
      }
    }

    matplotlibcpp::plot({l_start[0], l_end[0]}, {l_start[1], l_end[1]},
                        {{"color", "g"}, {"linewidth", "2.0"}});

    PlottingUtils::plot2DHPolyhedron(P, "r", 1000);
    cleanupPlotAndPauseForUser();
  }
}

// TEST_F(FCITest, LineInCollisionTest) {

//   //!!!!!!!!!!!!!!!!!!!!!
//   // TODO(wernerpe):
//   // we are not checking if the line segment is in collision yet. This edge
//   case
//   // leads to undefined behavior.
//   //!!!!!!!!!!!!!!!!!!!!!

//   EXPECT_TRUE(false);
//   Eigen::Vector2f l_start;
//   l_start << -1.5, 0;
//   Eigen::Vector2f l_end;
//   l_end << 0, 1.5;
//   EizoOptions options;
//   EizoInfo ei_info;
//   options.num_particles = 1000;
//   options.track_iteration_information = true;
//   MinimalHPolyhedron mpoly =
//       EizoCuda(l_start, l_end, domain, plant, options,
//       ei_info);
//   printEizoSummary(ei_info);
//   HPolyhedron P(mpoly);

//   // sample in resulting poly and estimate fraciton in collision
//   std::vector<HPolyhedron> poly_vec{P};
//   Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
//                                   poly_vec.size());
//   interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
//   const int num_coll_checks = 1000;
//   const std::vector<Eigen::MatrixXf> samples_poly =
//       UniformSampleInHPolyhedronCuda(poly_vec, interior_points,
//       num_coll_checks,
//                                      50);

//   std::vector<uint8_t> results =
//       checkCollisionFreeCuda(&samples_poly.at(0), &plant);

//   int num_collisions = std::count(results.begin(), results.end(), 0);
//   float frac_in_col = num_collisions / (1.0 * num_coll_checks);

//   EXPECT_LT(frac_in_col, options.epsilon * 1.1);

//   EXPECT_TRUE(P.PointInSet(l_start));
//   EXPECT_TRUE(P.PointInSet(l_end));
//   Eigen::Matrix2Xf test_points(2, 2);
//   // clang-format off
//     test_points << 0.0, 0.0,
//                    0.0, 0.8;
//   // clang-format on
//   EXPECT_TRUE(P.PointInSet(test_points.col(0)));
//   EXPECT_FALSE(P.PointInSet(test_points.col(1)));

//   if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
//     setupTestPlottingEnv();
//     plt::figure_size(1000, 1000);
//     PlottingUtils::scatter(test_points.row(0), test_points.row(1), 50,
//                            {{"color", "b"}});
//     PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
//     for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
//       Eigen::Matrix2Xf obstacle = obs_points;
//       obstacle.colwise() += centers.col(obstacle_idx);
//       PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
//                           {{"color", "k"}, {"linewidth", "2"}});
//     }
//     PlottingUtils::plot(env_points.row(0), env_points.row(1),
//                         {{"color", "k"}, {"linewidth", "2"}});
//     for (auto it_info : ei_info.iteration_info) {
//       if (it_info.points_post_bisection.size()) {
//         PlottingUtils::scatter(it_info.points_post_bisection.row(0),
//                                it_info.points_post_bisection.row(1), 10.0,
//                                {{"color", "r"}});
//       }
//     }

//     matplotlibcpp::plot({l_start[0], l_end[0]}, {l_start[1], l_end[1]},
//                         {{"color", "g"}, {"linewidth", "4.0"}});

//     PlottingUtils::plot2DHPolyhedron(P, "r", 1000);
//     cleanupPlotAndPauseForUser();
//   }
// }

TEST_F(FCITest, FCItest3) {
  Eigen::Vector2f l_start;
  l_start << -0.5, 1.75;
  Eigen::Vector2f l_end;
  l_end << 0.5, 1.75;
  EizoOptions options;
  EizoInfo ei_info;
  options.num_particles = 1000;
  options.track_iteration_information = true;
  Voxels vox;
  float voxel_radius = 0.01;
  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain, plant,
               inspector.robot_geometry_ids, options, ei_info);
  printEizoSummary(ei_info);
  HPolyhedron P(mpoly);

  // sample in resulting poly and estimate fraciton in collision
  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 1000;
  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_coll_checks,
                                     50);

  std::vector<uint8_t> results =
      checkCollisionFreeCuda(&samples_poly.at(0), &plant);

  int num_collisions = std::count(results.begin(), results.end(), 0);
  float frac_in_col = num_collisions / (1.0 * num_coll_checks);

  EXPECT_LT(frac_in_col, options.epsilon * 1.1);

  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));
  Eigen::Matrix2Xf test_points(2, 2);
  // clang-format off
    test_points << -1.75, 1.75,
                    1.75, 1.75;
  // clang-format on
  EXPECT_TRUE(P.PointInSet(test_points.col(0)));
  EXPECT_TRUE(P.PointInSet(test_points.col(1)));

  if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
    setupTestPlottingEnv();
    plt::figure_size(1000, 1000);
    PlottingUtils::scatter(test_points.row(0), test_points.row(1), 50,
                           {{"color", "b"}});
    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});
    for (auto it_info : ei_info.iteration_info) {
      if (it_info.points_post_bisection.size()) {
        PlottingUtils::scatter(it_info.points_post_bisection.row(0),
                               it_info.points_post_bisection.row(1), 10.0,
                               {{"color", "r"}});
      }
    }
    //

    matplotlibcpp::plot({l_start[0], l_end[0]}, {l_start[1], l_end[1]},
                        {{"color", "g"}, {"linewidth", "2.0"}});

    PlottingUtils::plot2DHPolyhedron(P, "r", 1000);
    cleanupPlotAndPauseForUser();
  }
}

TEST_F(FCITest, FCITestKinova) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  URDFParser parser2;
  parser2.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser2.parseDirectives(tmp));
  KinematicTree kt = parser2.getKinematicTree();
  MinimalPlant kinova_plant = parser2.getMinimalPlant();
  SceneInspector insp2 = parser2.getSceneInspector();
  Eigen::VectorXf l_start(7);
  l_start << 0, 0, 0, 0, 0, 0, 0;
  Eigen::VectorXf l_end(7);
  l_end << 0, 1, -1, 0.5, 0, 0, 0;
  HPolyhedron domain2;
  domain2.MakeBox(kt.getPositionLowerLimits(), kt.getPositionUpperLimits());
  EizoOptions options;
  EizoInfo ei_info;
  options.num_particles = 2000;
  options.max_iterations = 100;
  options.verbose = false;
  options.track_iteration_information = false;
  std::cout << domain2.A().rows() << std::endl;
  Voxels vox;
  float voxel_radius = 0.01;
  // burn in run
  EizoCuda(l_start, l_end, vox, voxel_radius, domain2, kinova_plant,
           insp2.robot_geometry_ids, options, ei_info);

  options.track_iteration_information = true;

  auto start = std::chrono::high_resolution_clock::now();
  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain2, kinova_plant,
               insp2.robot_geometry_ids, options, ei_info);

  auto stop = std::chrono::high_resolution_clock::now();
  tmp = fmt::format(
      "execution time FastEdgeInlfationCuda with malloc and copy: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;

  printEizoSummary(ei_info);

  HPolyhedron P(mpoly);
  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));

  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 1000;
  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_coll_checks,
                                     50);

  std::vector<uint8_t> results =
      checkCollisionFreeCuda(&samples_poly.at(0), &kinova_plant);

  int num_collisions = std::count(results.begin(), results.end(), 0);
  float frac_in_col = num_collisions / (1.0 * num_coll_checks);

  EXPECT_LT(frac_in_col, options.epsilon * 1.2);
}

TEST_F(FCITest, FCI2dVoxelTest1) {
  Eigen::Vector2f l_start;
  l_start << -0.5, 1.75;
  Eigen::Vector2f l_end;
  l_end << 0.5, 1.75;
  EizoOptions options;
  EizoInfo ei_info;
  options.num_particles = 1000;
  options.track_iteration_information = true;
  Voxels vox(3, 3);
  // clang-format off
    vox<< -0.55, -1.50, 1.90,  
           2.05,  1.80, 1.90,
           0.00,  0.00, 0.00;
  //clang-format on
  float voxel_radius = 0.05;
  std::cout << vox<<std::endl;
  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain, plant,
                            inspector.robot_geometry_ids, options, ei_info);
  printEizoSummary(ei_info);
  HPolyhedron P(mpoly);

  // sample in resulting poly and estimate fraciton in collision
  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 1000;

  const std::vector<Eigen::MatrixXf> samples_poly =
      UniformSampleInHPolyhedronCuda(poly_vec, interior_points, num_coll_checks,
                                     50);

  std::vector<uint8_t> results =
      checkCollisionFreeCuda(&samples_poly.at(0), &plant);

  int num_collisions = std::count(results.begin(), results.end(), 0);
  float frac_in_col = num_collisions / (1.0 * num_coll_checks);

  EXPECT_LT(frac_in_col, options.epsilon * 1.1);

  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));
  Eigen::Matrix2Xf test_points(2, 2);
  // clang-format off
    test_points << -1.75, 1.75,
                    1.75, 1.8;
  // clang-format on
  EXPECT_FALSE(P.PointInSet(test_points.col(0)));
  EXPECT_TRUE(P.PointInSet(test_points.col(1)));

  if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
    setupTestPlottingEnv();
    plt::figure_size(1000, 1000);
    PlottingUtils::scatter(test_points.row(0), test_points.row(1), 50,
                           {{"color", "b"}});

    for (size_t vox_id = 0; vox_id < vox.cols(); ++vox_id) {
      PlottingUtils::draw_circle(vox(0, vox_id), vox(1, vox_id), voxel_radius,
                                 "k");
    }

    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});
    for (auto it_info : ei_info.iteration_info) {
      if (it_info.points_post_bisection.size()) {
        std::cout << it_info.points_post_bisection.leftCols(10) << std::endl;
        std::cout << "------------------------------" << std::endl;
        PlottingUtils::scatter(it_info.points_post_bisection.row(0),
                               it_info.points_post_bisection.row(1), 10.0,
                               {{"color", "r"}});
      }
    }

    matplotlibcpp::plot({l_start[0], l_end[0]}, {l_start[1], l_end[1]},
                        {{"color", "g"}, {"linewidth", "2.0"}});

    PlottingUtils::plot2DHPolyhedron(P, "r", 1000);
    cleanupPlotAndPauseForUser();
  }
}

TEST_F(FCITest, FCITestKinovaVoxels) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  URDFParser parser2;
  parser2.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser2.parseDirectives(tmp));
  KinematicTree kt = parser2.getKinematicTree();
  MinimalPlant kinova_plant = parser2.getMinimalPlant();
  SceneInspector insp2 = parser2.getSceneInspector();
  Eigen::VectorXf l_start(7);
  l_start << -0.59, 1.1554, 1.01, 0., 0., 0., 0.;
  Eigen::VectorXf l_end(7);
  l_end << -0.59, 0.8644, 0.666, -0.4353, -0.525, -0.1871, 0.;
  HPolyhedron domain2;
  domain2.MakeBox(kt.getPositionLowerLimits(), kt.getPositionUpperLimits());
  EizoOptions options;
  EizoInfo ei_info;
  options.num_particles = 1000;
  options.max_iterations = 100;
  options.verbose = false;
  options.mixing_steps = 150;
  options.track_iteration_information = false;
  std::cout << domain2.A().rows() << std::endl;
  Voxels vox(3, 4);

  // clang-format off
    vox << 0.3, 0.3, 0.3, 0.3,
           0.0, 0.4, 0.4,-0.4,
           0.5, 0.5, 0.6, 0.5;
  // clang-format on

  float voxel_radius = 0.01;
  // burn in run
  EizoCuda(l_start, l_end, vox, voxel_radius, domain2, kinova_plant,
           insp2.robot_geometry_ids, options, ei_info);

  options.track_iteration_information = true;

  auto start = std::chrono::high_resolution_clock::now();
  MinimalHPolyhedron mpoly =
      EizoCuda(l_start, l_end, vox, voxel_radius, domain2, kinova_plant,
               insp2.robot_geometry_ids, options, ei_info);

  auto stop = std::chrono::high_resolution_clock::now();
  tmp = fmt::format(
      "execution time FastEdgeInlfationCuda with malloc and copy: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;

  printEizoSummary(ei_info);

  HPolyhedron P(mpoly);
  EXPECT_TRUE(P.PointInSet(l_start));
  EXPECT_TRUE(P.PointInSet(l_end));

  std::vector<HPolyhedron> poly_vec{P};
  Eigen::MatrixXf interior_points(poly_vec.at(0).ambient_dimension(),
                                  poly_vec.size());
  interior_points.col(0) = poly_vec.at(0).ChebyshevCenter();
  const int num_coll_checks = 1000;
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