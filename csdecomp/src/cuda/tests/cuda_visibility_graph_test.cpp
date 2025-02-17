#include <fmt/core.h>
#include <gtest/gtest.h>
#include <matplotlibcpp.h>

#include <chrono>
#include <string>
// #include "fast_clique_inflation.h"
// #include "collision_checker.h"
#include "cuda_collision_checker.h"
#include "cuda_hit_and_run_sampling.h"
#include "cuda_visibility_graph.h"
#include "hpolyhedron.h"
#include "plotting_utils.h"
#include "urdf_parser.h"

namespace plt = matplotlibcpp;
using namespace PlottingUtils;
using namespace csdecomp;
#define DO_PLOT 1

namespace {
std::pair<uint32_t, uint32_t> get_indices_from_flat_index_cpp(
    const uint64_t& flat_index, const uint32_t& n) {
  uint32_t i =
      n - 2 -
      std::floor(std::sqrt(-8 * flat_index + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
  uint32_t j =
      flat_index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;

  return std::make_pair(i, j);
}
}  // namespace

class VGTest : public ::testing::Test {
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

TEST_F(VGTest, VGTest0) {
  Eigen::MatrixXf nodes(2, 5);
  // clang-format off
    nodes  <<  0, 1.5, 0, -1.5, 0,
               0, 0, 1.5,  0, -1.5;
  // clang-format on
  Eigen::SparseMatrix<uint8_t> adjacency_matrix =
      VisibilityGraph(nodes, plant, 0.1, 1000000);

  // clang-format off
  std::vector<uint8_t> adj_entries = {1,1,1,1,
                                        0,1,0,
                                          0,1,
                                            0};
  // clang-format on
  for (int fl_idx = 0; fl_idx < adj_entries.size(); ++fl_idx) {
    auto [i, j] = get_indices_from_flat_index_cpp(fl_idx, nodes.cols());
    EXPECT_EQ(adjacency_matrix.coeff(i, j), adj_entries.at(fl_idx));
  }

  Eigen::SparseMatrix<uint8_t> adjacency_matrix2 =
      VisibilityGraph(nodes, plant, 0.1, 100);

  EXPECT_TRUE(adjacency_matrix2.isApprox(adjacency_matrix));

  EXPECT_DEATH(VisibilityGraph(nodes, plant, 0.1, 10000000000), "");

  if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
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
    PlottingUtils::scatter(nodes.row(0), nodes.row(1), 20, {{"color", "r"}});

    for (int i = 0; i < nodes.cols() - 1; ++i) {
      for (int j = i + 1; j < nodes.cols(); ++j) {
        if (adjacency_matrix.coeff(i, j)) {
          plt::plot({nodes(0, i), nodes(0, j)}, {nodes(1, i), nodes(1, j)},
                    {{"color", "k"}, {"linewidth", "1"}});
        }
      }
    }
    cleanupPlotAndPauseForUser();
  }
}

GTEST_TEST(VisibilityGraphTest7d, Kinova) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  URDFParser parser2;
  parser2.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser2.parseDirectives(tmp));
  Plant kinova_plant = parser2.buildPlant();
  MinimalPlant kinova_mplant = kinova_plant.getMinimalPlant();
  HPolyhedron domain2;
  domain2.MakeBox(kinova_plant.getPositionLowerLimits(),
                  kinova_plant.getPositionUpperLimits());
  std::vector<Eigen::MatrixXf> res_vec = UniformSampleInHPolyhedronCuda(
      {domain2}, domain2.ChebyshevCenter(), 1000, 100);
  std::vector<uint8_t> is_col_free =
      checkCollisionFreeCuda(&(res_vec.at(0)), &kinova_mplant);
  int num_col_free = std::accumulate(is_col_free.begin(), is_col_free.end(), 0);
  Eigen::MatrixXf nodes(kinova_plant.numConfigurationVariables(), num_col_free);
  int idx = 0;
  int idx_col_free = 0;
  for (const auto& r : is_col_free) {
    if (r) {
      nodes.col(idx_col_free) = res_vec.at(0).col(idx);
      ++idx_col_free;
    }
    ++idx;
  }
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::SparseMatrix<uint8_t> adjacency_matrix =
      VisibilityGraph(nodes, kinova_mplant, 0.2, 500000);
  auto stop = std::chrono::high_resolution_clock::now();

  // compare the first 10 results

  tmp = fmt::format(
      "Time to construct visibility graph on kinova example with {} nodes: {} "
      "ms",
      nodes.cols(),
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
}

// // This test only serves for profiling
// TEST_F(VGTest, VGTest1) {
//   std::vector<Eigen::MatrixXf> res_vec = UniformSampleInHPolyhedronCuda(
//       {domain}, domain.ChebyshevCenter(), 10000, 100);

//   std::vector<uint8_t> is_col_free =
//       checkCollisionFreeCuda(&(res_vec.at(0)), &plant);
//   int num_col_free = std::accumulate(is_col_free.begin(), is_col_free.end(),
//   0); Eigen::MatrixXf nodes(2, num_col_free);
//   //   Eigen::MatrixXf nodes(2, 4);
//   //   // clang-format off
//   //     nodes << 0.0479718,  0.147564,  0.152781,  -1.13069,
//   //              0.827174,  -1.19944, -0.326808,   1.92674;
//   //   // clang-format on

//   int idx = 0;
//   int idx_col_free = 0;
//   for (const auto& r : is_col_free) {
//     if (r) {
//       nodes.col(idx_col_free) = res_vec.at(0).col(idx);
//       ++idx_col_free;
//     }
//     ++idx;
//   }
//   //   std::cout << nodes << std::endl;
//   auto start = std::chrono::high_resolution_clock::now();
//   Eigen::SparseMatrix<uint8_t> adjacency_matrix =
//       VisibilityGraph(nodes, plant, 0.2, 1000000);
//   auto stop = std::chrono::high_resolution_clock::now();

//   std::string tmp = fmt::format(
//       "time to construct visibility graph with {} nodes: {} ms",
//       nodes.cols(),
//       std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
//           .count());
//   std::cout << tmp << std::endl;

//   if (std::getenv("BAZEL_TEST") == nullptr && DO_PLOT) {
//     setupTestPlottingEnv();
//     plt::figure_size(1000, 1000);
//     PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);
//     for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
//       Eigen::Matrix2Xf obstacle = obs_points;
//       obstacle.colwise() += centers.col(obstacle_idx);
//       PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
//                           {{"color", "k"}, {"linewidth", "2"}});
//     }
//     PlottingUtils::plot(env_points.row(0), env_points.row(1),
//                         {{"color", "k"}, {"linewidth", "2"}});
//     PlottingUtils::scatter(nodes.row(0), nodes.row(1), 20, {{"color", "r"}});
//     int num_plot = 0;
//     for (int i = 0; i < nodes.cols() - 1; ++i) {
//       for (int j = i + 1; j < nodes.cols(); ++j) {
//         if (adjacency_matrix.coeff(i, j)) {
//           plt::plot({nodes(0, i), nodes(0, j)}, {nodes(1, i), nodes(1, j)},
//                     {{"color", "k"}, {"linewidth", "1"}});
//           ++num_plot;
//         }
//       }
//       if (num_plot > 2000) break;
//     }
//     cleanupPlotAndPauseForUser();
//   }
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}