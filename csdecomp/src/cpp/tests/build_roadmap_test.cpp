#include <fmt/core.h>
#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <unordered_map>

#include "cuda_hit_and_run_sampling.h"
#include "drm_planner.h"
#include "hpolyhedron.h"
#include "plotting_utils.h"
#include "roadmap_builder.h"
#include "urdf_parser.h"

namespace plt = matplotlibcpp;
using namespace PlottingUtils;
using namespace csdecomp;

class DRMTest : public ::testing::Test {
 protected:
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
};

namespace {
std::string getTempDir() {
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  return (test_tmpdir != nullptr) ? std::string(test_tmpdir) : "tmp";
}
}  // namespace

TEST_F(DRMTest, CollisionSetTest) {
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

  URDFParser parser;
  EXPECT_TRUE(parser.parseURDFString(boxes_in_corners_urdf));
  Plant plant = parser.buildPlant();
  RoadmapOptions options;
  options.robot_map_size_x = 0.4;
  options.robot_map_size_y = 0.4;
  options.robot_map_size_z = 0.1;
  options.max_task_space_distance_between_nodes = 2;
  options.max_configuration_distance_between_nodes = 2;
  options.offline_voxel_resolution = 0.02;

  options.edge_step_size = 0.05;
  RoadmapBuilder roadmap_builder(plant, "movable", options);

  HPolyhedron domain;
  domain.MakeBox(plant.getPositionLowerLimits(),
                 plant.getPositionUpperLimits());

  Eigen::MatrixXf node_grid(2, 3);
  // clang-format off
  node_grid << 0, -0.2,  0.2,
               0, -0.2, -0.2;
  // clang-format on

  int num_added = roadmap_builder.AddNodesManual(node_grid);

  roadmap_builder.BuildRoadmap(10);
  roadmap_builder.BuildPoseMap();
  roadmap_builder.BuildCollisionMap();
  std::string tmp_dir = getTempDir();
  std::string road_map_file = std::string(tmp_dir) + "/drm.data";
  roadmap_builder.Write(road_map_file);

  DrmPlannerOptions opts;
  opts.try_shortcutting = false;
  opts.max_number_planning_attempts = 1000;
  DrmPlanner drm_planner(plant, opts);
  drm_planner.LoadRoadmap(road_map_file);

  Voxels root_p_voxels(3, 2);
  // clang-format off
  root_p_voxels << 0.09, -0.19,
                      0, -0.19,
                      0,     0;
  // clang-format on
  drm_planner.BuildCollisionSet(root_p_voxels);

  std::vector<int32_t> triggered_voxel_ids;
  for (int i = 0; i < root_p_voxels.cols(); ++i) {
    triggered_voxel_ids.push_back(
        GetCollisionVoxelId(root_p_voxels.col(i), options));
  }
  std::vector<Eigen::Vector3f> triggered_voxel_locations;
  for (const auto& t : triggered_voxel_ids) {
    if (t >= 0)
      triggered_voxel_locations.push_back(GetCollisionVoxelCenter(t, options));
  }

  std::vector<Eigen::Vector3f> offline_voxel_locations;
  int nx =
      ceil(options.robot_map_size_x / (2 * options.offline_voxel_resolution));
  int ny =
      ceil(options.robot_map_size_y / (2 * options.offline_voxel_resolution));
  int nz =
      ceil(options.robot_map_size_z / (2 * options.offline_voxel_resolution));
  // assert(nz == 1);
  for (int i = 0; i < nx * ny * nz; ++i) {
    offline_voxel_locations.push_back(GetCollisionVoxelCenter(i, options));
  }

  if (std::getenv("BAZEL_TEST") == nullptr) {
    setupTestPlottingEnv();
    plt::figure_size(2000, 2000);

    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);

    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});

    for (const auto& c : offline_voxel_locations) {
      PlottingUtils::draw_circle(c[0], c[1], options.GetOfflineVoxelRadius(),
                                 "m", 100, "1");
    }
    for (const auto& c : triggered_voxel_locations) {
      PlottingUtils::draw_circle(c[0], c[1], options.GetOfflineVoxelRadius(),
                                 "g", 100, "3");
    }
    const std::unordered_map<int32_t, Eigen::VectorXf> id_to_config =
        drm_planner.drm_.id_to_node_map;

    const std::unordered_map<int32_t, std::vector<int32_t>> adjacency =
        drm_planner.drm_.node_adjacency_map;
    // plot roadmap nodes
    std::cout << fmt::format("num nodes total  = {}\n", id_to_config.size());
    for (const auto& pair : id_to_config) {
      float radius = 0.02;
      PlottingUtils::draw_circle(pair.second(0), pair.second(1), radius, "b",
                                 10);
      const Eigen::VectorXf config = pair.second;
      if (adjacency.find(pair.first) != adjacency.end()) {
        for (const auto& other_id : adjacency.at(pair.first)) {
          // std::cout << pair.first << "," << other_id << std::endl;
          const Eigen::VectorXf other_config = id_to_config.at(other_id);
          plt::plot({config(0), other_config(0)}, {config(1), other_config(1)},
                    {{"color", "k"}, {"linewidth", "1.0"}});
        }
      }
    }
    for (const auto& id : drm_planner.collision_set_) {
      PlottingUtils::draw_circle(id_to_config.at(id)(0), id_to_config.at(id)(1),
                                 0.001, "r");
    }
    for (size_t vox_id = 0; vox_id < root_p_voxels.cols(); ++vox_id) {
      PlottingUtils::draw_circle(root_p_voxels(0, vox_id),
                                 root_p_voxels(1, vox_id), 0.01, "k", 10, "1");
    }
    cleanupPlotAndPauseForUser(1.9);
  }
}

TEST_F(DRMTest, MultiVoxelWReplanning) {
  // Build roadmap(offline)
  URDFParser parser;
  EXPECT_TRUE(parser.parseURDFString(boxes_in_corners_urdf));
  Plant plant = parser.buildPlant();
  RoadmapOptions options;
  options.robot_map_size_x = 4;
  options.robot_map_size_y = 2;
  options.robot_map_size_z = 0.2;

  options.max_task_space_distance_between_nodes = 2;
  options.max_configuration_distance_between_nodes = 2;
  options.offline_voxel_resolution = 0.05;

  options.edge_step_size = 0.05;
  RoadmapBuilder roadmap_builder(plant, "movable", options);
  constexpr int kMaxNeighbors = 10;

  int nx = 10;
  int ny = 10;
  float lx = 3.9;  // length x
  float ly = 1.5;  // length y
  Eigen::MatrixXf node_grid(2, nx * ny);
  float dx = lx / (nx - 1);
  float dy = ly / (ny - 1);

  int index = 0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      float x = -lx / 2.0 + i * dx;
      float y = -ly / 2.0 + j * dy;
      node_grid.col(index) << x, y;
      ++index;
    }
  }

  // std::cout << node_grid << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  int num_added = roadmap_builder.AddNodesManual(node_grid);
  roadmap_builder.BuildRoadmap(kMaxNeighbors);
  roadmap_builder.BuildPoseMap();
  roadmap_builder.BuildCollisionMap();
  auto stop = std::chrono::high_resolution_clock::now();

  std::string tmp = fmt::format(
      "Roadmap build time: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  std::string tmp_dir = getTempDir();
  std::string road_map_file_no_edges =
      std::string(tmp_dir) + "/drm_no_edges.data";
  roadmap_builder.Write(road_map_file_no_edges);
  // Now build the edge collision map
  roadmap_builder.BuildEdgeCollisionMap();
  std::string road_map_file = std::string(tmp_dir) + "/drm.data";
  roadmap_builder.Write(road_map_file);

  // Initialize planner.
  DrmPlannerOptions opts;
  opts.try_shortcutting = false;
  opts.max_number_planning_attempts = 1000;
  DrmPlanner drm_planner(plant, opts);
  drm_planner.LoadRoadmap(road_map_file);

  DRM drm_post_build = roadmap_builder.GetDRM();

  for (const auto& [key, edge_set] : drm_post_build.edge_collision_map) {
    EXPECT_EQ(edge_set.size(), drm_planner.drm_.edge_collision_map[key].size());
  }

  roadmap_builder.Reset();
  const DRM drm_reset = roadmap_builder.GetDRM();
  EXPECT_NE(drm_reset.options.max_task_space_distance_between_nodes,
            options.max_task_space_distance_between_nodes);

  roadmap_builder.Read(road_map_file);
  const DRM drm_read = roadmap_builder.GetDRM();
  EXPECT_EQ(drm_read.options.max_task_space_distance_between_nodes,
            options.max_task_space_distance_between_nodes);

  HPolyhedron domain;
  domain.MakeBox(plant.getPositionLowerLimits(),
                 plant.getPositionUpperLimits());

  // Initialize environment.
  Eigen::VectorXf initial_sample =
      Eigen::VectorXf::Zero(plant.numConfigurationVariables());
  std::vector<Eigen::VectorXf> robot_joint_path;
  int num_voxels = 20;
  float voxel_radius = 0.02;
  Voxels root_p_voxels(3, num_voxels);
  for (int ii = 0; ii < num_voxels; ++ii) {
    Eigen::Vector3f root_p_voxel;
    root_p_voxel[0] = 0.0;
    root_p_voxel[2] = 0.0;
    root_p_voxel[1] = -0.4 + 0.04 * ii;
    root_p_voxels.col(ii) = root_p_voxel;
  }

  std::vector<int32_t> triggered_voxel_ids;
  std::cout << fmt::format("used options\n x: {} y: {} z: {} \n",
                           options.robot_map_size_x, options.robot_map_size_y,
                           options.robot_map_size_z);
  for (int i = 0; i < root_p_voxels.cols(); ++i) {
    triggered_voxel_ids.push_back(
        GetCollisionVoxelId(root_p_voxels.col(i), options));
    std::cout << "\n Voxel: " << root_p_voxels.col(i).transpose()
              << "  Voxel ID: " << triggered_voxel_ids.back() << std::endl;
  }
  std::cout << " center 0--\n"
            << GetCollisionVoxelCenter(121, options) << std::endl;
  std::cout << "loop ----\n"
            << GetCollisionVoxelId(GetCollisionVoxelCenter(121, options),
                                   options)
            << std::endl;

  std::vector<Eigen::Vector3f> triggered_voxel_locations;
  for (const auto& t : triggered_voxel_ids) {
    triggered_voxel_locations.push_back(GetCollisionVoxelCenter(t, options));
  }

  Eigen::VectorXf start_configuration = Eigen::VectorXf::Zero(2);
  start_configuration[0] = -1.95;
  Eigen::VectorXf target_configuration = Eigen::VectorXf::Zero(2);
  target_configuration[0] = 1.95;
  std::cout << "Start configuration: " << start_configuration.transpose()
            << std::endl;
  std::cout << "Target configuration: " << target_configuration.transpose()
            << std::endl;

  // Plan path (online) with single voxel at origin.
  drm_planner.LoadRoadmap(road_map_file_no_edges);
  drm_planner.BuildCollisionSet(root_p_voxels);
  EXPECT_TRUE(drm_planner.Plan(start_configuration, target_configuration,
                               root_p_voxels, voxel_radius, &robot_joint_path));
  drm_planner.options_.try_shortcutting = true;
  EXPECT_TRUE(drm_planner.Plan(start_configuration, target_configuration,
                               root_p_voxels, voxel_radius, &robot_joint_path));

  std::cout << "Robot joint path without edge collision map: " << std::endl
            << "Size: " << robot_joint_path.size() << std::endl;
  for (const auto& joint : robot_joint_path) {
    std::cout << joint.transpose() << std::endl;
  }

  drm_planner.LoadRoadmap(road_map_file);
  drm_planner.BuildCollisionSet(root_p_voxels);
  EXPECT_TRUE(drm_planner.Plan(start_configuration, target_configuration,
                               root_p_voxels, voxel_radius, &robot_joint_path));
  drm_planner.options_.try_shortcutting = true;
  EXPECT_TRUE(drm_planner.Plan(start_configuration, target_configuration,
                               root_p_voxels, voxel_radius, &robot_joint_path));

  std::cout << "Robot joint path: " << std::endl
            << "Size: " << robot_joint_path.size() << std::endl;
  for (const auto& joint : robot_joint_path) {
    std::cout << joint.transpose() << std::endl;
  }

  // plotting
  if (std::getenv("BAZEL_TEST") == nullptr) {
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
    plt::figure_size(2000, 2000);

    PlottingUtils::plot2DHPolyhedron(domain, "b", 1000);

    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix2Xf obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      PlottingUtils::plot(obstacle.row(0), obstacle.row(1),
                          {{"color", "k"}, {"linewidth", "2"}});
    }
    PlottingUtils::plot(env_points.row(0), env_points.row(1),
                        {{"color", "k"}, {"linewidth", "2"}});

    PlottingUtils::draw_circle(start_configuration(0), start_configuration(1),
                               0.05, "g", 10);
    PlottingUtils::draw_circle(target_configuration(0), target_configuration(1),
                               0.05, "g", 10);
    const std::unordered_map<int32_t, Eigen::VectorXf> id_to_config =
        drm_planner.drm_.id_to_node_map;

    const std::unordered_map<int32_t, std::vector<int32_t>> adjacency =
        drm_planner.drm_.node_adjacency_map;
    // plot roadmap nodes
    std::cout << fmt::format("num nodes total  = {}\n", id_to_config.size());
    for (const auto& pair : id_to_config) {
      float radius = 0.02;
      PlottingUtils::draw_circle(pair.second(0), pair.second(1), radius, "b",
                                 10);
      const Eigen::VectorXf config = pair.second;
      if (adjacency.find(pair.first) != adjacency.end()) {
        for (const auto& other_id : adjacency.at(pair.first)) {
          // std::cout << pair.first << "," << other_id << std::endl;
          const Eigen::VectorXf other_config = id_to_config.at(other_id);
          plt::plot({config(0), other_config(0)}, {config(1), other_config(1)},
                    {{"color", "k"}, {"linewidth", "1.0"}});
        }
      }
    }

    for (const auto& c : triggered_voxel_locations) {
      PlottingUtils::draw_circle(c[0], c[1], options.GetOfflineVoxelRadius(),
                                 "m", 100);
    }

    for (const auto& edge : drm_planner.forbidden_edge_set_) {
      Eigen::VectorXf config;
      int32_t id = edge.at(0);
      int32_t other_id = edge.at(1);
      if (id == -1) {
        config = start_configuration;
      } else if (id == -2) {
        config = target_configuration;
      } else {
        config = id_to_config.at(id);
      }

      Eigen::VectorXf other_config;

      if (other_id == -1) {
        other_config = start_configuration;
      } else if (other_id == -2) {
        other_config = target_configuration;
      } else {
        other_config = id_to_config.at(other_id);
      }
      plt::plot({config(0), other_config(0)}, {config(1), other_config(1)},
                {{"color", "r"}, {"linewidth", "1.0"}});
    }

    for (const auto& id : drm_planner.collision_set_) {
      PlottingUtils::draw_circle(id_to_config.at(id)(0), id_to_config.at(id)(1),
                                 voxel_radius, "r");
    }

    // plot path
    if (robot_joint_path.size()) {
      for (int idx = 0; idx < robot_joint_path.size() - 1; ++idx) {
        plt::plot(
            {robot_joint_path.at(idx)(0), robot_joint_path.at(idx + 1)(0)},
            {robot_joint_path.at(idx)(1), robot_joint_path.at(idx + 1)(1)},
            {{"color", "g"}, {"linewidth", "5.0"}});
      }
    }

    for (size_t vox_id = 0; vox_id < root_p_voxels.cols(); ++vox_id) {
      PlottingUtils::draw_circle(root_p_voxels(0, vox_id),
                                 root_p_voxels(1, vox_id), voxel_radius, "k");
    }
    cleanupPlotAndPauseForUser(1.9);
  }
}

GTEST_TEST(DRMTest2, Kinova) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/simple_kinova_sens_on_table.yml";
  URDFParser parser;
  parser.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser.parseDirectives(tmp));
  Plant plant = parser.buildPlant();
  RoadmapOptions options;
  options.max_task_space_distance_between_nodes = 10.6;
  options.max_configuration_distance_between_nodes = 10.9;
  options.nodes_processed_before_debug_statement = 500;
  options.robot_map_size_x = 1.5;
  options.robot_map_size_y = 2;
  options.robot_map_size_z = 1;
  options.offline_voxel_resolution = 0.02;
  options.edge_step_size = 0.1;
  RoadmapBuilder roadmap_builder(plant, "kinova::end_effector_link", options);
  HPolyhedron domain;
  domain.MakeBox(plant.getPositionLowerLimits(),
                 plant.getPositionUpperLimits());
  std::vector<Eigen::MatrixXf> samples = UniformSampleInHPolyhedronCuda(
      {domain}, domain.ChebyshevCenter(), 100, 100);

  auto start = std::chrono::high_resolution_clock::now();

  int num_added = roadmap_builder.AddNodesManual(samples.at(0));
  roadmap_builder.BuildRoadmap(10);
  roadmap_builder.BuildPoseMap();
  roadmap_builder.BuildCollisionMap();
  auto stop = std::chrono::high_resolution_clock::now();
  tmp = fmt::format(
      "Roadmap build time: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}