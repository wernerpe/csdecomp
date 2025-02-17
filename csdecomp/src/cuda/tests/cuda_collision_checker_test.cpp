#include "cuda_collision_checker.h"

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "collision_checker.h"
#include "kinematic_tree.h"
#include "urdf_parser.h"

using namespace csdecomp;

class CudaColCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    // Remove the temporary file
    std::string tmp = prefix + "temp_test.urdf";
    std::remove(tmp.c_str());
  }

  std::string urdf_content;
  std::string prefix = "csdecomp/tests/test_assets/";
  URDFParser parser;
};

TEST_F(CudaColCheckTest, SphereBoxCollisionCheckerTest) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::MatrixXf configurations(5, 4);
  // clang-format off
    configurations << 0.6714,  0.6714,  0.0204,  0.0204,
                     -0.9096, -0.9096,  0.3714,  0.3714,
                      0.79,    0.78,    0.4714,  0.4714,
                     -0.1806, -0.1806,  1.0344,  1.0874,
                      0.,      0.,     -1.2816, -1.2816;
  // clang-format on
  EXPECT_TRUE(checkCollisionFree(configurations.col(0), plant));
  EXPECT_FALSE(checkCollisionFree(configurations.col(1), plant));
  EXPECT_TRUE(checkCollisionFree(configurations.col(2), plant));
  EXPECT_FALSE(checkCollisionFree(configurations.col(3), plant));

  std::vector<uint8_t> col_free =
      checkCollisionFreeCuda(&configurations, &plant);
  EXPECT_TRUE(col_free.at(0));
  EXPECT_FALSE(col_free.at(1));
  EXPECT_TRUE(col_free.at(2));
  EXPECT_FALSE(col_free.at(3));
}

TEST_F(CudaColCheckTest, SphereSphereCollisionCheckerTest) {
  std::string tmp = prefix + "3dofrobot_spheres.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  SceneInspector insp = parser.getSceneInspector();
  CollisionPairMatrix cpm = insp.getCollisionPairMatrix();
  std::cout << "cpm true\n" << cpm << std::endl;
  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::MatrixXf configurations(3, 2);
  // clang-format off
    configurations<< 0,    0, 
                     0,    0, 
                     2.67, 2.68;
  // clang-format on 
  std::vector<uint8_t> col_free = checkCollisionFreeCuda(
      &configurations, &plant);
  int idx =0;
  for(auto r: col_free){
    bool in_collision_expected = checkCollisionFree(configurations.col(idx), plant);
    //check collision uses flipped sign
    EXPECT_EQ(r, in_collision_expected);
    ++idx;
  }
}

TEST_F(CudaColCheckTest, MixedCollisionTest) {

  std::string tmp = prefix + "movable_block.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));
  
  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::MatrixXf qfree(5,6);
  Eigen::MatrixXf qcol(5,6);
  // clang-format off
  qfree << -0.375,  -0.375,  0.004,   0.004,  -0.003,  -0.002,
           -0.622,  -0.416,  0.002,   0.04 ,   0.088,   0.303,
           -0.099,  -0.099,  0.0  ,   0.008,   0.008,   0.692,
            1.0224, -0.3276, 0.0014, -0.0056, -0.8796,  0.0564,
            0.7724,  0.7724, 0.0024,  0.0024, -0.7956, -0.7956;
  
  qcol  << -0.375,  -0.375,  -0.002,  -0.002,   0.237,   0.591,
           -0.461,  -0.416,   0.088,   0.088,   0.202,   0.318,
           -0.099,  -0.099,   0.008,   0.008,   0.592,   0.449,
            1.0224, -0.8546, -0.8796,  0.0564, -0.0256, -0.0256,
            0.7724,  0.7974, -0.7956, -0.7956, -0.7956, -0.0116;
  // clang-format on
  std::vector<uint8_t> results_qfree = checkCollisionFreeCuda(&qfree, &plant);

  for (int col = 0; col < qfree.cols(); ++col) {
    EXPECT_TRUE(checkCollisionFree(qfree.col(col), plant));
    EXPECT_TRUE(results_qfree.at(col));
  }

  std::vector<uint8_t> results_qcol = checkCollisionFreeCuda(&qcol, &plant);

  for (int col = 0; col < qcol.cols(); ++col) {
    EXPECT_FALSE(checkCollisionFree(qcol.col(col), plant));
    EXPECT_FALSE(results_qcol.at(col));
  }
}

TEST_F(CudaColCheckTest, ManyMixedCollisionTest) {
  int num_configs = 50000;
  std::string tmp = prefix + "movable_block.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Create a custom random number generator function
  auto random_generator = [&]() { return dis(gen); };

  Eigen::MatrixXf q_cuda =
      Eigen::MatrixXf::NullaryExpr(5, num_configs, random_generator);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> results = checkCollisionFreeCuda(&q_cuda, &plant);
  auto stop = std::chrono::high_resolution_clock::now();

  tmp = fmt::format(
      "execution time cuda: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  for (int col = 0; col < std::min(50, static_cast<int>(q_cuda.cols()));
       ++col) {
    EXPECT_EQ(checkCollisionFree(q_cuda.col(col), plant), results.at(col));
  }
}

TEST_F(CudaColCheckTest, ManyMixedCollisionMoreGeomsTest) {
  int num_configs = 1000000;
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Create a custom random number generator function
  auto random_generator = [&]() { return dis(gen); };

  Eigen::MatrixXf q_cuda =
      Eigen::MatrixXf::NullaryExpr(5, num_configs, random_generator);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> results = checkCollisionFreeCuda(&q_cuda, &plant);
  auto stop = std::chrono::high_resolution_clock::now();

  tmp = fmt::format(
      "execution time cuda: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  // verify the first 1000 results
  for (int col = 0; col < std::min(50, static_cast<int>(q_cuda.cols()));
       ++col) {
    EXPECT_EQ(checkCollisionFree(q_cuda.col(col), plant), results.at(col));
  }
}

TEST_F(CudaColCheckTest, ThrowOnBigBatchSize) {
  int num_configs = 20000000;
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::MatrixXf q_cuda = Eigen::MatrixXf::Zero(5, num_configs);

  EXPECT_THROW(checkCollisionFreeCuda(&q_cuda, &plant), std::runtime_error);
}

TEST_F(CudaColCheckTest, CheckCollisionsAgainstVoxels) {
  std::string tmp = prefix + "movable_block.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();
  auto inspector = parser.getSceneInspector();
  auto robot_geometry_ids = inspector.robot_geometry_ids;

  // Generate voxels
  int nx = 6;
  int ny = 6;
  int nz = 6;
  float spx = 0.3f;
  float spy = 0.3f;
  float spz = 0.1f;
  Eigen::Vector3f offset;
  offset << -(nx - 1) * spx / 2, -(ny - 1) * spy / 2, 0;
  float vox_radius =
      0.02f;  // Note: This is declared but not used in the given code

  // Assuming Voxels is a typedef for Eigen::MatrixXf
  Voxels voxels(3, nx * ny * nz);

  int index = 0;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        Eigen::Vector3f position;
        position << ix * spx, iy * spy, iz * spz;
        voxels.col(index) = position + offset;
        ++index;
      }
    }
  }
  Eigen::MatrixXf qfree(5, 3);
  Eigen::MatrixXf qcol(5, 5);
  // clang-format off
  qfree << 0.00,  0.00,  0.00,
           0.00,  0.00,  0.00,
           0.00,  0.00,  0.27,
           0.00,  0.00,  0.57,
           0.00,  1.50,  1.27; 
  
   qcol << 0.15,  0.00,  0.00,   0.00,  0.16,
           0.00,  0.15,  0.48,   0.48,  0.28,
           0.00,  0.00,  0.00,   0.31,  0.06,
           0.00,  0.87, -0.43,  -0.43,  1.46,
           0.00,  1.17,  1.17,   1.17,  1.59;
  // clang-format on
  std::vector<uint8_t> results_qfree = checkCollisionFreeVoxelsCuda(
      &qfree, &voxels, vox_radius, &plant, robot_geometry_ids);

  for (auto r : results_qfree) {
    EXPECT_TRUE(r);
  }
  std::vector<uint8_t> results_qcol = checkCollisionFreeVoxelsCuda(
      &qcol, &voxels, vox_radius, &plant, robot_geometry_ids);

  for (auto r : results_qcol) {
    EXPECT_FALSE(r);
  }
}

TEST_F(CudaColCheckTest, CompareRandomConfigsVsCPU) {
  int num_configs = 100;
  std::string tmp = prefix + "movable_block.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();
  auto inspector = parser.getSceneInspector();
  auto robot_geometry_ids = inspector.robot_geometry_ids;

  // Generate voxels
  int nx = 10;
  int ny = 10;
  int nz = 10;
  float spx = 0.3f;
  float spy = 0.3f;
  float spz = 0.1f;
  Eigen::Vector3f offset;
  offset << -(nx - 1) * spx / 2, -(ny - 1) * spy / 2, 0;
  float vox_radius =
      0.02f;  // Note: This is declared but not used in the given code

  // Assuming Voxels is a typedef for Eigen::MatrixXf
  Voxels voxels(3, nx * ny * nz);

  int index = 0;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        Eigen::Vector3f position;
        position << ix * spx, iy * spy, iz * spz;
        voxels.col(index) = position + offset;
        ++index;
      }
    }
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Create a custom random number generator function
  auto random_generator = [&]() { return dis(gen); };

  Eigen::MatrixXf q_cuda =
      Eigen::MatrixXf::NullaryExpr(5, num_configs, random_generator);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> results = checkCollisionFreeVoxelsCuda(
      &q_cuda, &voxels, vox_radius, &plant, robot_geometry_ids);
  auto stop = std::chrono::high_resolution_clock::now();
  tmp = fmt::format(
      "execution time cuda: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());
  std::cout << tmp << std::endl;
  for (int col = 0; col < q_cuda.cols(); ++col) {
    EXPECT_EQ(results.at(col),
              checkCollisionFreeVoxels(q_cuda.col(col), voxels, vox_radius,
                                       plant, robot_geometry_ids));
    if (results.at(col)) {
      std::cout << "free!\n";
    } else {
      std::cout << "col!\n";
    }
  }
}

GTEST_TEST(OtherCudaColcheckerTest, Kinova) {
  std::string prefix = "csdecomp/tests/test_assets/";
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  URDFParser parser;
  parser.registerPackage("test_assets", prefix);
  EXPECT_TRUE(parser.parseDirectives(tmp));
  KinematicTree kt = parser.getKinematicTree();
  MinimalPlant mp = parser.getMinimalPlant();
  SceneInspector inspector = parser.getSceneInspector();
  const std::vector<CollisionGeometry> geoms =
      parser.getSceneCollisionGeometries();

  for (auto i : {7, 9}) {
    CollisionGeometry g = geoms.at(i);
    std::cout << kt.getLinkNameByIndex(g.link_index) << std::endl;
  }

  Eigen::MatrixXf l_start(7, 1);
  l_start << 0, 0, 0, 0, 0, 0, 0;
  std::vector<uint8_t> results;
  results = checkCollisionFreeCuda(&l_start, &mp);
  EXPECT_TRUE(results.at(0));
  l_start << -2.4934235, -1.6654704, -1.9375235, 1.9090002, 0.09470118,
      1.8022158, 1.6754638;
  results = checkCollisionFreeCuda(&l_start, &mp);
  EXPECT_TRUE(results.at(0));
  l_start << -2.4934235, -1.6654704, -1.9375235, 1.9090002, 0.09470118, 1.8349,
      1.6754638;
  results = checkCollisionFreeCuda(&l_start, &mp);
  EXPECT_TRUE(results.at(0));
  l_start << -2.4934235, -1.6654704, -1.9375235, 1.9090002, 0.09470118, 1.8679,
      1.6754638;
  results = checkCollisionFreeCuda(&l_start, &mp);
  EXPECT_FALSE(results.at(0));
}