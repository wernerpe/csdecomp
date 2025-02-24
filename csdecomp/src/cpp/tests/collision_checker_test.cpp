#include "collision_checker.h"

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "kinematic_tree.h"
#include "urdf_parser.h"
using namespace csdecomp;

class CPPCollisionCheckerTest : public ::testing::Test {
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

TEST_F(CPPCollisionCheckerTest, BoxBoxCollisionCheckerTest) {
  std::string tmp = prefix + "3dofrobot.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::Vector3f q;
  q << 0, 1.8, 2.46;

  EXPECT_TRUE(checkCollisionFree(q, plant));
  q << 0, 1.8, 2.47;

  EXPECT_FALSE(checkCollisionFree(q, plant));
}

TEST_F(CPPCollisionCheckerTest, SphereBoxCollisionCheckerTest) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();
  Eigen::VectorXf q(5);
  q << 0.6714, -0.9096, 0.79, -0.1806, 0.;

  EXPECT_TRUE(checkCollisionFree(q, plant));
  q << 0.6714, -0.9096, 0.78, -0.1806, 0.;

  EXPECT_FALSE(checkCollisionFree(q, plant));
  q << 0.0204, 0.3714, 0.4714, 1.0344, -1.2816;

  EXPECT_TRUE(checkCollisionFree(q, plant));
  q << 0.0204, 0.3714, 0.4714, 1.0874, -1.2816;

  EXPECT_FALSE(checkCollisionFree(q, plant));
}

TEST_F(CPPCollisionCheckerTest, SphereSphereCollisionCheckerTest) {
  std::string tmp = prefix + "3dofrobot_spheres.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::VectorXf q(3);
  q << 0, 0, 2.67;

  EXPECT_TRUE(checkCollisionFree(q, plant));
  q << 0, 0, 2.68;
  EXPECT_FALSE(checkCollisionFree(q, plant));
}

TEST_F(CPPCollisionCheckerTest, MixedCollisionTest) {
  std::string tmp = prefix + "movable_block.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  MinimalPlant plant = parser.getMinimalPlant();

  Eigen::MatrixXf qfree(5, 6);
  Eigen::MatrixXf qcol(5, 6);
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

  for (int col = 0; col < qfree.cols(); ++col) {
    EXPECT_TRUE(checkCollisionFree(qfree.col(col), plant));
  }
  EXPECT_FALSE(checkCollisionFree(qcol.col(0), plant));
  //   for (int col = 0; col < qcol.cols(); ++col) {
  //     EXPECT_FALSE(checkCollisionFree(qcol.col(col), plant));
  //   }
}

// TEST_F(CPPCollisionCheckerTest, CapsuleCollisionTest) {
//   std::string tmp = prefix + "movable_capsule.urdf";
//   EXPECT_TRUE(parser.parseURDF(tmp));

//   MinimalPlant plant = parser.getMinimalPlant();

//   Eigen::MatrixXf qfree(6, 4);
//   Eigen::MatrixXf qcol(6, 4);
//   // clang-format off
//   qfree <<  0.2,   0.2,  -0.2,  -0.2,
//             0.1,   0.1,  -0.0,  -0.0,
//            -0.0,  -0.0,  -0.0,  -0.0,
//             0.5,   0.5,   0.5,   0.5,
//            -0.3,   1.9,   1.9,   3.1,
//             0.5,   0.5,   0.5,   0.5;

//   qcol <<  -0.5,   0.5,   0.5,   0.1,
//            -0.0,  -0.1,   0.5,   0.0,
//             0.1,   0.1,   0.2,   0.4,
//             0.0,   0.0,   0.0,  -0.3,
//             0.7,   0.7,   0.7,   0.1,
//             0.5,  -0.8,  -0.8,  -1.1;
//   // clang-format on

//   for (int col = 0; col < qfree.cols(); ++col) {
//     EXPECT_TRUE(checkCollisionFree(qfree.col(col), plant));
//   }
//   for (int col = 0; col < qcol.cols(); ++col) {
//     EXPECT_FALSE(checkCollisionFree(qcol.col(col), plant));
//   }
// }

TEST_F(CPPCollisionCheckerTest, CheckCollisionsAgainstVoxels) {
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

  for (int col = 0; col < qfree.cols(); ++col) {
    EXPECT_TRUE(checkCollisionFreeVoxels(qfree.col(col), voxels, vox_radius,
                                         plant, robot_geometry_ids));
  }
  for (int col = 0; col < qcol.cols(); ++col) {
    EXPECT_FALSE(checkCollisionFreeVoxels(qcol.col(col), voxels, vox_radius,
                                          plant, robot_geometry_ids));
  }
}

GTEST_TEST(ColcheckerTest, Kinova) {
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

  Eigen::VectorXf l_start(7);
  l_start << 0, 0, 0, 0, 0, 0, 0;
  EXPECT_TRUE(checkCollisionFree(l_start, mp));
  l_start << -2.4934235, -1.6654704, -1.9375235, 1.9090002, 0.09470118,
      1.8022158, 1.6754638;
  EXPECT_TRUE(checkCollisionFree(l_start, mp));
  l_start << -2.4934235, -1.6654704, -1.9375235, 1.9090002, 0.09470118, 1.8349,
      1.6754638;
  EXPECT_TRUE(checkCollisionFree(l_start, mp));
  l_start << -2.4934235, -1.6654704, -1.9375235, 1.9090002, 0.09470118, 1.8679,
      1.6754638;
  EXPECT_FALSE(checkCollisionFree(l_start, mp));
}