// csdecomp/tests/urdf_parser_test.cpp
#include "minimal_kinematic_tree.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>

#include "kinematic_tree.h"
#include "urdf_parser.h"

using namespace csdecomp;

class MinimalKinematicTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  std::string urdf_content;
  std::string prefix = "csdecomp/tests/test_assets/";
  URDFParser parser;
};

TEST_F(MinimalKinematicTreeTest, TransfromsTest) {
  MinimalJoint test_joint;
  MinimalLink test_link;
  test_joint.index = 1;
  test_link.index = 1231;
  EXPECT_EQ(test_joint.index, 1);
  EXPECT_EQ(test_link.index, 1231);

  std::string tmp = prefix + "2_branch_5dof.urdf";
  Eigen::Matrix4f tf_expected, tf_actual, tf_minimal;
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  MinimalKinematicTree mtree = tree.getMyMinimalKinematicTreeStruct();
  Eigen::MatrixXf transforms_minimal =
      Eigen::MatrixXf::Zero(4 * mtree.num_links, 4);

  //   const auto &joints = tree.getJoints();
  EXPECT_TRUE(tree.isFinalized());
  Eigen::VectorXf q(5);
  q << 1, 1, 1, 1, 1;

  auto transforms = tree.computeLinkFrameToWorldTransforms(q);
  computeLinkFrameToWorldTransformsMinimal(q, mtree, &transforms_minimal);

  // world
  tf_expected << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  tf_actual = transforms.at(tree.getLinkIndexByName("world"));
  tf_minimal =
      transforms_minimal.block(4 * tree.getLinkIndexByName("world"), 0, 4, 4);
  EXPECT_TRUE(tf_actual.isApprox(tf_expected));
  EXPECT_TRUE(tf_minimal.isApprox(tf_actual));

  // link1
  // clang-format off
    tf_expected << 0.540302, -0.841471, 0, 0,
                   0.841471,  0.540302, 0, 0,
                          0,         0, 1, 0,
                          0,         0, 0, 1;
  // clang-format on
  tf_actual = transforms.at(tree.getLinkIndexByName("link1"));
  tf_minimal =
      transforms_minimal.block(4 * tree.getLinkIndexByName("link1"), 0, 4, 4);
  EXPECT_TRUE(tf_actual.isApprox(tf_expected));
  EXPECT_TRUE(tf_minimal.isApprox(tf_actual));

  // link2
  // clang-format off
    tf_expected << -0.41614684, -0.90929743, 0.0, 0.54030231,
                    0.90929743, -0.41614684, 0.0, 0.84147098,
                    0.0,         0.0,        1.0, 0.0,
                    0.0,         0.0,        0.0, 1.0;
  // clang-format on
  tf_actual = transforms.at(tree.getLinkIndexByName("link2"));
  tf_minimal =
      transforms_minimal.block(4 * tree.getLinkIndexByName("link2"), 0, 4, 4);
  EXPECT_TRUE(tf_actual.isApprox(tf_expected));
  EXPECT_TRUE(tf_minimal.isApprox(tf_actual));
  // link6
  // clang-format off
    tf_expected << -0.9899925, -0.14112001, 0.0, -1.0596646,
                    0.14112001, -0.9899925, 0.0, 0.58067513,
                    0.0,         0.0,       1.0, 0.0,
                    0.0,         0.0,       0.0, 1.0;
  // clang-format on
  tf_actual = transforms.at(tree.getLinkIndexByName("link6"));
  tf_minimal =
      transforms_minimal.block(4 * tree.getLinkIndexByName("link6"), 0, 4, 4);
  EXPECT_TRUE(tf_actual.isApprox(tf_actual));
  EXPECT_TRUE(tf_minimal.isApprox(tf_actual));
}

TEST_F(MinimalKinematicTreeTest, PrismaticTransfromsTest) {
  std::string tmp = prefix + "3dofrobot_prismatic.urdf";
  Eigen::Matrix4f tf_expected, tf_actual, tf_minimal;
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  MinimalKinematicTree mtree = tree.getMyMinimalKinematicTreeStruct();
  Eigen::MatrixXf transforms_minimal =
      Eigen::MatrixXf::Zero(4 * mtree.num_links, 4);

  //   const auto &joints = tree.getJoints();
  EXPECT_TRUE(tree.isFinalized());
  Eigen::VectorXf q(3);
  q << 1, 1, 1;

  auto transforms = tree.computeLinkFrameToWorldTransforms(q);
  computeLinkFrameToWorldTransformsMinimal(q, mtree, &transforms_minimal);
  // link3
  // clang-format off
    tf_expected << 0.60952558,  0.17919833,  0.77224771,  1.75935346,
                   0.59500984,  0.54030231, -0.59500984,  1.19001968,
                  -0.52387199,  0.8221687 ,  0.22270331, -1.88921496,
                   0.0,         0.0,         0.0,         1.0;
  // clang-format on
  tf_actual = transforms.at(tree.getLinkIndexByName("link3"));
  tf_minimal =
      transforms_minimal.block(4 * tree.getLinkIndexByName("link3"), 0, 4, 4);
  EXPECT_TRUE(tf_actual.isApprox(tf_expected));
  EXPECT_TRUE(tf_minimal.isApprox(tf_actual));
}