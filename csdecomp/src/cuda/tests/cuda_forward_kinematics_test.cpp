// csdecomp/tests/urdf_parser_test.cpp
#include "cuda_forward_kinematics.h"

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "kinematic_tree.h"
#include "minimal_kinematic_tree.h"
#include "urdf_parser.h"

using namespace csdecomp;

class MinimalKinematicTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  std::string urdf_content;
  std::string prefix = "csdecomp/tests/test_assets/";
  URDFParser parser;
};

TEST_F(MinimalKinematicTreeTest, SingleConfigPrismaticTransfromsTest) {
  std::string tmp = prefix + "3dofrobot_prismatic.urdf";
  Eigen::Matrix4f tf_expected, tf_actual, tf_minimal;
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  MinimalKinematicTree mtree = tree.getMyMinimalKinematicTreeStruct();
  Eigen::MatrixXf transforms_minimal =
      Eigen::MatrixXf::Zero(4 * mtree.num_links, 4);

  Eigen::MatrixXf transforms_cuda =
      Eigen::MatrixXf::Zero(4, 4 * mtree.num_links);

  //   const auto &joints = tree.getJoints();
  EXPECT_TRUE(tree.isFinalized());
  Eigen::VectorXf q(3);
  // clang-format off
    q << 1, 1, 1;

  Eigen::MatrixXf q_cuda(3,1);
  q_cuda<<1,
          1,
          1;
  // clang-format on
  auto transforms = tree.computeLinkFrameToWorldTransforms(q);
  computeLinkFrameToWorldTransformsMinimal(q, mtree, &transforms_minimal);
  computeForwardKinematicsCuda(&transforms_cuda, &q_cuda, &mtree);
  std::cout << "transforms cuda \n";
  std::cout << transforms_cuda << std::endl;
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
  Eigen::MatrixXf tf_cuda =
      transforms_cuda.block(0, 4 * tree.getLinkIndexByName("link3"), 4, 4);
  EXPECT_TRUE(tf_actual.isApprox(tf_expected));
  EXPECT_TRUE(tf_minimal.isApprox(tf_expected));
  EXPECT_TRUE(tf_cuda.isApprox(tf_expected));
}

TEST_F(MinimalKinematicTreeTest, ManyConfigsPrismaticTransfromsTest) {
  std::string tmp = prefix + "3dofrobot_prismatic.urdf";
  const int num_configs = 10000;
  Eigen::Matrix4f tf_expected, tf_actual, tf_minimal;
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  MinimalKinematicTree mtree = tree.getMyMinimalKinematicTreeStruct();
  Eigen::MatrixXf transforms_minimal =
      Eigen::MatrixXf::Zero(4 * mtree.num_links, 4);

  Eigen::MatrixXf transforms_cuda =
      Eigen::MatrixXf::Zero(4, 4 * mtree.num_links * num_configs);

  //   const auto &joints = tree.getJoints();
  EXPECT_TRUE(tree.isFinalized());

  // Set up the random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Create a custom random number generator function
  auto random_generator = [&]() { return dis(gen); };

  Eigen::MatrixXf q_cuda =
      Eigen::MatrixXf::NullaryExpr(3, num_configs, random_generator);

  auto start = std::chrono::high_resolution_clock::now();
  computeForwardKinematicsCuda(&transforms_cuda, &q_cuda, &mtree);
  auto stop = std::chrono::high_resolution_clock::now();

  tmp = fmt::format(
      "execution time cuda: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());

  std::cout << tmp << std::endl;
  // std::cout << "transforms cuda \n";
  // std::cout << transforms_cuda << std::endl;
  for (int conf = 0; conf < q_cuda.cols(); ++conf) {
    // check link3
    auto transforms = tree.computeLinkFrameToWorldTransforms(q_cuda.col(conf));
    tf_actual = transforms.at(tree.getLinkIndexByName("link3"));

    Eigen::MatrixXf tf_cuda = transforms_cuda.block(
        0, 4 * tree.getLinkIndexByName("link3") + 4 * conf * mtree.num_links, 4,
        4);
    EXPECT_TRUE(tf_cuda.isApprox(tf_actual));
  }
}

TEST_F(MinimalKinematicTreeTest, ManyConfigs2Branch5Dof) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  const int num_configs = 10000;
  Eigen::Matrix4f tf_expected, tf_actual, tf_minimal;
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  MinimalKinematicTree mtree = tree.getMyMinimalKinematicTreeStruct();
  Eigen::MatrixXf transforms_minimal =
      Eigen::MatrixXf::Zero(4 * mtree.num_links, 4);

  Eigen::MatrixXf transforms_cuda =
      Eigen::MatrixXf::Zero(4, 4 * mtree.num_links * num_configs);

  //   const auto &joints = tree.getJoints();
  EXPECT_TRUE(tree.isFinalized());

  // Set up the random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Create a custom random number generator function
  auto random_generator = [&]() { return dis(gen); };

  Eigen::MatrixXf q_cuda =
      Eigen::MatrixXf::NullaryExpr(5, num_configs, random_generator);

  auto start = std::chrono::high_resolution_clock::now();
  computeForwardKinematicsCuda(&transforms_cuda, &q_cuda, &mtree);
  auto stop = std::chrono::high_resolution_clock::now();

  tmp = fmt::format(
      "execution time cuda: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count());

  std::cout << tmp << std::endl;
  // std::cout << "transforms cuda \n";
  // std::cout << transforms_cuda << std::endl;
  for (int conf = 0; conf < q_cuda.cols(); ++conf) {
    // check link3
    auto transforms = tree.computeLinkFrameToWorldTransforms(q_cuda.col(conf));
    tf_actual = transforms.at(tree.getLinkIndexByName("link3"));

    Eigen::MatrixXf tf_cuda = transforms_cuda.block(
        0, 4 * tree.getLinkIndexByName("link3") + 4 * conf * mtree.num_links, 4,
        4);
    EXPECT_TRUE(tf_cuda.isApprox(tf_actual));

    tf_actual = transforms.at(tree.getLinkIndexByName("link6"));

    tf_cuda = transforms_cuda.block(
        0, 4 * tree.getLinkIndexByName("link6") + 4 * conf * mtree.num_links, 4,
        4);
    EXPECT_TRUE(tf_cuda.isApprox(tf_actual));
  }
}