#include "linesegment_aabb_checker.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

using namespace csdecomp;

// Test fixture for LinesegmentAABBIntersecting tests
class LinesegmentAABBIntersectingTest : public ::testing::Test {
 protected:
  // Set up common test data
  void SetUp() override {
    // 2D test data
    box2d_min = Eigen::Vector2d(0, 0);
    box2d_max = Eigen::Vector2d(1, 1);

    // 3D test data
    box3d_min = Eigen::Vector3d(0, 0, 0);
    box3d_max = Eigen::Vector3d(1, 1, 1);
  }

  // 2D test data
  Eigen::Vector2d box2d_min;
  Eigen::Vector2d box2d_max;

  // 3D test data
  Eigen::Vector3d box3d_min;
  Eigen::Vector3d box3d_max;
};

// Test cases for 2D line segments
TEST_F(LinesegmentAABBIntersectingTest, LineCompletelyInside2D) {
  Eigen::Vector2d p1(0.25, 0.25);
  Eigen::Vector2d p2(0.75, 0.75);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineIntersecting2D) {
  Eigen::Vector2d p1(-0.5, -0.5);
  Eigen::Vector2d p2(0.5, 0.5);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineCompletelyOutside2D) {
  Eigen::Vector2d p1(-2, -2);
  Eigen::Vector2d p2(-1, -1);

  EXPECT_FALSE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineTouchingCorner2D) {
  Eigen::Vector2d p1(-1, -1);
  Eigen::Vector2d p2(0, 0);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineParallelToEdge2D) {
  // Line parallel to x-axis at y=0.5, going from x=-1 to x=2
  Eigen::Vector2d p1(-1, 0.5);
  Eigen::Vector2d p2(2, 0.5);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));

  // Line parallel to x-axis at y=2, going from x=-1 to x=2 (outside box)
  p1 = Eigen::Vector2d(-1, 2);
  p2 = Eigen::Vector2d(2, 2);

  EXPECT_FALSE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));
}

// Test cases for 3D line segments
TEST_F(LinesegmentAABBIntersectingTest, LineCompletelyInside3D) {
  Eigen::Vector3d p1(0.25, 0.25, 0.25);
  Eigen::Vector3d p2(0.75, 0.75, 0.75);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box3d_min, box3d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineIntersecting3D) {
  Eigen::Vector3d p1(-0.5, -0.5, -0.5);
  Eigen::Vector3d p2(0.5, 0.5, 0.5);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box3d_min, box3d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineCompletelyOutside3D) {
  Eigen::Vector3d p1(-2, -2, -2);
  Eigen::Vector3d p2(-1, -1, -1);

  EXPECT_FALSE(LinesegmentAABBIntersecting(p1, p2, box3d_min, box3d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineTouchingCorner3D) {
  Eigen::Vector3d p1(-1, -1, -1);
  Eigen::Vector3d p2(0, 0, 0);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box3d_min, box3d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineThroughFace3D) {
  // Line going through the x=0.5 face
  Eigen::Vector3d p1(0.5, -1, 0.5);
  Eigen::Vector3d p2(0.5, 2, 0.5);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box3d_min, box3d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, LineAlongEdge3D) {
  // Line along the x=0, y=0 edge
  Eigen::Vector3d p1(0, 0, -1);
  Eigen::Vector3d p2(0, 0, 2);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box3d_min, box3d_max));
}

// Edge cases
TEST_F(LinesegmentAABBIntersectingTest, ZeroLengthLine) {
  // Point inside the box
  Eigen::Vector2d p(0.5, 0.5);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p, p, box2d_min, box2d_max));

  // Point outside the box
  p = Eigen::Vector2d(2, 2);

  EXPECT_FALSE(LinesegmentAABBIntersecting(p, p, box2d_min, box2d_max));
}

TEST_F(LinesegmentAABBIntersectingTest, NumericalStability) {
  // Test with points very close to boundary
  Eigen::Vector2d p1(0, 1e-10);
  Eigen::Vector2d p2(1, 1e-10);

  EXPECT_TRUE(LinesegmentAABBIntersecting(p1, p2, box2d_min, box2d_max));

  // Test with very small box
  Eigen::Vector2d small_box_min(0, 0);
  Eigen::Vector2d small_box_max(1e-10, 1e-10);
  Eigen::Vector2d p3(0, 0);
  Eigen::Vector2d p4(1e-11, 1e-11);

  EXPECT_TRUE(
      LinesegmentAABBIntersecting(p3, p4, small_box_min, small_box_max));
}

// Main function that runs all the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}