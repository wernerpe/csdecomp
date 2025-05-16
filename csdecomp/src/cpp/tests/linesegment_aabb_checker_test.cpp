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

TEST_F(LinesegmentAABBIntersectingTest, MultipleBoxes2D) {
  // Create line segment
  Eigen::Vector2d p1(0.5, -2);
  Eigen::Vector2d p2(0.5, 3);

  // Create multiple boxes
  int num_boxes = 3;
  Eigen::MatrixXd boxes_min(2, num_boxes);
  Eigen::MatrixXd boxes_max(2, num_boxes);

  // Box 1: Unit box at origin (should intersect)
  boxes_min.col(0) = Eigen::Vector2d(0, 0);
  boxes_max.col(0) = Eigen::Vector2d(1, 1);

  // Box 2: Box at (2,2) (should not intersect)
  boxes_min.col(1) = Eigen::Vector2d(2, 2);
  boxes_max.col(1) = Eigen::Vector2d(3, 3);

  // Box 3: Box at (-1, -1) but still intersecting the line
  boxes_min.col(2) = Eigen::Vector2d(-1, -1);
  boxes_max.col(2) = Eigen::Vector2d(1, 0);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 1);  // Box 1 should intersect
  EXPECT_EQ(results[1], 0);  // Box 2 should not intersect
  EXPECT_EQ(results[2], 1);  // Box 3 should intersect
}

TEST_F(LinesegmentAABBIntersectingTest, MultipleBoxes2D_AllIntersect) {
  // Create a horizontal line segment
  Eigen::Vector2d p1(-2, 0.5);
  Eigen::Vector2d p2(4, 0.5);

  // Create multiple boxes that all intersect the line
  int num_boxes = 4;
  Eigen::MatrixXd boxes_min(2, num_boxes);
  Eigen::MatrixXd boxes_max(2, num_boxes);

  // Four boxes along the x-axis, all intersecting the line
  boxes_min.col(0) = Eigen::Vector2d(-3, 0);
  boxes_max.col(0) = Eigen::Vector2d(-1, 1);

  boxes_min.col(1) = Eigen::Vector2d(-1, 0);
  boxes_max.col(1) = Eigen::Vector2d(1, 1);

  boxes_min.col(2) = Eigen::Vector2d(1, 0);
  boxes_max.col(2) = Eigen::Vector2d(3, 1);

  boxes_min.col(3) = Eigen::Vector2d(3, 0);
  boxes_max.col(3) = Eigen::Vector2d(5, 1);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 1);
  EXPECT_EQ(results[1], 1);
  EXPECT_EQ(results[2], 1);
  EXPECT_EQ(results[3], 1);
}

TEST_F(LinesegmentAABBIntersectingTest, singleBox2Dinmulti_NoneIntersect) {
  // Create a line segment
  Eigen::Vector2d p1(-0.5, -0.5);
  Eigen::Vector2d p2(-0.2, -0.2);

  // Create multiple boxes that don't intersect the line
  int num_boxes = 1;
  Eigen::MatrixXd boxes_min(2, num_boxes);
  Eigen::MatrixXd boxes_max(2, num_boxes);

  // Three boxes that don't intersect
  boxes_min.col(0) = Eigen::Vector2d(0, 0);
  boxes_max.col(0) = Eigen::Vector2d(1, 1);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 0);
}

TEST_F(LinesegmentAABBIntersectingTest, MultipleBoxes2D_NoneIntersect) {
  // Create a line segment
  Eigen::Vector2d p1(-0.5, -0.5);
  Eigen::Vector2d p2(-0.2, -0.2);

  // Create multiple boxes that don't intersect the line
  int num_boxes = 3;
  Eigen::MatrixXd boxes_min(2, num_boxes);
  Eigen::MatrixXd boxes_max(2, num_boxes);

  // Three boxes that don't intersect
  boxes_min.col(0) = Eigen::Vector2d(0, 0);
  boxes_max.col(0) = Eigen::Vector2d(1, 1);

  boxes_min.col(1) = Eigen::Vector2d(2, 0);
  boxes_max.col(1) = Eigen::Vector2d(3, 1);

  boxes_min.col(2) = Eigen::Vector2d(0, 2);
  boxes_max.col(2) = Eigen::Vector2d(1, 3);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 0);
  EXPECT_EQ(results[1], 0);
  EXPECT_EQ(results[2], 0);
}

// Test cases for multiple AABBs (3D)
TEST_F(LinesegmentAABBIntersectingTest, MultipleBoxes3D) {
  // Create line segment
  Eigen::Vector3d p1(0.5, 0.5, -2);
  Eigen::Vector3d p2(0.5, 0.5, 3);

  // Create multiple boxes
  int num_boxes = 3;
  Eigen::MatrixXd boxes_min(3, num_boxes);
  Eigen::MatrixXd boxes_max(3, num_boxes);

  // Box 1: Unit box at origin (should intersect)
  boxes_min.col(0) = Eigen::Vector3d(0, 0, 0);
  boxes_max.col(0) = Eigen::Vector3d(1, 1, 1);

  // Box 2: Box away from line (should not intersect)
  boxes_min.col(1) = Eigen::Vector3d(2, 2, 2);
  boxes_max.col(1) = Eigen::Vector3d(3, 3, 3);

  // Box 3: Box along line but past endpoint (should not intersect)
  boxes_min.col(2) = Eigen::Vector3d(0, 0, 4);
  boxes_max.col(2) = Eigen::Vector3d(1, 1, 5);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 1);  // Box 1 should intersect
  EXPECT_EQ(results[1], 0);  // Box 2 should not intersect
  EXPECT_EQ(results[2], 0);  // Box 3 should not intersect
}

TEST_F(LinesegmentAABBIntersectingTest, MultipleBoxes3D_EdgeCases) {
  // Create a zero-length line segment (a point)
  Eigen::Vector3d p1(0.5, 0.5, 0.5);
  Eigen::Vector3d p2(0.5, 0.5, 0.5);

  // Create multiple boxes
  int num_boxes = 3;
  Eigen::MatrixXd boxes_min(3, num_boxes);
  Eigen::MatrixXd boxes_max(3, num_boxes);

  // Box 1: Box containing the point
  boxes_min.col(0) = Eigen::Vector3d(0, 0, 0);
  boxes_max.col(0) = Eigen::Vector3d(1, 1, 1);

  // Box 2: Box with the point exactly on its boundary
  boxes_min.col(1) = Eigen::Vector3d(0.5, 0.5, 0.5);
  boxes_max.col(1) = Eigen::Vector3d(1.5, 1.5, 1.5);

  // Box 3: Box not containing the point
  boxes_min.col(2) = Eigen::Vector3d(2, 2, 2);
  boxes_max.col(2) = Eigen::Vector3d(3, 3, 3);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 1);  // Point inside box
  EXPECT_EQ(results[1], 1);  // Point on boundary
  EXPECT_EQ(results[2], 0);  // Point outside box
}

TEST_F(LinesegmentAABBIntersectingTest, MultipleBoxes3D_NumericalEdgeCases) {
  // Create line segment with very small values
  Eigen::Vector3d p1(1e-10, 1e-10, 1e-10);
  Eigen::Vector3d p2(2e-10, 2e-10, 2e-10);

  // Create multiple boxes
  int num_boxes = 2;
  Eigen::MatrixXd boxes_min(3, num_boxes);
  Eigen::MatrixXd boxes_max(3, num_boxes);

  // Box 1: Tiny box containing the line
  boxes_min.col(0) = Eigen::Vector3d(0, 0, 0);
  boxes_max.col(0) = Eigen::Vector3d(3e-10, 3e-10, 3e-10);

  // Box 2: Box with very close but non-intersecting line
  boxes_min.col(1) = Eigen::Vector3d(3e-10, 3e-10, 3e-10);
  boxes_max.col(1) = Eigen::Vector3d(4e-10, 4e-10, 4e-10);

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);
  EXPECT_EQ(results[0], 1);  // Should intersect with tiny box
  EXPECT_EQ(results[1], 0);  // Should not intersect with nearby box
}

// Test for large number of boxes
TEST_F(LinesegmentAABBIntersectingTest, ManyBoxes2D) {
  // Create line segment
  Eigen::Vector2d p1(0, 0);
  Eigen::Vector2d p2(10, 10);

  // Create a grid of boxes
  int grid_size = 5;  // 5x5 grid = 25 boxes
  int num_boxes = grid_size * grid_size;
  Eigen::MatrixXd boxes_min(2, num_boxes);
  Eigen::MatrixXd boxes_max(2, num_boxes);

  // Create a 5x5 grid of unit boxes
  int box_idx = 0;
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      boxes_min.col(box_idx) = Eigen::Vector2d(i, j);
      boxes_max.col(box_idx) = Eigen::Vector2d(i + 1, j + 1);
      box_idx++;
    }
  }

  std::vector<uint8_t> results =
      LinesegmentAABBsIntersecting(p1, p2, boxes_min, boxes_max);

  EXPECT_EQ(results.size(), num_boxes);

  // The diagonal boxes (0,0)-(1,1), (1,1)-(2,2), ..., (4,4)-(5,5) should
  // intersect
  for (int i = 0; i < grid_size; ++i) {
    EXPECT_EQ(results[i * grid_size + i], 1)
        << "Box at grid position (" << i << "," << i << ") should intersect";
  }
}

// Main function that runs all the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}