#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

namespace csdecomp {

/**
 * Check if a line segment intersects with an axis-aligned bounding box (AABB)
 *
 * @tparam Vector Type of vector (usually Eigen::Vector2d or Eigen::Vector3d)
 * @param p1 First endpoint of the line segment
 * @param p2 Second endpoint of the line segment
 * @param box_min Minimum corner of the AABB
 * @param box_max Maximum corner of the AABB
 * @return true if the line segment intersects with the AABB, false otherwise
 */

bool LinesegmentAABBIntersecting(const Eigen::VectorXd& p1,
                                 const Eigen::VectorXd& p2,
                                 const Eigen::VectorXd& box_min,
                                 const Eigen::VectorXd& box_max);

/**
 * Check if a line segment intersects with multible axis-aligned bounding boxes
 * (AABBs)
 *
 * @tparam Vector Type of vector (usually Eigen::Vector2d or Eigen::Vector3d)
 * @param p1 First endpoint of the line segment
 * @param p2 Second endpoint of the line segment
 * @param box_min (dim, N) Minimum corners of the AABBs
 * @param box_max (dim, N) Maximum corners of the AABBs
 * @return true if the line segment intersects with the AABB, false otherwise
 */

std::vector<uint8_t> LinesegmentAABBsIntersecting(
    const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
    const Eigen::MatrixXd& boxes_min, const Eigen::MatrixXd& boxes_max);

}  // namespace csdecomp
