#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

namespace csdecomp {
/**
 * @brief Check if a point is inside any of the axis-aligned bounding boxes
 * (AABBs).
 *
 * Tests whether the given point lies within at least one of the provided AABBs.
 * The function performs early exit optimization - it returns 1 immediately upon
 * finding the first AABB that contains the point.
 *
 * @param point Point coordinates as a vector of size D (where D is the spatial
 * dimension)
 * @param boxes_min Matrix of shape (D, N) containing minimum corners of N
 * bounding boxes. Each column represents the minimum corner of one AABB.
 * @param boxes_max Matrix of shape (D, N) containing maximum corners of N
 * bounding boxes. Each column represents the maximum corner of one AABB.
 *
 * @return uint8_t Returns 1 if the point is inside any AABB, 0 otherwise
 *
 * @note The point is considered inside an AABB if:
 *       boxes_min(i, j) <= point(i) <= boxes_max(i, j) for all dimensions i
 * @note All input matrices must have compatible dimensions:
 *       - point.size() must equal boxes_min.rows() and boxes_max.rows()
 *       - boxes_min.cols() must equal boxes_max.cols()
 *
 * @complexity O(D * N) in worst case, but often much faster due to early exit
 */
uint8_t PointInAABBs(const Eigen::VectorXd& point,
                     const Eigen::MatrixXd& boxes_min,
                     const Eigen::MatrixXd& boxes_max);

/**
 * @brief Check if multiple points are inside any of the axis-aligned bounding
 * boxes (AABBs).
 *
 * Tests each point against all provided AABBs to determine which points lie
 * within at least one bounding box. Each point is processed independently,
 * making this function suitable for parallel execution.
 *
 * @param points Matrix of shape (D, M) containing M points in D-dimensional
 * space. Each column represents one point.
 * @param boxes_min Matrix of shape (D, N) containing minimum corners of N
 * bounding boxes. Each column represents the minimum corner of one AABB.
 * @param boxes_max Matrix of shape (D, N) containing maximum corners of N
 * bounding boxes. Each column represents the maximum corner of one AABB.
 * @param parallelize If true, uses OpenMP to distribute computation across
 * available threads. If false, processes points sequentially. Default is false.
 *
 * @return std::vector<uint8_t> Vector of size M where element i is 1 if
 * points.col(i) is inside any AABB, 0 otherwise
 *
 * @note Each point is tested against all AABBs with early exit on first
 * collision
 * @note When parallelize=true, the computation is distributed across threads
 * using dynamic scheduling for better load balancing
 * @note All input matrices must have compatible dimensions:
 *       - points.rows() must equal boxes_min.rows() and boxes_max.rows()
 *       - boxes_min.cols() must equal boxes_max.cols()
 *
 * @complexity O(D * M * N) in worst case, but often faster due to early exits.
 *             With parallelization: O(D * M * N / num_threads) expected time.
 *
 * @warning Parallelization may be slower than serial execution for small
 * problem sizes due to thread overhead. Consider using parallelize=false for
 * small inputs.
 */
std::vector<uint8_t> PointsInAABBs(const Eigen::MatrixXd& points,
                                   const Eigen::MatrixXd& boxes_min,
                                   const Eigen::MatrixXd& boxes_max,
                                   bool parallelize = false);

std::pair<std::vector<std::vector<int>>, bool> TaggedAABBsatPoints(
    const Eigen::MatrixXd& points, const Eigen::MatrixXd& boxes_min,
    const Eigen::MatrixXd& boxes_max, bool parallelize = false);

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

/**
 * Check if a line segment intersects with multible axis-aligned bounding boxes
 * (AABBs)
 *
 * @tparam Vector Type of vector (usually Eigen::Vector2d or Eigen::Vector3d)
 * @param p1 (dim, N) First endpoints of the line segments
 * @param p2 (dim, N) Second endpoints of the line segments
 * @param box_min (dim, N) Minimum corners of the AABBs
 * @param box_max (dim, N) Maximum corners of the AABBs
 * @return true if the line segment intersects with the AABB, false otherwise
 */

std::vector<std::vector<uint8_t>> PwlPathAABBsIntersecting(
    const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2,
    const Eigen::MatrixXd& boxes_min, const Eigen::MatrixXd& boxes_max);

}  // namespace csdecomp
