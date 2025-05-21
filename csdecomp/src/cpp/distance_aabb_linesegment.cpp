#include "distance_aabb_linesegment.h"

namespace csdecomp {
namespace {
// Helper function to project a point onto an AABB
Eigen::VectorXd ProjectOntoAABB(const Eigen::VectorXd& p,
                                const Eigen::VectorXd& lb,
                                const Eigen::VectorXd& ub) {
  Eigen::VectorXd result = p;
  for (int dim = 0; dim < p.size(); dim++) {
    if (p(dim) > ub(dim)) {
      result(dim) = ub(dim);
    } else if (p(dim) < lb(dim)) {
      result(dim) = lb(dim);
    }
    // If the point is already inside the AABB for this dimension,
    // we keep its original coordinate
  }
  return result;
}
}  // namespace

std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> DistanceLinesegmentAABB(
    const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
    const Eigen::VectorXd& box_min, const Eigen::VectorXd& box_max,
    const int maxit, const double tol) {
  // Golden ratio minus 1
  const double invphi = (std::sqrt(5.0) - 1.0) / 2.0;

  Eigen::VectorXd dir = p2 - p1;
  double segment_length = dir.norm();

  // Early termination if segment is too short
  if (segment_length < tol) {
    Eigen::VectorXd p_proj = ProjectOntoAABB(p1, box_min, box_max);
    return std::make_tuple(p_proj, p1, (p1 - p_proj).norm());
  }

  // Normalize direction vector
  Eigen::VectorXd dir_normalized = dir / segment_length;

  // Initialize search interval [a, b]
  double a = 0.0;
  double b = segment_length;

  // Golden section search iterations
  for (int i = 0; i < maxit; i++) {
    // Calculate new points in the interval
    double c = b - (b - a) * invphi;
    double d = a + (b - a) * invphi;

    // Compute points on the line segment
    Eigen::VectorXd p_c = p1 + c * dir_normalized;
    Eigen::VectorXd p_d = p1 + d * dir_normalized;

    // Project the points onto the AABB and compute distances
    Eigen::VectorXd p_c_proj = ProjectOntoAABB(p_c, box_min, box_max);
    double fc = (p_c - p_c_proj).norm();

    Eigen::VectorXd p_d_proj = ProjectOntoAABB(p_d, box_min, box_max);
    double fd = (p_d - p_d_proj).norm();

    // Update search interval based on function values
    if (fc < fd) {
      // New interval [a, d]
      b = d;
    } else {
      // New interval [c, b]
      a = c;
    }

    // Check convergence
    if (std::abs(b - a) < tol) {
      break;
    }
  }

  // Compute optimal point on the line segment
  double t_opt = (a + b) / 2.0;
  Eigen::VectorXd p_optimal = p1 + t_opt * dir_normalized;

  // Project the optimal point onto the AABB
  Eigen::VectorXd p_proj = ProjectOntoAABB(p_optimal, box_min, box_max);

  // Return the projected point, point on line, and distance
  return std::make_tuple(p_proj, p_optimal, (p_optimal - p_proj).norm());
}
}  // namespace csdecomp