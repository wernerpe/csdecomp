#include "linesegment_aabb_checker.h"

namespace csdecomp {

// Define A2 and A3 as static constants
static const Eigen::Matrix<double, 4, 2> A2 =
    (Eigen::Matrix<double, 4, 2>() << 1, 0, 0, 1, -1, 0, 0, -1).finished();

static const Eigen::Matrix<double, 6, 3> A3 =
    (Eigen::Matrix<double, 6, 3>() << 1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0,
     -1, 0, 0, 0, -1)
        .finished();

bool LinesegmentAABBIntersecting(const Eigen::VectorXd& p1,
                                 const Eigen::VectorXd& p2,
                                 const Eigen::VectorXd& box_min,
                                 const Eigen::VectorXd& box_max) {
  const int dim = p1.size();

  // Create b vector by concatenating box_max and -box_min
  Eigen::VectorXd b(dim * 2);
  for (int i = 0; i < dim; ++i) {
    b[i] = box_max[i];
    b[i + dim] = -box_min[i];
  }

  double tmax = std::numeric_limits<double>::infinity();
  double tmin = -std::numeric_limits<double>::infinity();

  // Calculate L and R
  Eigen::VectorXd L, R;
  Eigen::VectorXd p_diff = p2 - p1;

  if (dim == 3) {
    R = b - A3 * p1;
    L = A3 * p_diff;
  } else {
    R = b - A2 * p1;
    L = A2 * p_diff;
  }

  // Check intersections
  for (int i = 0; i < dim * 2; ++i) {
    if (L[i] < -1e-9) {
      double tmin_c = R[i] / L[i];
      if (tmin_c > tmin) {
        tmin = tmin_c;
      }
      if (tmin > tmax + 1e-9) {
        return false;
      }
    }

    if (L[i] > 1e-9) {
      double tmax_c = R[i] / L[i];
      if (tmax_c < tmax) {
        tmax = tmax_c;
      }
      if (tmin > tmax + 1e-9) {
        return false;
      }
    }
    // Check if intersection is within the line segment [0,1]
    if (tmin > 1 || tmax < 0) {
      return false;
    }

    if (std::abs(L[i]) < 1e-9 && R[i] < -1e-9) {
      return false;
    }
  }

  return true;
}

}  // namespace csdecomp