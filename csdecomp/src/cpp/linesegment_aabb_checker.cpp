#include "linesegment_aabb_checker.h"

#include <iostream>

namespace csdecomp {
// Define A2 and A3 as static constants
static const Eigen::Matrix<double, 4, 2> A2 =
    (Eigen::Matrix<double, 4, 2>() << 1, 0, 0, 1, -1, 0, 0, -1).finished();

static const Eigen::Matrix<double, 6, 3> A3 =
    (Eigen::Matrix<double, 6, 3>() << 1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0,
     -1, 0, 0, 0, -1)
        .finished();

uint8_t PointInAABBs(const Eigen::VectorXd& point,
                     const Eigen::MatrixXd& boxes_min,
                     const Eigen::MatrixXd& boxes_max) {
  const int point_dim = point.size();
  const int num_boxes = boxes_min.cols();

  // Check each box (each column is one box)
  for (int box_idx = 0; box_idx < num_boxes; ++box_idx) {
    bool inside_box = true;

    // Check each dimension of current box
    for (int dim = 0; dim < point_dim; ++dim) {
      if (point[dim] < boxes_min(dim, box_idx) ||
          point[dim] > boxes_max(dim, box_idx)) {
        inside_box = false;
        break;  // Early exit for this box
      }
    }

    if (inside_box) {
      return 1;  // Early exit - collision found!
    }
  }

  return 0;  // No collision found
}

std::vector<uint8_t> PointsInAABBs(const Eigen::MatrixXd& points,
                                   const Eigen::MatrixXd& boxes_min,
                                   const Eigen::MatrixXd& boxes_max,
                                   bool parallelize) {
  const int point_dim = points.rows();
  const int box_dim = boxes_min.rows();
  assert(point_dim == box_dim &&
         "box dimension does not match point dimension");
  assert(boxes_max.rows() == box_dim &&
         "upper and lower bounds of boxes have different dimensions");
  assert(boxes_max.cols() == boxes_min.cols() &&
         "upper and lower bounds of boxes list different number of boxes");

  const int num_points = points.cols();  // Each column is a point
  std::vector<uint8_t> results(num_points);

  if (parallelize) {
    std::cout << "Available threads " << omp_get_max_threads() << std::endl;
#pragma omp parallel for schedule(static, 100)
    for (int i = 0; i < num_points; ++i) {
      const int point_dim = points.rows();
      bool found_collision = false;

      for (int box_idx = 0; box_idx < boxes_min.cols() && !found_collision;
           ++box_idx) {
        bool inside_box = true;
        for (int dim = 0; dim < point_dim; ++dim) {
          if (points(dim, i) < boxes_min(dim, box_idx) ||
              points(dim, i) > boxes_max(dim, box_idx)) {
            inside_box = false;
            break;
          }
        }
        if (inside_box) found_collision = true;
      }
      results[i] = found_collision ? 1 : 0;
    }
  } else {
    // Serial version
    for (int i = 0; i < num_points; ++i) {
      Eigen::VectorXd point = points.col(i);
      results[i] = PointInAABBs(point, boxes_min, boxes_max);
    }
  }

  return results;
}

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

std::vector<uint8_t> LinesegmentAABBsIntersecting(
    const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
    const Eigen::MatrixXd& boxes_min, const Eigen::MatrixXd& boxes_max) {
  const int dim = p1.size();
  const int N = boxes_max.cols();
  // Create b vector by concatenating box_max and -box_min
  Eigen::MatrixXd b(dim * 2, N);
  b.block(0, 0, dim, N) = boxes_max;
  b.block(dim, 0, dim, N) = -boxes_min;

  Eigen::VectorXd tmax =
      Eigen::VectorXd::Constant(N, std::numeric_limits<double>::infinity());
  Eigen::VectorXd tmin =
      Eigen::VectorXd::Constant(N, -std::numeric_limits<double>::infinity());

  // Calculate L and R
  Eigen::MatrixXd L, R, R_div_L;
  Eigen::VectorXd p_diff = p2 - p1;
  R = b;
  if (dim == 3) {
    R = R.colwise() - A3 * p1;
    L = (A3 * p_diff).replicate(1, N);
  } else {
    R = R.colwise() - A2 * p1;
    L = (A2 * p_diff).replicate(1, N);
  }
  R_div_L = R.cwiseQuotient(L);

  // Check intersections
  for (int i = 0; i < dim * 2; ++i) {
    for (int j = 0; j < N; ++j) {
      if (L(i, j) < -1e-9) {
        double tmin_c = R_div_L(i, j);
        if (tmin_c > tmin(j)) {
          tmin(j) = tmin_c;
        }
      }
      if (L(i, j) > 1e-9) {
        double tmax_c = R_div_L(i, j);
        if (tmax_c < tmax(j)) {
          tmax(j) = tmax_c;
        }
      }
      if (std::abs(L(i, j)) <= 1e-9 && R(i, j) < -1e-9) {
        tmin(j) = std::numeric_limits<double>::infinity();
        tmax(j) = -std::numeric_limits<double>::infinity();
      }
    }
  }
  std::vector<uint8_t> results;
  for (int idx = 0; idx < N; ++idx) {
    if (tmin(idx) > tmax(idx) || tmin(idx) > 1 || tmax(idx) < 0) {
      results.push_back(0);
    } else {
      results.push_back(1);
    }
  }
  return results;
}

std::vector<std::vector<uint8_t>> PwlPathAABBsIntersecting(
    const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2,
    const Eigen::MatrixXd& boxes_min, const Eigen::MatrixXd& boxes_max) {
  int N = p1.cols();
  int dim = p1.rows();
  assert(dim <= 3 && "This is only implemented for 2 and 3 dimensions");
  std::vector<std::vector<uint8_t>> result;
  for (int segment_idx = 0; segment_idx < N; segment_idx++) {
    result.push_back(LinesegmentAABBsIntersecting(
        p1.col(segment_idx), p2.col(segment_idx), boxes_min, boxes_max));
  }
  return result;
}

}  // namespace csdecomp