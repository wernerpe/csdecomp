/** This file has been created using Claude 4 by translating
 * https://github.com/TobiaMarcucci/pybezier from python.
 * commit/0ff3893eae89a0dedff984e1aaf35f71ffb0bbf1*/
#include "bezier_curve.h"

#include <omp.h>

#include <algorithm>
#include <exception>
#include <numeric>
#include <unordered_set>

namespace csdecomp {

double binomial_coefficient(int n, int k) {
  if (k > n || k < 0) return 0.0;
  if (k == 0 || k == n) return 1.0;

  // Use symmetry: C(n,k) = C(n,n-k)
  if (k > n - k) k = n - k;

  double result = 1.0;
  for (int i = 0; i < k; ++i) {
    result = result * (n - i) / (i + 1);
  }
  return result;
}

// BezierCurve implementation

void BezierCurve::precompute_binomials() {
  binomial_coeffs_.resize(degree_ + 1);
  for (int n = 0; n <= degree_; ++n) {
    binomial_coeffs_[n] = binomial_coefficient(degree_, n);
  }
}

void BezierCurve::assert_same_times(const BezierCurve& curve) const {
  const double eps = 1e-10;
  if (std::abs(initial_time_ - curve.initial_time_) > eps) {
    throw std::invalid_argument(
        "Incompatible curves, initial times don't match.");
  }
  if (std::abs(final_time_ - curve.final_time_) > eps) {
    throw std::invalid_argument(
        "Incompatible curves, final times don't match.");
  }
}

BezierCurve::BezierCurve(const Eigen::MatrixXd& points, double initial_time,
                         double final_time)
    : points_(points), initial_time_(initial_time), final_time_(final_time) {
  if (initial_time >= final_time) {
    throw std::invalid_argument(
        "Initial time must be smaller than final time.");
  }

  degree_ = points.rows() - 1;
  dimension_ = points.cols();
  duration_ = final_time - initial_time;

  precompute_binomials();
}

Eigen::VectorXd BezierCurve::initial_point() const { return points_.row(0); }

Eigen::VectorXd BezierCurve::final_point() const {
  return points_.row(points_.rows() - 1);
}

double BezierCurve::bernstein(double time, int n) const {
  double t = (time - initial_time_) / duration_;

  // Clamp t to [0, 1] for numerical stability
  t = std::max(0.0, std::min(1.0, t));

  return binomial_coeffs_[n] * std::pow(t, n) * std::pow(1.0 - t, degree_ - n);
}

Eigen::VectorXd BezierCurve::operator()(double time) const {
  Eigen::VectorXd result = Eigen::VectorXd::Zero(dimension_);

  for (int n = 0; n <= degree_; ++n) {
    double basis_val = bernstein(time, n);
    result += basis_val * points_.row(n).transpose();
  }

  return result;
}

Eigen::MatrixXd BezierCurve::evaluate_at_times(
    const std::vector<double>& times) const {
  const int num_times = times.size();
  Eigen::MatrixXd result(dimension_, num_times);

  for (int i = 0; i < num_times; ++i) {
    result.col(i) = (*this)(times[i]);
  }

  return result;
}

BezierCurve BezierCurve::scalar_to_curve(double scalar) const {
  Eigen::MatrixXd points(1, 1);
  points(0, 0) = scalar;
  return BezierCurve(points, initial_time_, final_time_);
}

BezierCurve BezierCurve::operator*(const BezierCurve& other) const {
  assert_same_times(other);

  int new_degree = degree_ + other.degree_;
  int new_dimension = std::max(dimension_, other.dimension_);

  Eigen::MatrixXd new_points =
      Eigen::MatrixXd::Zero(new_degree + 1, new_dimension);

  for (int i = 0; i <= new_degree; ++i) {
    int j_min = std::max(0, i - other.degree_);
    int j_max = std::min(degree_, i);

    for (int j = j_min; j <= j_max; ++j) {
      double coeff = binomial_coefficient(degree_, j) *
                     binomial_coefficient(other.degree_, i - j);

      // Get the actual control points (without padding)
      Eigen::VectorXd p1 = points_.row(j);
      Eigen::VectorXd p2 = other.points_.row(i - j);

      // Handle different dimension cases properly
      if (dimension_ == other.dimension_) {
        // Same dimensions: element-wise multiplication
        Eigen::VectorXd product = p1.cwiseProduct(p2);
        new_points.row(i).head(dimension_) += coeff * product.transpose();
      } else if (other.dimension_ == 1) {
        // Other is scalar: multiply each component of p1 by the scalar p2[0]
        Eigen::VectorXd product = p1 * p2(0);
        new_points.row(i).head(dimension_) += coeff * product.transpose();
        // If new_dimension > dimension_, the remaining components stay zero
      } else if (dimension_ == 1) {
        // This curve is scalar: multiply each component of p2 by scalar p1[0]
        Eigen::VectorXd product = p2 * p1(0);
        new_points.row(i).head(other.dimension_) += coeff * product.transpose();
        // If new_dimension > other.dimension_, the remaining components stay
        // zero
      } else {
        // Different non-scalar dimensions: zero-pad and element-wise multiply
        Eigen::VectorXd padded_p1 = Eigen::VectorXd::Zero(new_dimension);
        Eigen::VectorXd padded_p2 = Eigen::VectorXd::Zero(new_dimension);

        padded_p1.head(dimension_) = p1;
        padded_p2.head(other.dimension_) = p2;

        Eigen::VectorXd product = padded_p1.cwiseProduct(padded_p2);
        new_points.row(i) += coeff * product.transpose();
      }
    }

    new_points.row(i) /= binomial_coefficient(new_degree, i);
  }

  return BezierCurve(new_points, initial_time_, final_time_);
}

BezierCurve BezierCurve::operator*(double scalar) const {
  Eigen::MatrixXd new_points = points_ * scalar;
  return BezierCurve(new_points, initial_time_, final_time_);
}

BezierCurve operator*(double scalar, const BezierCurve& curve) {
  return curve * scalar;
}

BezierCurve BezierCurve::elevate_degree(int new_degree) const {
  if (new_degree < degree_) {
    throw std::invalid_argument("New degree must be >= current degree");
  }

  if (new_degree == degree_) {
    return *this;
  }

  // The issue might be that we need to elevate step by step
  BezierCurve result = *this;

  for (int target = degree_ + 1; target <= new_degree; ++target) {
    // Create a curve of ones with degree 1 (linear curve of constant 1)
    Eigen::MatrixXd ones_points = Eigen::MatrixXd::Ones(2, 1);
    BezierCurve ones_curve(ones_points, initial_time_, final_time_);
    result = result * ones_curve;
  }

  return result;
}

BezierCurve BezierCurve::operator+(const BezierCurve& other) const {
  assert_same_times(other);

  // Elevate degrees to match
  BezierCurve curve1 = *this;
  BezierCurve curve2 = other;

  if (other.degree_ > degree_) {
    curve1 = elevate_degree(other.degree_);
  } else if (degree_ > other.degree_) {
    curve2 = other.elevate_degree(degree_);
  }

  int new_dimension = std::max(curve1.dimension_, curve2.dimension_);
  Eigen::MatrixXd new_points =
      Eigen::MatrixXd::Zero(curve1.degree_ + 1, new_dimension);

  // Add points with dimension compatibility
  new_points.leftCols(curve1.dimension_) += curve1.points_;
  new_points.leftCols(curve2.dimension_) += curve2.points_;

  return BezierCurve(new_points, initial_time_, final_time_);
}

BezierCurve BezierCurve::operator+(double scalar) const {
  Eigen::MatrixXd new_points = points_;
  for (int i = 0; i <= degree_; ++i) {
    new_points.row(i).array() += scalar;
  }
  return BezierCurve(new_points, initial_time_, final_time_);
}

BezierCurve operator+(double scalar, const BezierCurve& curve) {
  return curve + scalar;
}

BezierCurve BezierCurve::operator-(const BezierCurve& other) const {
  return *this + (other * (-1.0));
}

BezierCurve BezierCurve::operator-(double scalar) const {
  return *this + (-scalar);
}

BezierCurve operator-(double scalar, const BezierCurve& curve) {
  Eigen::MatrixXd new_points = -curve.points_;
  for (int i = 0; i <= curve.degree_; ++i) {
    new_points.row(i).array() += scalar;
  }
  return BezierCurve(new_points, curve.initial_time_, curve.final_time_);
}

BezierCurve BezierCurve::operator-() const { return *this * (-1.0); }

BezierCurve BezierCurve::derivative() const {
  if (degree_ == 0) {
    // Derivative of constant curve is zero
    Eigen::MatrixXd zero_points = Eigen::MatrixXd::Zero(1, dimension_);
    return BezierCurve(zero_points, initial_time_, final_time_);
  }

  Eigen::MatrixXd deriv_points(degree_, dimension_);
  double scale_factor = static_cast<double>(degree_) / duration_;

  for (int i = 0; i < degree_; ++i) {
    deriv_points.row(i) = (points_.row(i + 1) - points_.row(i)) * scale_factor;
  }

  return BezierCurve(deriv_points, initial_time_, final_time_);
}

BezierCurve BezierCurve::integral(
    const Eigen::VectorXd& initial_condition) const {
  Eigen::MatrixXd integ_points(degree_ + 2, dimension_);
  double scale_factor = duration_ / static_cast<double>(degree_ + 1);

  // First row is zeros (will be modified by initial condition)
  integ_points.row(0) = Eigen::VectorXd::Zero(dimension_);

  // Scale the original points and place them
  for (int i = 0; i <= degree_; ++i) {
    integ_points.row(i + 1) = points_.row(i) * scale_factor;
  }

  // Cumulative sum
  for (int i = 1; i <= degree_ + 1; ++i) {
    integ_points.row(i) += integ_points.row(i - 1);
  }

  // Add initial condition if provided
  if (initial_condition.size() > 0) {
    if (initial_condition.size() != dimension_) {
      throw std::invalid_argument(
          "Initial condition dimension must match curve dimension");
    }
    for (int i = 0; i <= degree_ + 1; ++i) {
      integ_points.row(i) += initial_condition.transpose();
    }
  }

  return BezierCurve(integ_points, initial_time_, final_time_);
}

std::pair<BezierCurve, BezierCurve> BezierCurve::domain_split(
    double split_time) const {
  if (split_time < initial_time_) {
    throw std::invalid_argument("Split time must be larger than initial time.");
  }
  if (split_time > final_time_) {
    throw std::invalid_argument("Split time must be smaller than final time.");
  }

  Eigen::MatrixXd points = points_;
  Eigen::MatrixXd points1 = Eigen::MatrixXd::Zero(degree_ + 1, dimension_);
  Eigen::MatrixXd points2 = Eigen::MatrixXd::Zero(degree_ + 1, dimension_);

  double c = (split_time - initial_time_) / duration_;
  double d = (final_time_ - split_time) / duration_;

  // De Casteljau's algorithm for domain splitting
  for (int i = 0; i < degree_; ++i) {
    points1.row(i) = points.row(0);
    points2.row(degree_ - i) = points.row(degree_ - i);

    // Update points array for next iteration
    Eigen::MatrixXd new_points(points.rows() - 1, dimension_);
    for (int j = 0; j < points.rows() - 1; ++j) {
      new_points.row(j) = points.row(j + 1) * c + points.row(j) * d;
    }
    points = new_points;
  }

  // Set the final shared point
  points1.row(degree_) = points.row(0);
  points2.row(0) = points.row(0);

  BezierCurve curve1(points1, initial_time_, split_time);
  BezierCurve curve2(points2, split_time, final_time_);

  return std::make_pair(curve1, curve2);
}

double BezierCurve::l2_squared() const {
  double result = 0.0;

  for (int i = 0; i <= degree_; ++i) {
    double bi = binomial_coefficient(degree_, i);
    for (int j = i; j <= degree_; ++j) {
      double bj = binomial_coefficient(degree_, j);
      double bij = binomial_coefficient(2 * degree_, i + j);

      double coeff = bi * bj / bij;
      if (j > i) {
        coeff *= 2.0;
      }

      result += coeff * points_.row(i).dot(points_.row(j));
    }
  }

  return duration_ * result / (2.0 * degree_ + 1.0);
}
// CompositeBezierCurve implementation

CompositeBezierCurve::CompositeBezierCurve(
    const std::vector<BezierCurve>& curves)
    : curves_(curves) {
  if (curves.empty()) {
    throw std::invalid_argument(
        "Cannot create composite curve with no segments.");
  }

  // Validate continuity
  const double eps = 1e-10;
  for (size_t i = 1; i < curves.size(); ++i) {
    if (std::abs(curves[i - 1].final_time() - curves[i].initial_time()) > eps) {
      throw std::invalid_argument(
          "Initial and final times don't match between segments.");
    }
  }

  // Validate same dimension
  dimension_ = curves[0].dimension();
  for (const auto& curve : curves) {
    if (curve.dimension() != dimension_) {
      throw std::invalid_argument("All curves must have the same dimension.");
    }
  }

  initial_time_ = curves[0].initial_time();
  final_time_ = curves.back().final_time();
  duration_ = final_time_ - initial_time_;

  // Build knot times
  knot_times_.push_back(initial_time_);
  for (const auto& curve : curves) {
    knot_times_.push_back(curve.final_time());
  }
}

int CompositeBezierCurve::curve_segment(double time) const {
  // Handle boundary cases
  if (time <= initial_time_) return 0;
  if (time >= final_time_) return curves_.size() - 1;

  // Binary search for efficiency with many segments
  int left = 0, right = curves_.size() - 1;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (curves_[mid].final_time() < time) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

Eigen::VectorXd CompositeBezierCurve::operator()(double time) const {
  int segment = curve_segment(time);
  return curves_[segment](time);
}

Eigen::MatrixXd CompositeBezierCurve::evaluate_at_times(
    const std::vector<double>& times) const {
  const int num_times = times.size();
  Eigen::MatrixXd result(dimension_, num_times);

  for (int i = 0; i < num_times; ++i) {
    result.col(i) = (*this)(times[i]);
  }

  return result;
}

std::vector<int> CompositeBezierCurve::curve_segments(
    const std::vector<double>& times) const {
  std::vector<int> segments;
  segments.reserve(times.size());

  for (double time : times) {
    segments.push_back(curve_segment(time));
  }

  return segments;
}

const BezierCurve& CompositeBezierCurve::operator[](int index) const {
  return curves_[index];
}

uint8_t BezierCurveHPolyhedronCollisionFree(const BezierCurve& curve,
                                            const Eigen::MatrixXd& A,
                                            const Eigen::VectorXd& b,
                                            double tol) {
  // Get control points as matrix (rows = control points, cols = dimensions)
  const Eigen::MatrixXd& pts = curve.points();

  // Compute A*pts^T - b for collision detection
  // pts.transpose() gives us (dimension x num_points)
  // A * pts.transpose() gives us (num_constraints x num_points)
  Eigen::MatrixXd axmb = A * pts.transpose() - b.replicate(1, pts.rows());

  // Check if points are outside region (any constraint violated means outside)
  // A point is outside if max(A*x - b) >= 0
  Eigen::VectorXd max_violations = axmb.colwise().maxCoeff();
  Eigen::Array<bool, Eigen::Dynamic, 1> pts_outside =
      (max_violations.array() >= 0.0);

  // Check if start and end points are collision-free
  bool start_and_end_col_free =
      pts_outside(0) && pts_outside(pts_outside.size() - 1);

  // Check if all points are outside at least one hyperplane
  // This means there exists a hyperplane that separates all control points from
  // the obstacle
  Eigen::VectorXd min_violations = axmb.rowwise().minCoeff();
  bool all_points_outside_hyperplane = (min_violations.array() >= 0.0).any();

  // Compute bounding sphere radius of control points
  Eigen::VectorXd mean = pts.colwise().mean();
  Eigen::MatrixXd diff = pts.rowwise() - mean.transpose();

  // Compute norms of each row (control point differences from mean)
  Eigen::VectorXd norms(diff.rows());
  for (int i = 0; i < diff.rows(); ++i) {
    norms(i) = diff.row(i).norm();
  }
  double radius = norms.maxCoeff();

  bool min_tol_reached = radius <= tol;

  // Early returns based on collision conditions
  if (!start_and_end_col_free) {
    return false;
  }

  if (all_points_outside_hyperplane) {
    return true;
  }

  if (min_tol_reached) {
    // Uncomment for debugging:
    // std::cout << "min tol reached" << std::endl;
    return false;
  }

  // Recursive subdivision
  double mid_time = (curve.initial_time() + curve.final_time()) / 2.0;
  auto [r1, r2] = curve.domain_split(mid_time);

  bool col_free1 = BezierCurveHPolyhedronCollisionFree(r1, A, b, tol);
  bool col_free2 = BezierCurveHPolyhedronCollisionFree(r2, A, b, tol);

  return col_free1 && col_free2;
}

uint8_t BezierCurveSphereCollisionFree(const BezierCurve& curve,
                                       const Eigen::VectorXd& center,
                                       double radius, double tol) {
  // Get control points as matrix (rows = control points, cols = dimensions)
  const Eigen::MatrixXd& pts = curve.points();

  // Compute mean of control points
  Eigen::VectorXd mean = pts.colwise().mean();

  // Construct separating hyperplane
  // Normal vector pointing from sphere center toward control points
  Eigen::VectorXd a = mean - center;
  double a_norm = a.norm();

  // Check if mean is too close to center (degenerate case)
  if (a_norm < 1e-10) {
    // Control points are centered at sphere center, check if any point is
    // outside
    for (int i = 0; i < pts.rows(); ++i) {
      Eigen::VectorXd diff = pts.row(i).transpose() - center;
      if (diff.norm() > radius) {
        // At least one point is outside, need to subdivide
        // Fall through to subdivision logic
        a = Eigen::VectorXd::Constant(center.size(), 1.0 / std::sqrt(center.size()));
        a_norm = 1.0;
        break;
      }
    }
    if (a_norm < 1e-10) {
      // All points are at center, definitely in collision
      return false;
    }
  }

  // Normalize to get unit normal
  a = a / a_norm;

  // Tangent point on sphere surface
  Eigen::VectorXd tangent_pt = center + a * radius;

  // Hyperplane offset: b = -a^T * tangent_pt
  double b = -a.dot(tangent_pt);

  // Check if all control points are on the safe side: a^T * p + b >= 0
  Eigen::VectorXd violations(pts.rows());
  for (int i = 0; i < pts.rows(); ++i) {
    violations(i) = a.dot(pts.row(i).transpose()) + b;
  }

  // Check if start and end points are collision-free (outside sphere)
  Eigen::VectorXd start_diff = pts.row(0).transpose() - center;
  Eigen::VectorXd end_diff = pts.row(pts.rows() - 1).transpose() - center;
  bool start_col_free = start_diff.norm() > radius;
  bool end_col_free = end_diff.norm() > radius;
  bool start_and_end_col_free = start_col_free && end_col_free;

  // Check if all points are on the safe side of the separating hyperplane
  bool all_points_safe = (violations.array() >= 0.0).all();

  // Compute bounding sphere radius of control points
  Eigen::MatrixXd diff = pts.rowwise() - mean.transpose();
  Eigen::VectorXd norms(diff.rows());
  for (int i = 0; i < diff.rows(); ++i) {
    norms(i) = diff.row(i).norm();
  }
  double bounding_radius = norms.maxCoeff();

  bool min_tol_reached = bounding_radius <= tol;

  // Early returns based on collision conditions
  if (!start_and_end_col_free) {
    return false;
  }

  if (all_points_safe) {
    return true;
  }

  if (min_tol_reached) {
    return false;
  }

  // Recursive subdivision
  double mid_time = (curve.initial_time() + curve.final_time()) / 2.0;
  auto [r1, r2] = curve.domain_split(mid_time);

  bool col_free1 = BezierCurveSphereCollisionFree(r1, center, radius, tol);
  bool col_free2 = BezierCurveSphereCollisionFree(r2, center, radius, tol);

  return col_free1 && col_free2;
}

std::vector<std::vector<int>> IntersectCompositeBezierCurveWithHPolyhedra(
    const CompositeBezierCurve& c, const std::vector<Eigen::MatrixXd>& As,
    const std::vector<Eigen::VectorXd>& bs,
    const std::vector<std::vector<int>>& hpoly_to_ignore, double tol,
    bool parallelize) {
  // Validate input dimensions
  if (As.size() != bs.size()) {
    throw std::invalid_argument("As and bs must have the same size");
  }

  const size_t num_segments = c.num_segments();
  const size_t num_obstacles = As.size();

  if (hpoly_to_ignore.size() != num_segments) {
    throw std::invalid_argument(
        "hpoly_to_ignore size must equal number of segments");
  }

  // Initialize result vector
  std::vector<std::vector<int>> intersections(num_segments);

  if (parallelize) {
#pragma omp parallel for schedule(dynamic)
    for (size_t seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
      const BezierCurve& segment = c[seg_idx];

      // Convert ignore list to unordered_set for O(1) lookup
      std::unordered_set<int> ignore_set(hpoly_to_ignore[seg_idx].begin(),
                                         hpoly_to_ignore[seg_idx].end());

      for (size_t obs_idx = 0; obs_idx < num_obstacles; ++obs_idx) {
        // Skip this obstacle if it's in the ignore list for this segment
        if (ignore_set.find(static_cast<int>(obs_idx)) != ignore_set.end()) {
          continue;
        }

        if (!BezierCurveHPolyhedronCollisionFree(segment, As[obs_idx],
                                                 bs[obs_idx], tol)) {
          intersections[seg_idx].push_back(static_cast<int>(obs_idx));
        }
      }
    }
  } else {
    // Sequential version
    for (size_t seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
      const BezierCurve& segment = c[seg_idx];

      // Convert ignore list to unordered_set for O(1) lookup
      std::unordered_set<int> ignore_set(hpoly_to_ignore[seg_idx].begin(),
                                         hpoly_to_ignore[seg_idx].end());

      for (size_t obs_idx = 0; obs_idx < num_obstacles; ++obs_idx) {
        // Skip this obstacle if it's in the ignore list for this segment
        if (ignore_set.find(static_cast<int>(obs_idx)) != ignore_set.end()) {
          continue;
        }

        if (!BezierCurveHPolyhedronCollisionFree(segment, As[obs_idx],
                                                 bs[obs_idx], tol)) {
          intersections[seg_idx].push_back(static_cast<int>(obs_idx));
        }
      }
    }
  }

  return intersections;
}

std::vector<std::vector<int>> IntersectCompositeBezierCurveWithSpheres(
    const CompositeBezierCurve& c,
    const std::vector<Eigen::VectorXd>& centers,
    const std::vector<double>& radii,
    const std::vector<std::vector<int>>& spheres_to_ignore, double tol,
    bool parallelize) {
  // Validate input dimensions
  if (centers.size() != radii.size()) {
    throw std::invalid_argument("centers and radii must have the same size");
  }

  const size_t num_segments = c.num_segments();
  const size_t num_obstacles = centers.size();

  if (spheres_to_ignore.size() != num_segments) {
    throw std::invalid_argument(
        "spheres_to_ignore size must equal number of segments");
  }

  // Initialize result vector
  std::vector<std::vector<int>> intersections(num_segments);

  if (parallelize) {
#pragma omp parallel for schedule(dynamic)
    for (size_t seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
      const BezierCurve& segment = c[seg_idx];

      // Convert ignore list to unordered_set for O(1) lookup
      std::unordered_set<int> ignore_set(spheres_to_ignore[seg_idx].begin(),
                                         spheres_to_ignore[seg_idx].end());

      for (size_t obs_idx = 0; obs_idx < num_obstacles; ++obs_idx) {
        // Skip this obstacle if it's in the ignore list for this segment
        if (ignore_set.find(static_cast<int>(obs_idx)) != ignore_set.end()) {
          continue;
        }

        if (!BezierCurveSphereCollisionFree(segment, centers[obs_idx],
                                            radii[obs_idx], tol)) {
          intersections[seg_idx].push_back(static_cast<int>(obs_idx));
        }
      }
    }
  } else {
    // Sequential version
    for (size_t seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
      const BezierCurve& segment = c[seg_idx];

      // Convert ignore list to unordered_set for O(1) lookup
      std::unordered_set<int> ignore_set(spheres_to_ignore[seg_idx].begin(),
                                         spheres_to_ignore[seg_idx].end());

      for (size_t obs_idx = 0; obs_idx < num_obstacles; ++obs_idx) {
        // Skip this obstacle if it's in the ignore list for this segment
        if (ignore_set.find(static_cast<int>(obs_idx)) != ignore_set.end()) {
          continue;
        }

        if (!BezierCurveSphereCollisionFree(segment, centers[obs_idx],
                                            radii[obs_idx], tol)) {
          intersections[seg_idx].push_back(static_cast<int>(obs_idx));
        }
      }
    }
  }

  return intersections;
}

}  // namespace csdecomp