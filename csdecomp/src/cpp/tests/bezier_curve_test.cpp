#include "bezier_curve.h"

#include <algorithm>
#include <numeric>

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

      // Handle dimension compatibility by extending with zeros
      Eigen::VectorXd p1 = Eigen::VectorXd::Zero(new_dimension);
      Eigen::VectorXd p2 = Eigen::VectorXd::Zero(new_dimension);

      p1.head(dimension_) = points_.row(j);
      p2.head(other.dimension_) = other.points_.row(i - j);

      // Element-wise multiplication
      Eigen::VectorXd product = p1.cwiseProduct(p2);
      new_points.row(i) += coeff * product.transpose();
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

  // Use the multiplication approach with a curve of ones
  int elevation_degree = new_degree - degree_;
  Eigen::MatrixXd ones_points = Eigen::MatrixXd::Ones(elevation_degree + 1, 1);
  BezierCurve ones_curve(ones_points, initial_time_, final_time_);

  return *this * ones_curve;
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

  int max_degree = std::max(curve1.degree_, curve2.degree_);
  int new_dimension = std::max(curve1.dimension_, curve2.dimension_);

  Eigen::MatrixXd new_points =
      Eigen::MatrixXd::Zero(max_degree + 1, new_dimension);

  // Add points with dimension compatibility
  new_points.leftCols(curve1.dimension_) += curve1.points_;
  new_points.leftCols(curve2.dimension_) += curve2.points_;

  return BezierCurve(new_points, initial_time_, final_time_);
}

BezierCurve BezierCurve::operator+(double scalar) const {
  // Create a constant curve with the scalar value
  return *this + scalar_to_curve(scalar);
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
  return curve.scalar_to_curve(scalar) - curve;
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

  // First row starts as zeros
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
  if (split_time <= initial_time_) {
    throw std::invalid_argument("Split time must be larger than initial time.");
  }
  if (split_time >= final_time_) {
    throw std::invalid_argument("Split time must be smaller than final time.");
  }

  Eigen::MatrixXd points = points_;
  Eigen::MatrixXd points1 = Eigen::MatrixXd::Zero(degree_ + 1, dimension_);
  Eigen::MatrixXd points2 = Eigen::MatrixXd::Zero(degree_ + 1, dimension_);

  double c = (split_time - initial_time_) / duration_;
  double d = (final_time_ - split_time) / duration_;

  // De Casteljau's algorithm for domain splitting
  for (int i = 0; i <= degree_; ++i) {
    points1.row(i) = points.row(0);
    points2.row(degree_ - i) = points.row(degree_ - i);

    if (i < degree_) {  // Don't update points on last iteration
      Eigen::MatrixXd new_points(points.rows() - 1, dimension_);
      for (int j = 0; j < points.rows() - 1; ++j) {
        new_points.row(j) = points.row(j + 1) * c + points.row(j) * d;
      }
      points = new_points;
    }
  }

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

  // Linear search (can be optimized to binary search for many segments)
  for (size_t i = 0; i < curves_.size(); ++i) {
    if (time <= curves_[i].final_time()) {
      return i;
    }
  }
  return curves_.size() - 1;  // Fallback, should not reach here
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

}  // namespace csdecomp