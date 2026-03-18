/** This file has been created using Claude 4 by translating
 * https://github.com/TobiaMarcucci/pybezier from python.
 * commit/0ff3893eae89a0dedff984e1aaf35f71ffb0bbf1*/

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// #include "hpolyhedron.h"

namespace csdecomp {

/**
 * @brief Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
 */
double binomial_coefficient(int n, int k);

/**
 * @brief Single Bezier curve implementation
 *
 * Represents a single Bezier curve defined by control points and time interval.
 * The curve is parameterized over [initial_time, final_time].
 */
class BezierCurve {
 private:
  Eigen::MatrixXd points_;  // Shape: (degree+1, dimension)
  double initial_time_;
  double final_time_;
  double duration_;
  int degree_;
  int dimension_;

  // Precomputed binomial coefficients for efficiency
  std::vector<double> binomial_coeffs_;

  void precompute_binomials();
  void assert_same_times(const BezierCurve& curve) const;

 public:
  /**
   * @brief Constructor
   * @param points Control points matrix of shape (degree+1, dimension)
   * @param initial_time Start time of the curve
   * @param final_time End time of the curve
   */
  BezierCurve(const Eigen::MatrixXd& points, double initial_time = 0.0,
              double final_time = 1.0);

  // Copy constructor and assignment operator
  BezierCurve(const BezierCurve& other) = default;
  BezierCurve& operator=(const BezierCurve& other) = default;

  // Move constructor and assignment operator
  BezierCurve(BezierCurve&& other) noexcept = default;
  BezierCurve& operator=(BezierCurve&& other) noexcept = default;

  // Getters
  int degree() const { return degree_; }
  int dimension() const { return dimension_; }
  double initial_time() const { return initial_time_; }
  double final_time() const { return final_time_; }
  double duration() const { return duration_; }
  const Eigen::MatrixXd& points() const { return points_; }

  /**
   * @brief Get the shape of control points (for compatibility with Python
   * tests)
   * @return Pair of (rows, cols) representing the shape
   */
  std::pair<int, int> shape() const {
    return std::make_pair(degree_ + 1, dimension_);
  }

  Eigen::VectorXd initial_point() const;
  Eigen::VectorXd final_point() const;

  /**
   * @brief Compute Bernstein basis function value
   * @param time Time parameter
   * @param n Basis function index
   * @return Bernstein basis value
   */
  double bernstein(double time, int n) const;

  /**
   * @brief Evaluate the Bezier curve at a given time
   * @param time Time parameter
   * @return Point on the curve
   */
  Eigen::VectorXd operator()(double time) const;

  /**
   * @brief Vectorized evaluation at multiple time points
   * @param times Vector of time parameters
   * @return Matrix where each column is a point on the curve (shape: dimension
   * x num_times)
   */
  Eigen::MatrixXd evaluate_at_times(const std::vector<double>& times) const;

  /**
   * @brief Multiply two Bezier curves or multiply by a scalar
   * @param other Another BezierCurve or a scalar value
   * @return New BezierCurve representing the product
   */
  BezierCurve operator*(const BezierCurve& other) const;
  BezierCurve operator*(double scalar) const;
  friend BezierCurve operator*(double scalar, const BezierCurve& curve);

  /**
   * @brief In-place multiplication by scalar
   * @param scalar Scalar value to multiply by
   * @return Reference to this curve
   */
  BezierCurve& operator*=(double scalar) {
    points_ *= scalar;
    return *this;
  }

  /**
   * @brief Add two Bezier curves or add a scalar
   * @param other Another BezierCurve or a scalar value
   * @return New BezierCurve representing the sum
   */
  BezierCurve operator+(const BezierCurve& other) const;
  BezierCurve operator+(double scalar) const;
  friend BezierCurve operator+(double scalar, const BezierCurve& curve);

  /**
   * @brief In-place addition
   * @param other Another BezierCurve to add
   * @return Reference to this curve
   */
  BezierCurve& operator+=(const BezierCurve& other) {
    *this = *this + other;
    return *this;
  }

  /**
   * @brief Subtract two Bezier curves or subtract a scalar
   * @param other Another BezierCurve or a scalar value
   * @return New BezierCurve representing the difference
   */
  BezierCurve operator-(const BezierCurve& other) const;
  BezierCurve operator-(double scalar) const;
  friend BezierCurve operator-(double scalar, const BezierCurve& curve);

  /**
   * @brief In-place subtraction
   * @param other Another BezierCurve to subtract
   * @return Reference to this curve
   */
  BezierCurve& operator-=(const BezierCurve& other) {
    *this = *this - other;
    return *this;
  }

  /**
   * @brief Unary minus operator
   * @return New BezierCurve with negated control points
   */
  BezierCurve operator-() const;

  /**
   * @brief Scalar product of two Bezier curves using the @ operator equivalent
   *
   * Computes the scalar product by multiplying the curves pointwise
   * and then summing across dimensions to produce a scalar-valued curve.
   * Note: C++ doesn't have @ operator, so this is named method
   * @param other Another BezierCurve
   * @return New scalar-valued BezierCurve representing the scalar product
   */
  BezierCurve scalar_product(const BezierCurve& other) const {
    if (dimension_ != other.dimension_) {
      throw std::invalid_argument(
          "Curves must have same dimension for scalar product");
    }

    // Multiply the curves pointwise
    BezierCurve product_curve = *this * other;

    // Sum across dimensions (axis=1) to get scalar values
    Eigen::VectorXd summed_points(product_curve.degree_ + 1);
    for (int i = 0; i <= product_curve.degree_; ++i) {
      summed_points(i) = product_curve.points_.row(i).sum();
    }

    // Reshape to maintain the (n_points, 1) structure for scalar curve
    Eigen::MatrixXd scalar_points = summed_points.transpose();

    return BezierCurve(scalar_points, initial_time_, final_time_);
  }

  /**
   * @brief Elevate the degree of the Bezier curve
   * @param new_degree Target degree (must be >= current degree)
   * @return New BezierCurve with elevated degree
   */
  BezierCurve elevate_degree(int new_degree) const;

  /**
   * @brief Compute the derivative of the Bezier curve
   * @return New BezierCurve representing the derivative
   */
  BezierCurve derivative() const;

  /**
   * @brief Compute the integral of the Bezier curve
   * @param initial_condition Initial condition for integration (optional)
   * @return New BezierCurve representing the integral
   */
  BezierCurve integral(
      const Eigen::VectorXd& initial_condition = Eigen::VectorXd()) const;

  /**
   * @brief Split the curve at a given time into two curves
   * @param split_time Time at which to split the curve
   * @return Pair of BezierCurves (left curve, right curve)
   */
  std::pair<BezierCurve, BezierCurve> domain_split(double split_time) const;

  /**
   * @brief Split the curve at a given time, handling boundary cases
   * Python-compatible version that returns nullptrs for boundary cases
   * @param split_time Time at which to split the curve
   * @return Pair of BezierCurve pointers (nullptr for out-of-bounds splits)
   */
  std::pair<std::unique_ptr<BezierCurve>, std::unique_ptr<BezierCurve>>
  split_domain(double split_time) const {
    if (split_time <= initial_time_) {
      return std::make_pair(nullptr, std::make_unique<BezierCurve>(*this));
    }
    if (split_time >= final_time_) {
      return std::make_pair(std::make_unique<BezierCurve>(*this), nullptr);
    }

    auto [left, right] = domain_split(split_time);
    return std::make_pair(std::make_unique<BezierCurve>(left),
                          std::make_unique<BezierCurve>(right));
  }

  /**
   * @brief Shift the time domain of the curve
   * @param time_shift Amount to shift the time domain
   * @return New BezierCurve with shifted time domain
   */
  BezierCurve shift_domain(double time_shift) const {
    return BezierCurve(points_, initial_time_ + time_shift,
                       final_time_ + time_shift);
  }

  /**
   * @brief Compute the L2 squared norm of the curve
   * @return L2 squared norm value
   */
  double l2_squared() const;

  /**
   * @brief Compute the squared L2 norm of the curve (alias for l2_squared)
   * @return Squared L2 norm value
   */
  double squared_l2_norm() const { return l2_squared(); }

  /**
   * @brief Compute integral of a convex function over the curve
   * Uses Jensen's inequality upper bound approximation
   * @param f Function to integrate (should take VectorXd and return double)
   * @return Upper bound approximation of the integral
   */
  template <typename Function>
  double integral_of_convex_function(Function f) const {
    double c = duration_ / (degree_ + 1);
    double sum = 0.0;
    for (int i = 0; i <= degree_; ++i) {
      sum += f(points_.row(i).transpose());
    }
    return c * sum;
  }

  /**
   * @brief Check if another curve has compatible time domain (Python
   * compatibility)
   * @param curve Another BezierCurve to check against
   */
  void _check_same_times(const BezierCurve& curve) const {
    assert_same_times(curve);
  }

  /**
   * @brief Access to Bernstein basis function (Python compatibility)
   * @param time Time parameter
   * @param n Basis function index
   * @return Bernstein basis value
   */
  double _berstein(double time, int n) const { return bernstein(time, n); }

 private:
  /**
   * @brief Convert a scalar to a constant Bezier curve
   * @param scalar Scalar value
   * @return BezierCurve representing the constant
   */
  BezierCurve scalar_to_curve(double scalar) const;
};

/**
 * @brief Composite Bezier curve implementation
 *
 * Represents a piecewise Bezier curve composed of multiple connected segments.
 */
class CompositeBezierCurve {
 private:
  std::vector<BezierCurve> curves_;
  std::vector<double> knot_times_;
  double initial_time_;
  double final_time_;
  double duration_;
  int dimension_;

 public:
  /**
   * @brief Constructor
   * @param curves Vector of BezierCurve segments
   */
  CompositeBezierCurve(const std::vector<BezierCurve>& curves);

  // Getters
  int dimension() const { return dimension_; }
  double initial_time() const { return initial_time_; }
  double final_time() const { return final_time_; }
  double duration() const { return duration_; }
  size_t num_segments() const { return curves_.size(); }
  const std::vector<double>& knot_times() const { return knot_times_; }
  const std::vector<BezierCurve>& curves() const { return curves_; }
  /**
   * @brief Find which curve segment contains the given time
   * @param time Time parameter
   * @return Segment index
   */
  int curve_segment(double time) const;

  /**
   * @brief Evaluate the composite curve at a given time
   * @param time Time parameter
   * @return Point on the curve
   */
  Eigen::VectorXd operator()(double time) const;

  /**
   * @brief Vectorized evaluation at multiple time points
   * @param times Vector of time parameters
   * @return Matrix where each column is a point on the curve (shape: dimension
   * x num_times)
   */
  Eigen::MatrixXd evaluate_at_times(const std::vector<double>& times) const;

  /**
   * @brief Get segment indices for multiple time points
   * @param times Vector of time parameters
   * @return Vector of segment indices
   */
  std::vector<int> curve_segments(const std::vector<double>& times) const;

  /**
   * @brief Access individual curve segments
   */
  const BezierCurve& operator[](int index) const;
};

/**
 * @brief Check if a Bezier curve is collision-free with respect to an
 * H-polyhedron obstacle
 *
 * Determines whether a Bezier curve avoids collision with a convex polytope
 * obstacle defined by linear inequalities Ax <= b. Uses recursive subdivision
 * with control point analysis for efficient collision detection. This algorithm
 * never returns a false positive: It never claims a curve is collision-free
 * when it is in collision.
 *
 * The algorithm works by:
 * 1. Checking if start/end points are outside the obstacle (necessary
 * condition)
 * 2. Testing if all control points lie outside at least one hyperplane
 * (sufficient condition)
 * 3. If inconclusive, recursively subdividing the curve until tolerance is
 * reached
 *
 * @param curve The Bezier curve to check for collision
 * @param A Constraint matrix defining the H-polyhedron (num_constraints x
 * dimension)
 * @param b Constraint vector defining the H-polyhedron (num_constraints x 1)
 * @param tol Tolerance for recursive subdivision termination (default: 1e-2)
 *             When the bounding sphere radius of control points <= tol,
 * subdivision stops
 *
 * @return uint8_t Collision status:
 *         - 1 (true): Curve is collision-free (does not intersect the obstacle
 * region Ax <= b)
 *         - 0 (false): Curve may collide or collision cannot be determined
 * within tolerance
 *
 * @note The obstacle region is defined as the set {x : Ax <= b}. Points outside
 * this region satisfy at least one constraint Ax > b.
 * @note This is a conservative algorithm - it may return false negatives
 * (reporting potential collision when the curve is actually collision-free) but
 * never returns false positives.
 * @note Computational complexity depends on curve complexity and required
 * precision (tol parameter).
 *
 * @pre A.cols() must equal curve.dimension()
 * @pre A.rows() must equal b.size()
 * @pre tol must be positive
 */
uint8_t BezierCurveHPolyhedronCollisionFree(const BezierCurve& c,
                                            const Eigen::MatrixXd& A,
                                            const Eigen::VectorXd& b,
                                            double tol = 1e-2);

/**
 * @brief Find intersections between composite Bezier curve segments and
 * H-polyhedron obstacles
 *
 * For each segment of the composite Bezier curve, determines which H-polyhedron
 * obstacles it potentially intersects with. Uses the
 * BezierCurveHPolyhedronCollisionFree function to check collision status -
 * segments that are NOT collision-free are considered to intersect.
 *
 * @param c The composite Bezier curve to analyze
 * @param As Vector of constraint matrices, one for each H-polyhedron (each A_i
 * has shape num_constraints_i x dimension)
 * @param bs Vector of constraint vectors, one for each H-polyhedron (each b_i
 * has shape num_constraints_i x 1)
 * @param hpoly_to_ignore Map from segment indices to obstacle indices to ignore
 * for that segment. If an obstacle index is present in the vector for a
 * segment, that obstacle is not checked for collision with that segment.
 * @param tol Tolerance for collision detection algorithm (default: 1e-2)
 * @param parallelize Whether to use parallel processing over segments (default:
 * true)
 *
 * @return std::unordered_map<int, std::vector<int>> A map where result[seg_idx]
 * contains the indices of all H-polyhedrons that segment seg_idx intersects
 * with. Segments with no collisions are not included in the map.
 *
 * @pre As.size() must equal bs.size()
 * @pre For each i: As[i].cols() must equal c.dimension()
 * @pre For each i: As[i].rows() must equal bs[i].size()
 * @pre All keys in hpoly_to_ignore must be valid segment indices (<
 * c.num_segments())
 * @pre tol must be positive
 *
 * @note An intersection is detected when BezierCurveHPolyhedronCollisionFree
 * returns false, meaning the segment is not proven to be collision-free.
 * @note Parallelizes over curve segments (recommended when num_obstacles >>
 * num_segments).
 */
std::unordered_map<int, std::vector<int>>
IntersectCompositeBezierCurveWithHPolyhedra(
    const CompositeBezierCurve& c, const std::vector<Eigen::MatrixXd>& As,
    const std::vector<Eigen::VectorXd>& bs,
    const std::unordered_map<int, std::vector<int>>& hpoly_to_ignore,
    double tol = 1e-2, bool parallelize = true);

/**
 * @brief Check if a Bezier curve is collision-free with a sphere obstacle
 *
 * Uses recursive subdivision algorithm to determine if a Bezier curve
 * intersects with a sphere defined by center and radius. The algorithm
 * constructs a separating hyperplane by computing the mean of control points
 * and creating a tangent plane to the sphere facing the control points.
 *
 * The separation certificate works as follows:
 * - Normal: a = normalize(mean - center) points from sphere to control points
 * - Tangent point: center + a * radius
 * - Hyperplane: a^T * x + b = 0 where b = -a^T * (center + a * radius)
 * - Points are safe if a^T * p + b >= 0
 *
 * @param curve The Bezier curve to check
 * @param center The center of the sphere (dimension x 1 vector)
 * @param radius The radius of the sphere
 * @param tol Tolerance for collision detection algorithm (default: 1e-2)
 *
 * @return uint8_t Returns 1 if curve is proven collision-free, 0 if collision
 * detected or cannot be proven collision-free within tolerance
 *
 * @pre center.size() must equal curve.dimension()
 * @pre radius must be positive
 * @pre tol must be positive
 *
 * @note Uses convex hull property of Bezier curves - if control points are
 * separated by a hyperplane, the curve is guaranteed collision-free.
 * @note May return false positives (report collision when curve is actually
 * safe) but never returns false negatives.
 */
uint8_t BezierCurveSphereCollisionFree(const BezierCurve& curve,
                                       const Eigen::VectorXd& center,
                                       double radius, double tol = 1e-2);

/**
 * @brief Find intersections between composite Bezier curve segments and sphere
 * obstacles
 *
 * For each segment of the composite Bezier curve, determines which sphere
 * obstacles it potentially intersects with. Uses the
 * BezierCurveSphereCollisionFree function to check collision status - segments
 * that are NOT collision-free are considered to intersect.
 *
 * @param c The composite Bezier curve to analyze
 * @param centers Vector of sphere centers, one for each sphere (each center_i
 * has shape dimension x 1)
 * @param radii Vector of sphere radii, one for each sphere
 * @param spheres_to_ignore Vector containing one vector per segment that
 * contains obstacle indices. If an index is present the corresponding sphere is
 * not checked for collision.
 * @param tol Tolerance for collision detection algorithm (default: 1e-2)
 * @param parallelize Whether to use parallel processing over segments (default:
 * true)
 *
 * @return std::vector<std::vector<int>> A vector where result[i] contains the
 * indices of all spheres that segment i intersects with. Empty vector for
 * segment i means segment i is collision-free with respect to all spheres.
 *
 * @pre centers.size() must equal radii.size()
 * @pre For each i: centers[i].size() must equal c.dimension()
 * @pre For each i: radii[i] must be positive
 * @pre tol must be positive
 *
 * @note An intersection is detected when BezierCurveSphereCollisionFree returns
 * false, meaning the segment is not proven to be collision-free.
 * @note Parallelizes over curve segments.
 */
std::vector<std::vector<int>> IntersectCompositeBezierCurveWithSpheres(
    const CompositeBezierCurve& c, const std::vector<Eigen::VectorXd>& centers,
    const std::vector<double>& radii,
    const std::vector<std::vector<int>>& spheres_to_ignore, double tol = 1e-2,
    bool parallelize = true);

}  // namespace csdecomp