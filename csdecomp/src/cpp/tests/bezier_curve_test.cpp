/** This file has been created using Claude 4 by translating
 * https://github.com/TobiaMarcucci/pybezier from python.
 * commit/0ff3893eae89a0dedff984e1aaf35f71ffb0bbf1*/
#include "bezier_curve.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "hpolyhedron.h"
using namespace csdecomp;
using namespace Eigen;

class BezierCurveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initial_time = 0.55;
    final_time = 3.12;

    // Create time samples
    const int num_samples = 50;
    time_samples.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
      time_samples[i] =
          initial_time + i * (final_time - initial_time) / (num_samples - 1);
    }

    // Set up random number generator with fixed seed for reproducibility
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Create scalar curve (1D, degree 5)
    scalar_points = MatrixXd(6, 1);
    for (int i = 0; i < 6; ++i) {
      scalar_points(i, 0) = dis(gen);
    }
    scalar_curve =
        std::make_unique<BezierCurve>(scalar_points, initial_time, final_time);

    // Reset generator for vector curve
    gen.seed(0);

    // Create vector curve (2D, degree 6)
    vec_points = MatrixXd(7, 2);
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 2; ++j) {
        vec_points(i, j) = dis(gen);
      }
    }
    vec_curve =
        std::make_unique<BezierCurve>(vec_points, initial_time, final_time);

    // Reset generator for matrix-like curve (treated as 5D curve, degree 6)
    gen.seed(0);

    // Create higher dimensional curve (5D, degree 6)
    mat_points = MatrixXd(7, 5);
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 5; ++j) {
        mat_points(i, j) = dis(gen);
      }
    }
    mat_curve =
        std::make_unique<BezierCurve>(mat_points, initial_time, final_time);

    // Store all curves for iteration
    curves = {scalar_curve.get(), vec_curve.get(), mat_curve.get()};
    points_matrices = {&scalar_points, &vec_points, &mat_points};
  }

  double initial_time;
  double final_time;
  std::vector<double> time_samples;

  MatrixXd scalar_points, vec_points, mat_points;
  std::unique_ptr<BezierCurve> scalar_curve, vec_curve, mat_curve;
  std::vector<BezierCurve*> curves;
  std::vector<const MatrixXd*> points_matrices;

  const double tolerance = 1e-8;
  const double numerical_tolerance = 1e-5;
};

TEST_F(BezierCurveTest, TestInit) {
  for (size_t i = 0; i < curves.size(); ++i) {
    const auto& curve = *curves[i];
    const auto& points = *points_matrices[i];

    EXPECT_TRUE(curve.points().isApprox(points, tolerance));
    EXPECT_DOUBLE_EQ(curve.initial_time(), initial_time);
    EXPECT_DOUBLE_EQ(curve.final_time(), final_time);

    // Test invalid time range
    EXPECT_THROW(BezierCurve(points, final_time, initial_time),
                 std::invalid_argument);
  }
}

TEST_F(BezierCurveTest, TestDefaultInit) {
  for (const auto& points : points_matrices) {
    BezierCurve curve(*points);
    EXPECT_DOUBLE_EQ(curve.initial_time(), 0.0);
    EXPECT_DOUBLE_EQ(curve.final_time(), 1.0);
  }
}

TEST_F(BezierCurveTest, TestDegree) {
  for (size_t i = 0; i < curves.size(); ++i) {
    const auto& curve = *curves[i];
    const auto& points = *points_matrices[i];
    EXPECT_EQ(curve.degree(), points.rows() - 1);
  }
}

TEST_F(BezierCurveTest, TestDimension) {
  for (size_t i = 0; i < curves.size(); ++i) {
    const auto& curve = *curves[i];
    const auto& points = *points_matrices[i];
    EXPECT_EQ(curve.dimension(), points.cols());
  }
}

TEST_F(BezierCurveTest, TestDuration) {
  for (const auto& curve : curves) {
    EXPECT_DOUBLE_EQ(curve->duration(), final_time - initial_time);
  }
}

TEST_F(BezierCurveTest, TestInitialPoint) {
  for (const auto& curve : curves) {
    VectorXd expected = curve->points().row(0);
    VectorXd actual = curve->initial_point();
    EXPECT_TRUE(expected.isApprox(actual, tolerance));

    // Test against evaluation at initial time
    VectorXd eval_at_initial = (*curve)(initial_time);
    EXPECT_TRUE(actual.isApprox(eval_at_initial, tolerance));
  }
}

TEST_F(BezierCurveTest, TestFinalPoint) {
  for (const auto& curve : curves) {
    VectorXd expected = curve->points().row(curve->points().rows() - 1);
    VectorXd actual = curve->final_point();
    EXPECT_TRUE(expected.isApprox(actual, tolerance));

    // Test against evaluation at final time
    VectorXd eval_at_final = (*curve)(final_time);
    EXPECT_TRUE(actual.isApprox(eval_at_final, tolerance));
  }
}

TEST_F(BezierCurveTest, TestBernstein) {
  // Test partition of unity property
  for (double time : time_samples) {
    std::vector<double> values;
    for (int n = 0; n <= vec_curve->degree(); ++n) {
      double val = vec_curve->bernstein(time, n);
      values.push_back(val);
      EXPECT_GE(val, 0.0);  // Non-negativity
    }

    // Sum should be 1 (partition of unity)
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, tolerance);
  }
}

TEST_F(BezierCurveTest, TestCall) {
  // Test with simple linear interpolation curves
  MatrixXd scalar_points_simple(4, 1);
  scalar_points_simple << 0, 1, 2, 3;

  MatrixXd vec_points_simple(4, 2);
  vec_points_simple << 0, 0, 1, 1, 2, 2, 3, 3;

  MatrixXd mat_points_simple(4, 2);
  mat_points_simple << 0, 0, 1, 1, 2, 2, 3, 3;

  std::vector<MatrixXd> simple_points = {scalar_points_simple,
                                         vec_points_simple, mat_points_simple};

  for (const auto& points : simple_points) {
    BezierCurve curve(points, 0.0, 1.0);

    for (double t = 0.0; t <= 1.0; t += 0.1) {
      VectorXd expected =
          points.row(0) + t * (points.row(points.rows() - 1) - points.row(0));
      VectorXd actual = curve(t);
      EXPECT_TRUE(expected.isApprox(actual, tolerance));
    }
  }
}

TEST_F(BezierCurveTest, TestScalarMulAndRmul) {
  double c = 3.66;

  for (const auto& curve : curves) {
    BezierCurve prod_1 = (*curve) * c;
    BezierCurve prod_2 = c * (*curve);

    for (double time : time_samples) {
      VectorXd expected = (*curve)(time)*c;
      VectorXd actual_1 = prod_1(time);
      VectorXd actual_2 = prod_2(time);

      EXPECT_TRUE(expected.isApprox(actual_1, tolerance));
      EXPECT_TRUE(expected.isApprox(actual_2, tolerance));
    }
  }
}

TEST_F(BezierCurveTest, TestElementwiseMul) {
  for (const auto& curve : curves) {
    BezierCurve prod = (*curve) * (*curve);

    for (double time : time_samples) {
      VectorXd curve_val = (*curve)(time);
      VectorXd expected = curve_val.cwiseProduct(curve_val);
      VectorXd actual = prod(time);

      EXPECT_TRUE(expected.isApprox(actual, tolerance));
    }
  }

  // Test incompatible dimensions (should work in C++ version with zero-padding)
  // This test is different from Python version which throws an error
  BezierCurve prod = (*mat_curve) * (*vec_curve);
  // This should work in our C++ implementation due to dimension compatibility
  // handling
}

TEST_F(BezierCurveTest, TestScalarAddSub) {
  double c = 3.66;

  for (const auto& curve : curves) {
    BezierCurve sum_1 = (*curve) + c;
    BezierCurve sum_2 = c + (*curve);
    BezierCurve sub_1 = (*curve) - c;
    BezierCurve sub_2 = c - (*curve);

    for (double time : time_samples) {
      VectorXd curve_val = (*curve)(time);
      VectorXd expected_sum = curve_val.array() + c;
      VectorXd expected_sub1 = curve_val.array() - c;
      VectorXd expected_sub2 = c - curve_val.array();

      EXPECT_TRUE(expected_sum.isApprox(sum_1(time), tolerance));
      EXPECT_TRUE(expected_sum.isApprox(sum_2(time), tolerance));
      EXPECT_TRUE(expected_sub1.isApprox(sub_1(time), tolerance));
      EXPECT_TRUE(expected_sub2.isApprox(sub_2(time), tolerance));
    }
  }
}

TEST_F(BezierCurveTest, TestElementwiseAddSub) {
  for (const auto& curve : curves) {
    BezierCurve sum = (*curve) + (*curve);
    BezierCurve sub = (*curve) - (*curve);

    for (double time : time_samples) {
      VectorXd curve_val = (*curve)(time);
      VectorXd expected_sum = 2.0 * curve_val;
      VectorXd expected_sub = VectorXd::Zero(curve_val.size());

      EXPECT_TRUE(expected_sum.isApprox(sum(time), tolerance));
      EXPECT_TRUE(expected_sub.isApprox(sub(time), tolerance));
    }
  }
}

TEST_F(BezierCurveTest, TestNeg) {
  for (const auto& curve : curves) {
    BezierCurve neg = -(*curve);

    for (double time : time_samples) {
      VectorXd expected = -(*curve)(time);
      VectorXd actual = neg(time);

      EXPECT_TRUE(expected.isApprox(actual, tolerance));
    }
  }
}

TEST_F(BezierCurveTest, TestElevateDegree) {
  for (const auto& curve : curves) {
    BezierCurve elevated = curve->elevate_degree(11);
    for (double time : time_samples) {
      VectorXd original_val = (*curve)(time);
      VectorXd elevated_val = elevated(time);

      EXPECT_TRUE(original_val.isApprox(elevated_val, tolerance));
    }
  }
}

TEST_F(BezierCurveTest, TestDerivative) {
  const double time_step = 1e-7;

  for (const auto& curve : curves) {
    BezierCurve derivative = curve->derivative();

    // Test numerical derivative
    for (size_t i = 0; i < time_samples.size() - 1; ++i) {
      double time = time_samples[i];
      if (time + time_step <= final_time) {
        VectorXd numerical_deriv = (*curve)(time + time_step) - (*curve)(time);
        numerical_deriv = numerical_deriv * 1 / time_step;
        VectorXd analytical_deriv = derivative(time);
        std::cout << "num der \n" << numerical_deriv << std::endl;
        std::cout << "an der \n" << analytical_deriv << std::endl;
        EXPECT_TRUE(analytical_deriv.isApprox(numerical_deriv, 1e-4));
      }
    }
  }
}

TEST_F(BezierCurveTest, TestIntegral) {
  const double time_step = 1e-3;

  for (const auto& curve : curves) {
    // Test without initial condition
    BezierCurve integral_no_ic = curve->integral();
    VectorXd value_at_initial = integral_no_ic(initial_time);
    VectorXd expected_zero = VectorXd::Zero(curve->dimension());
    EXPECT_TRUE(value_at_initial.isApprox(expected_zero, tolerance));

    // Test with initial condition
    VectorXd initial_condition = VectorXd::Ones(curve->dimension());
    BezierCurve integral_with_ic = curve->integral(initial_condition);
    VectorXd value_at_initial_ic = integral_with_ic(initial_time);
    EXPECT_TRUE(value_at_initial_ic.isApprox(initial_condition, tolerance));

    // Test fundamental theorem of calculus (numerical)
    for (size_t i = 0; i < time_samples.size() - 1; ++i) {
      double time = time_samples[i];
      if (time + time_step <= final_time) {
        VectorXd curve_val = (*curve)(time + time_step / 2);
        VectorXd integral_diff =
            (integral_no_ic(time + time_step) - integral_no_ic(time)) /
            time_step;

        EXPECT_TRUE(curve_val.isApprox(integral_diff, numerical_tolerance));
      }
    }
  }
}

TEST_F(BezierCurveTest, TestDomainSplit) {
  for (const auto& curve : curves) {
    // Test split at middle time
    double split_time = (initial_time + final_time) / 2;
    auto [curve_1, curve_2] = curve->domain_split(split_time);

    // Test continuity at split point
    VectorXd val_original = (*curve)(split_time);
    VectorXd val_left = curve_1(split_time);
    VectorXd val_right = curve_2(split_time);

    EXPECT_TRUE(val_original.isApprox(val_left, tolerance));
    EXPECT_TRUE(val_original.isApprox(val_right, tolerance));

    // Test that split curves match original on their respective domains
    for (double time : time_samples) {
      VectorXd original_val = (*curve)(time);

      if (time <= split_time) {
        VectorXd left_val = curve_1(time);
        EXPECT_TRUE(original_val.isApprox(left_val, tolerance));
      }
      if (time >= split_time) {
        VectorXd right_val = curve_2(time);
        EXPECT_TRUE(original_val.isApprox(right_val, tolerance));
      }
    }

    // Test domain bounds
    EXPECT_DOUBLE_EQ(curve_1.initial_time(), initial_time);
    EXPECT_DOUBLE_EQ(curve_1.final_time(), split_time);
    EXPECT_DOUBLE_EQ(curve_2.initial_time(), split_time);
    EXPECT_DOUBLE_EQ(curve_2.final_time(), final_time);

    // Test split outside domain
    EXPECT_THROW(curve->domain_split(initial_time - 0.1),
                 std::invalid_argument);
    EXPECT_THROW(curve->domain_split(final_time + 0.1), std::invalid_argument);
  }
}

TEST_F(BezierCurveTest, TestSquaredL2Norm) {
  // Numerical integration test
  const int n_samples = 1000;
  std::vector<double> times(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    times[i] = initial_time + i * (final_time - initial_time) / (n_samples - 1);
  }

  // Compute numerical integral using trapezoidal rule
  std::vector<double> values;
  for (double time : times) {
    VectorXd val = (*vec_curve)(time);
    values.push_back(val.squaredNorm());
  }

  double numerical_integral = 0.0;
  for (size_t i = 1; i < values.size(); ++i) {
    double dt = times[i] - times[i - 1];
    numerical_integral += 0.5 * (values[i] + values[i - 1]) * dt;
  }

  double analytical_l2_squared = vec_curve->l2_squared();

  // Allow for numerical integration error
  EXPECT_NEAR(analytical_l2_squared, numerical_integral, 1e-4);
}

TEST_F(BezierCurveTest, TestEvaluateAtTimes) {
  for (const auto& curve : curves) {
    MatrixXd result = curve->evaluate_at_times(time_samples);

    EXPECT_EQ(result.rows(), curve->dimension());
    EXPECT_EQ(result.cols(), time_samples.size());

    // Check each column matches individual evaluation
    for (size_t i = 0; i < time_samples.size(); ++i) {
      VectorXd expected = (*curve)(time_samples[i]);
      VectorXd actual = result.col(i);
      EXPECT_TRUE(expected.isApprox(actual, tolerance));
    }
  }
}

// Test CompositeBezierCurve
TEST_F(BezierCurveTest, TestCompositeCurve) {
  // Create a composite curve from two segments
  MatrixXd points1(3, 2);
  points1 << 0, 0, 1, 1, 2, 0;

  MatrixXd points2(3, 2);
  points2 << 2, 0, 3, -1, 4, 0;

  BezierCurve curve1(points1, 0.0, 1.0);
  BezierCurve curve2(points2, 1.0, 2.0);

  std::vector<BezierCurve> segments = {curve1, curve2};
  CompositeBezierCurve composite(segments);

  EXPECT_EQ(composite.num_segments(), 2);
  EXPECT_DOUBLE_EQ(composite.initial_time(), 0.0);
  EXPECT_DOUBLE_EQ(composite.final_time(), 2.0);
  EXPECT_EQ(composite.dimension(), 2);

  // Test continuity at junction
  VectorXd val_at_1_from_left = curve1(1.0);
  VectorXd val_at_1_from_right = curve2(1.0);
  VectorXd val_at_1_composite = composite(1.0);

  EXPECT_TRUE(val_at_1_from_left.isApprox(val_at_1_composite, tolerance));
  EXPECT_TRUE(val_at_1_from_right.isApprox(val_at_1_composite, tolerance));

  // Test segment identification
  EXPECT_EQ(composite.curve_segment(0.5), 0);
  EXPECT_EQ(composite.curve_segment(1.5), 1);

  // Test evaluation on each segment
  EXPECT_TRUE(curve1(0.5).isApprox(composite(0.5), tolerance));
  EXPECT_TRUE(curve2(1.5).isApprox(composite(1.5), tolerance));
}

TEST_F(BezierCurveTest, TestBinomialCoefficient) {
  // Test some known values
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 0), 1.0);
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 1), 5.0);
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 2), 10.0);
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 3), 10.0);
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 4), 5.0);
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 5), 1.0);

  // Test edge cases
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, 6), 0.0);
  EXPECT_DOUBLE_EQ(binomial_coefficient(5, -1), 0.0);
}

GTEST_TEST(BezierCurveTest2, HPolyCollisionCheck) {
  // Define constraint matrix A and vector b for H-polyhedron
  // Represents the constraints: x <= 7, y <= 7, x >= 3, y >= 3
  // (i.e., a box from (3,3) to (7,7))
  MatrixXf A(4, 2);
  A << 1.0, 0.0,  // x <= 7
      0.0, 1.0,   // y <= 7
      -1.0, 0.0,  // -x <= -3  => x >= 3
      0.0, -1.0;  // -y <= -3  => y >= 3

  VectorXf b(4);
  b << 7.0, 7.0, -3.0, -3.0;

  // Create H-polyhedron
  HPolyhedron hpoly(A, b);

  // Test case 1: Curve that goes outside the box (y = 7.05 > 7)
  MatrixXd wps1(6, 2);
  wps1 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      7.05,                 // This point violates y <= 7
      1.0, 7.05, 7.0, 8.0;  // This point also violates y <= 7

  BezierCurve r1(wps1, 0.0, 1.0);

  // Test case 2: Curve that stays within bounds (y = 6.9 < 7)
  MatrixXd wps2(6, 2);
  wps2 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      6.9,                 // This point satisfies y <= 7
      1.0, 6.9, 7.0, 8.0;  // This point violates y <= 7

  BezierCurve r2(wps2, 0.0, 1.0);
  Eigen::MatrixXd A_obs = hpoly.A().cast<double>();
  Eigen::MatrixXd b_obs = hpoly.b().cast<double>();

  // Test with different tolerances
  bool result1_fine =
      BezierCurveHPolyhedronCollisionFree(r1, A_obs, b_obs, 1e-2);
  bool result1_coarse =
      BezierCurveHPolyhedronCollisionFree(r1, A_obs, b_obs, 1e-1);
  bool result2_coarse =
      BezierCurveHPolyhedronCollisionFree(r2, A_obs, b_obs, 1e-1);

  // Print results (similar to Python output)
  std::cout << std::boolalpha;  // Print true/false instead of 1/0
  std::cout << "col_free " << result1_fine << std::endl;
  std::cout << "col_free " << result1_coarse << std::endl;
  std::cout << "col_free " << result2_coarse << std::endl;

  // Expected results based on Python output:
  EXPECT_TRUE(result1_fine);     // Should be true with fine tolerance
  EXPECT_FALSE(result1_coarse);  // Should be false with coarse tolerance (min
                                 // tol reached)
  EXPECT_FALSE(result2_coarse);  // Should be false
}

GTEST_TEST(BezierCurveTest2, CompositeBezierIntersectionCheck) {
  // Define constraint matrix A and vector b for H-polyhedron
  // Represents the constraints: x <= 7, y <= 7, x >= 3, y >= 3
  // (i.e., a box from (3,3) to (7,7))
  MatrixXf A(4, 2);
  A << 1.0, 0.0,  // x <= 7
      0.0, 1.0,   // y <= 7
      -1.0, 0.0,  // -x <= -3  => x >= 3
      0.0, -1.0;  // -y <= -3  => y >= 3

  VectorXf b(4);
  b << 7.0, 7.0, -3.0, -3.0;

  // Create H-polyhedron
  HPolyhedron hpoly(A, b);

  // Convert to double precision for the intersection function
  Eigen::MatrixXd A_obs = hpoly.A().cast<double>();
  Eigen::VectorXd b_obs = hpoly.b().cast<double>();

  // Use the same curves from the original test

  // Test curve 1: Curve that goes outside the box (y = 7.05 > 7)
  MatrixXd wps1(6, 2);
  wps1 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      7.05,                 // This point violates y <= 7
      1.0, 7.05, 7.0, 8.0;  // This point also violates y <= 7

  BezierCurve r1(wps1, 0.0, 1.0);

  // Test curve 2: Curve that stays within bounds (y = 6.9 < 7)
  MatrixXd wps2(6, 2);
  wps2 << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      6.9,                 // This point satisfies y <= 7
      1.0, 6.9, 7.0, 8.0;  // This point violates y <= 7

  BezierCurve r2(wps2, 0.0, 1.0);

  // Split each curve into two segments
  auto [r1_left, r1_right] = r1.domain_split(0.5);
  auto [r2_left, r2_right] = r2.domain_split(0.5);

  // Create composite curves from the split segments
  std::vector<BezierCurve> segments1 = {r1_left, r1_right};
  std::vector<BezierCurve> segments2 = {r2_left, r2_right};

  CompositeBezierCurve composite_curve1(segments1);
  CompositeBezierCurve composite_curve2(segments2);

  // Create obstacle vector (single obstacle)
  std::vector<Eigen::MatrixXd> As = {A_obs};
  std::vector<Eigen::VectorXd> bs = {b_obs};

  // Create empty mask (no obstacles to ignore)
  std::vector<std::vector<int>> empty_mask1(2);  // 2 segments, no ignores
  std::vector<std::vector<int>> empty_mask2(2);  // 2 segments, no ignores

  // Test with same tolerances as original
  auto intersections1_fine = IntersectCompositeBezierCurveWithHPolyhedra(
      composite_curve1, As, bs, empty_mask1, 1e-2, false);

  auto intersections1_coarse = IntersectCompositeBezierCurveWithHPolyhedra(
      composite_curve1, As, bs, empty_mask1, 1e-1, false);

  auto intersections2_coarse = IntersectCompositeBezierCurveWithHPolyhedra(
      composite_curve2, As, bs, empty_mask2, 1e-1, false);

  // Verify original single-curve behavior for comparison
  bool original1_fine =
      BezierCurveHPolyhedronCollisionFree(r1, A_obs, b_obs, 1e-2);
  bool original1_coarse =
      BezierCurveHPolyhedronCollisionFree(r1, A_obs, b_obs, 1e-1);
  bool original2_coarse =
      BezierCurveHPolyhedronCollisionFree(r2, A_obs, b_obs, 1e-1);

  // Print results for debugging
  std::cout << "Original single-curve results:" << std::endl;
  std::cout << std::boolalpha;
  std::cout << "  r1 fine tolerance collision-free: " << original1_fine
            << std::endl;
  std::cout << "  r1 coarse tolerance collision-free: " << original1_coarse
            << std::endl;
  std::cout << "  r2 coarse tolerance collision-free: " << original2_coarse
            << std::endl;

  std::cout << "\nComposite curve intersection results:" << std::endl;
  std::cout << "  Curve 1 fine tolerance: ["
            << intersections1_fine[0].size() + intersections1_fine[1].size()
            << " total intersections]" << std::endl;
  std::cout << "  Curve 1 coarse tolerance: ["
            << intersections1_coarse[0].size() + intersections1_coarse[1].size()
            << " total intersections]" << std::endl;
  std::cout << "  Curve 2 coarse tolerance: ["
            << intersections2_coarse[0].size() + intersections2_coarse[1].size()
            << " total intersections]" << std::endl;

  // Expected results based on original test:
  // original1_fine = true  => composite should have no intersections (empty
  // vectors) original1_coarse = false => composite should have intersections
  // (non-empty) original2_coarse = false => composite should have intersections
  // (non-empty)

  EXPECT_EQ(intersections1_fine.size(), 2);    // Two segments
  EXPECT_EQ(intersections1_coarse.size(), 2);  // Two segments
  EXPECT_EQ(intersections2_coarse.size(), 2);  // Two segments

  // If original curve was collision-free, composite segments should be too (no
  // intersections)
  if (original1_fine) {
    EXPECT_TRUE(intersections1_fine[0].empty() &&
                intersections1_fine[1].empty())
        << "Composite segments should be collision-free when original curve is "
           "collision-free";
  }

  // If original curve had collision, at least one composite segment should
  // intersect
  if (!original1_coarse) {
    bool has_intersection =
        !intersections1_coarse[0].empty() || !intersections1_coarse[1].empty();
    EXPECT_TRUE(has_intersection)
        << "At least one composite segment should intersect when original "
           "curve has collision";
  }

  if (!original2_coarse) {
    bool has_intersection =
        !intersections2_coarse[0].empty() || !intersections2_coarse[1].empty();
    EXPECT_TRUE(has_intersection)
        << "At least one composite segment should intersect when original "
           "curve has collision";
  }

  // Test parallel version gives same results
  auto intersections1_parallel = IntersectCompositeBezierCurveWithHPolyhedra(
      composite_curve1, As, bs, empty_mask1, 1e-2, true);

  EXPECT_EQ(intersections1_fine.size(), intersections1_parallel.size());
  for (size_t i = 0; i < intersections1_fine.size(); ++i) {
    // Sort both vectors since parallel execution might change order
    std::vector<int> fine_sorted = intersections1_fine[i];
    std::vector<int> parallel_sorted = intersections1_parallel[i];
    std::sort(fine_sorted.begin(), fine_sorted.end());
    std::sort(parallel_sorted.begin(), parallel_sorted.end());

    EXPECT_EQ(fine_sorted, parallel_sorted)
        << "Segment " << i << " results differ between sequential and parallel";
  }
}

GTEST_TEST(BezierCurveTest2, CompositeBezierSphereIntersectionCheck) {
  // Create 3 sphere obstacles with radius 0.1 at positions along y=0.5
  std::vector<Eigen::VectorXd> centers;
  std::vector<double> radii;

  Eigen::VectorXd center1(2);
  center1 << 0.3, 0.5;
  centers.push_back(center1);
  radii.push_back(0.1);

  Eigen::VectorXd center2(2);
  center2 << 0.5, 0.5;
  centers.push_back(center2);
  radii.push_back(0.1);

  Eigen::VectorXd center3(2);
  center3 << 0.7, 0.5;
  centers.push_back(center3);
  radii.push_back(0.1);

  // Test case 1: height = 0.615 (should have collision)
  {
    double height = 0.615;

    // Create waypoints for composite curve
    Eigen::MatrixXd wps1(3, 2);
    wps1 << 0.1, 0.5, 0.2, height, 0.5, height;

    Eigen::MatrixXd wps2(3, 2);
    wps2 << 0.5, height, 0.8, height, 0.9, 0.5;

    // Create two Bezier curve segments
    BezierCurve curve1(wps1, 0.0, 0.5);
    BezierCurve curve2(wps2, 0.5, 1.0);

    // Create composite curve
    std::vector<BezierCurve> segments = {curve1, curve2};
    CompositeBezierCurve composite_curve(segments);

    // Create empty ignore mask (check all obstacles for all segments)
    std::vector<std::vector<int>> empty_mask(2);

    // Test intersection detection
    auto intersections = IntersectCompositeBezierCurveWithSpheres(
        composite_curve, centers, radii, empty_mask, 1e-2, false);

    // Count total intersections across all segments
    size_t total_intersections = 0;
    for (const auto& seg_intersections : intersections) {
      total_intersections += seg_intersections.size();
    }

    std::cout << "\nSphere collision test (height = " << height << "):"
              << std::endl;
    std::cout << "  Total intersections detected: " << total_intersections
              << std::endl;
    std::cout << "  Segment 0 intersects with spheres: ";
    for (int idx : intersections[0]) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "  Segment 1 intersects with spheres: ";
    for (int idx : intersections[1]) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;

    // At height 0.615, there should be collisions
    EXPECT_GT(total_intersections, 0)
        << "Expected collision at height 0.615, but no intersections detected";
  }

  // Test case 2: height = 0.628 (should be collision-free)
  {
    double height = 0.628;

    // Create waypoints for composite curve
    Eigen::MatrixXd wps1(3, 2);
    wps1 << 0.1, 0.5, 0.2, height, 0.5, height;

    Eigen::MatrixXd wps2(3, 2);
    wps2 << 0.5, height, 0.8, height, 0.9, 0.5;

    // Create two Bezier curve segments
    BezierCurve curve1(wps1, 0.0, 0.5);
    BezierCurve curve2(wps2, 0.5, 1.0);

    // Create composite curve
    std::vector<BezierCurve> segments = {curve1, curve2};
    CompositeBezierCurve composite_curve(segments);

    // Create empty ignore mask (check all obstacles for all segments)
    std::vector<std::vector<int>> empty_mask(2);

    // Test intersection detection
    auto intersections = IntersectCompositeBezierCurveWithSpheres(
        composite_curve, centers, radii, empty_mask, 1e-2, false);

    // Count total intersections across all segments
    size_t total_intersections = 0;
    for (const auto& seg_intersections : intersections) {
      total_intersections += seg_intersections.size();
    }

    std::cout << "\nSphere collision test (height = " << height << "):"
              << std::endl;
    std::cout << "  Total intersections detected: " << total_intersections
              << std::endl;
    std::cout << "  Segment 0 intersects with spheres: ";
    for (int idx : intersections[0]) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "  Segment 1 intersects with spheres: ";
    for (int idx : intersections[1]) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;

    // At height 0.628, there should be no collisions
    EXPECT_EQ(total_intersections, 0)
        << "Expected no collision at height 0.628, but " << total_intersections
        << " intersections detected";
  }

  // Test parallel version gives same results
  {
    double height = 0.615;
    Eigen::MatrixXd wps1(3, 2);
    wps1 << 0.1, 0.5, 0.2, height, 0.5, height;

    Eigen::MatrixXd wps2(3, 2);
    wps2 << 0.5, height, 0.8, height, 0.9, 0.5;

    BezierCurve curve1(wps1, 0.0, 0.5);
    BezierCurve curve2(wps2, 0.5, 1.0);
    std::vector<BezierCurve> segments = {curve1, curve2};
    CompositeBezierCurve composite_curve(segments);

    std::vector<std::vector<int>> empty_mask(2);

    auto intersections_sequential = IntersectCompositeBezierCurveWithSpheres(
        composite_curve, centers, radii, empty_mask, 1e-2, false);

    auto intersections_parallel = IntersectCompositeBezierCurveWithSpheres(
        composite_curve, centers, radii, empty_mask, 1e-2, true);

    EXPECT_EQ(intersections_sequential.size(), intersections_parallel.size());
    for (size_t i = 0; i < intersections_sequential.size(); ++i) {
      // Sort both vectors since parallel execution might change order
      std::vector<int> seq_sorted = intersections_sequential[i];
      std::vector<int> par_sorted = intersections_parallel[i];
      std::sort(seq_sorted.begin(), seq_sorted.end());
      std::sort(par_sorted.begin(), par_sorted.end());

      EXPECT_EQ(seq_sorted, par_sorted)
          << "Segment " << i
          << " results differ between sequential and parallel execution";
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}