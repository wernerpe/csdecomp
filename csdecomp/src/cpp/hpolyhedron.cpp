#include "hpolyhedron.h"

#include <fmt/core.h>
#include <glpk.h>

#include <iostream>
#include <random>
#include <stdexcept>

namespace {
bool solveLP(const Eigen::MatrixXf& A, const Eigen::VectorXf& b,
             const Eigen::VectorXf& c, Eigen::VectorXf& x_sol) {
  int m = A.rows();  // Number of constraints
  int n = A.cols();  // Number of variables

  glp_prob* lp;

  // Create problem
  lp = glp_create_prob();
  glp_set_obj_dir(lp, GLP_MIN);

  // Set rows (constraints)
  glp_add_rows(lp, m);
  for (int i = 1; i <= m; i++) {
    glp_set_row_bnds(lp, i, GLP_UP, 0.0, b(i - 1));
  }

  // Set columns (variables)
  glp_add_cols(lp, n);
  for (int j = 1; j <= n; j++) {
    glp_set_col_bnds(lp, j, GLP_FR, 0.0, 0.0);
    glp_set_obj_coef(lp, j, c(j - 1));
  }

  // Load matrix
  std::vector<int> ia(1 + m * n);
  std::vector<int> ja(1 + m * n);
  std::vector<double> ar(1 + m * n);
  ia[0] = 0;
  ja[0] = 0;

  int k = 1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      ia[k] = i + 1;
      ja[k] = j + 1;
      ar[k] = A(i, j);
      k++;
    }
  }
  glp_load_matrix(lp, m * n, ia.data(), ja.data(), ar.data());

  // Solve problem
  glp_smcp parm;
  glp_init_smcp(&parm);
  parm.msg_lev = GLP_MSG_ERR;
  int ret = glp_simplex(lp, &parm);

  // Check if solution is optimal
  if (ret == 0 && glp_get_status(lp) == GLP_OPT) {
    // Extract solution
    x_sol.resize(n);
    for (int j = 0; j < n; j++) {
      x_sol(j) = glp_get_col_prim(lp, j + 1);
    }

    // Clean up
    glp_delete_prob(lp);
    return true;
  } else {
    // Clean up
    glp_delete_prob(lp);
    return false;
  }
}
}  // namespace

namespace csdecomp {
HPolyhedron::HPolyhedron() : _ambient_dimension(0) {}

HPolyhedron::HPolyhedron(const MinimalHPolyhedron& hpoly) {
  _ambient_dimension = hpoly.dim;
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      Amin(hpoly.A, hpoly.num_faces, hpoly.dim);
  Eigen::Map<const Eigen::VectorXf> bmin(hpoly.b, hpoly.num_faces);
  // create a copy
  _A = Amin;
  _b = bmin;
}

HPolyhedron::HPolyhedron(const Eigen::MatrixXf& A, Eigen::VectorXf& b)
    : _A(A), _b(b), _ambient_dimension(A.cols()) {
  if (A.rows() != b.size()) {
    throw std::invalid_argument("A and b dimensions do not match");
  }
}

const Eigen::MatrixXf& HPolyhedron::A() const { return _A; }

const Eigen::VectorXf& HPolyhedron::b() const { return _b; }

void HPolyhedron::MakeBox(const Eigen::VectorXf& lower_limit,
                          const Eigen::VectorXf& upper_limit) {
  if (lower_limit.size() != upper_limit.size()) {
    throw std::invalid_argument(
        "Lower and upper limits must have the same dimension");
  }

  int dim = lower_limit.size();
  _ambient_dimension = dim;
  _A = Eigen::MatrixXf::Zero(2 * dim, dim);
  _b = Eigen::VectorXf::Zero(2 * dim);

  for (int i = 0; i < dim; ++i) {
    _A(2 * i, i) = -1;
    _A(2 * i + 1, i) = 1;
    _b(2 * i) = -lower_limit(i);
    _b(2 * i + 1) = upper_limit(i);
  }
}

const Eigen::VectorXf HPolyhedron::ChebyshevCenter() const {
  int m = _A.rows();  // Number of constraints
  int n = _A.cols();  // Dimension of the space

  // Prepare the LP matrices
  Eigen::MatrixXf A_cheb(m, n + 1);
  Eigen::VectorXf b_cheb(m);
  Eigen::VectorXf c_cheb(n + 1);

  // Fill A_cheb
  A_cheb.block(0, 0, m, n) = _A;
  A_cheb.block(0, n, m, 1) = _A.rowwise().norm();

  // Fill b_cheb
  b_cheb.head(m) = _b;

  // Fill c_cheb (we want to maximize the radius, which is the last variable)
  c_cheb.head(n) = Eigen::VectorXf::Zero(n);
  c_cheb(n) = -1;  // Negative because we're minimizing

  // Solve the LP
  Eigen::VectorXf x_sol;
  bool success = solveLP(A_cheb, b_cheb, c_cheb, x_sol);

  if (!success) {
    // Handle the case where LP solving failed
    throw std::runtime_error("Failed to solve LP for Chebyshev center");
  }

  // Extract the center (first n components of the solution)
  return x_sol.head(n);
}

const MinimalHPolyhedron HPolyhedron::GetMyMinimalHPolyhedron() const {
  MinimalHPolyhedron minimal;
  minimal.num_faces = _A.rows();
  minimal.dim = _A.cols();

  if (minimal.num_faces > MAX_NUM_FACES || minimal.dim > MAX_DIM) {
    throw std::runtime_error("Polyhedron exceeds maximum size");
  }

  for (int row = 0; row < minimal.num_faces; ++row) {
    for (int col = 0; col < minimal.dim; ++col) {
      minimal.A[col + row * minimal.dim] = _A(row, col);
    }
  }
  for (int row = 0; row < minimal.num_faces; ++row) {
    minimal.b[row] = _b(row);
  }

  return minimal;
}

// this is closely following the implementation of drake
const Eigen::VectorXf HPolyhedron::UniformSample(
    const Eigen::VectorXf& previous_sample, const int mixing_steps) const {
  static std::default_random_engine generator(1337);
  static std::normal_distribution<float> gaussian_distribution;

  Eigen::VectorXf current_sample = previous_sample;
  int dim = previous_sample.size();

  if (dim != _ambient_dimension) {
    throw std::runtime_error("wrong dimension of previous sample!");
  }

  Eigen::VectorXf gaussian_sample(dim);
  Eigen::VectorXf direction(dim);

  for (int step = 0; step < mixing_steps; ++step) {
    for (int i = 0; i < gaussian_sample.size(); ++i) {
      gaussian_sample[i] = gaussian_distribution(generator);
    }

    direction = gaussian_sample;
    Eigen::VectorXf line_b = _b - _A * current_sample;
    Eigen::VectorXf line_A = _A * direction;
    float theta_max = std::numeric_limits<float>::infinity();
    float theta_min = -theta_max;

    for (int i = 0; i < line_A.size(); ++i) {
      if (line_A[i] < 0.0) {
        theta_min = std::max(theta_min, line_b[i] / line_A[i]);
      } else if (line_A[i] > 0.0) {
        theta_max = std::min(theta_max, line_b[i] / line_A[i]);
      }
    }

    if (std::isinf(theta_max) || std::isinf(theta_min) ||
        theta_max < theta_min) {
      float tol = (_A * current_sample - _b).maxCoeff();
      throw std::invalid_argument(fmt::format(
          "The Hit and Run algorithm failed to find a feasible point in the "
          "set. The `previous_sample` must be in the set.\n"
          "max(A * previous_sample - b) = {}",
          tol));
    }

    // Now pick θ uniformly from [θ_min, θ_max).
    std::uniform_real_distribution<double> uniform_theta(theta_min, theta_max);
    const double theta = uniform_theta(generator);
    current_sample = current_sample + theta * direction;
  }

  return current_sample;
}

const Eigen::VectorXf HPolyhedron::GetFeasiblePoint() const {
  int n = _A.cols();  // number of variables
  int m = _A.rows();  // number of constraints

  // Construct the augmented constraint matrix [A | -1]
  Eigen::MatrixXf A_aug(m, n + 1);
  A_aug << _A, -Eigen::VectorXf::Ones(m);

  // Construct the augmented cost vector [0 ... 0 1]
  Eigen::VectorXf c_aug = Eigen::VectorXf::Zero(n + 1);
  c_aug(n) = 1;

  // Solve the auxiliary linear program
  Eigen::VectorXf x_aug;
  bool success = solveLP(A_aug, _b, c_aug, x_aug);

  if (!success) {
    throw std::runtime_error(
        "Failed to find a feasible point. The polyhedron might be empty.");
  }

  // Check if we found a feasible point
  float s = x_aug(n);
  if (s > 1e-6) {  // Allow for some numerical error
    throw std::runtime_error("The polyhedron is empty.");
  }

  // Return the feasible point
  return x_aug.head(n);
}

const int HPolyhedron::ambient_dimension() const { return _ambient_dimension; }

const bool HPolyhedron::PointInSet(const Eigen::VectorXf& point) const {
  if (point.size() != _ambient_dimension) {
    throw std::invalid_argument(
        "Point dimension does not match polyhedron dimension");
  }

  // A point is in the set if it satisfies all inequalities: A * x <= b
  return ((_A * point).array() <= _b.array()).all();
}
}  // namespace csdecomp