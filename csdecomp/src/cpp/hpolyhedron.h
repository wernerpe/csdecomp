#pragma once
#include <Eigen/Dense>

namespace csdecomp {
#define MAX_NUM_FACES 4000
#define MAX_DIM 14

/**
 * @struct MinimalHPolyhedron
 * @brief A minimal representation of an H-polyhedron.
 *
 * This structure represents an H-polyhedron in a compact form, suitable for
 * efficient storage and transmission.
 */
struct MinimalHPolyhedron {
  float A[MAX_DIM * MAX_NUM_FACES];  ///< Flattened matrix of face normals,
                                     ///< stored in row major
  float b[MAX_NUM_FACES];            ///< Right-hand side vector
  int32_t num_faces;                 ///< Number of faces in the polyhedron
  int32_t dim;                       ///< Dimension of the polyhedron
};

/**
 * @class HPolyhedron
 * @brief Represents a convex polyhedron in half-space representation.
 *
 * This class provides methods for creating, manipulating, and querying
 * H-polyhedra (polyhedra defined by linear inequalities).
 */
class HPolyhedron {
 public:
  /**
   * @brief Default constructor.
   */
  HPolyhedron();

  /**
   * @brief Make from MinimalHpolyhedron.
   */
  HPolyhedron(const MinimalHPolyhedron& hpoly);

  /**
   * @brief Constructs an H-polyhedron from given matrices.
   * @param A The matrix of face normals.
   * @param b The right-hand side vector.
   */
  HPolyhedron(const Eigen::MatrixXf& A, Eigen::VectorXf& b);

  /**
   * @brief Gets the matrix of face normals.
   * @return A constant reference to the A matrix.
   */
  const Eigen::MatrixXf& A() const;

  /**
   * @brief Gets the right-hand side vector.
   * @return A constant reference to the b vector.
   */
  const Eigen::VectorXf& b() const;

  /**
   * @brief Creates a box-shaped polyhedron.
   * @param lower_limit The lower bounds of the box.
   * @param upper_limit The upper bounds of the box.
   */
  void MakeBox(const Eigen::VectorXf& lower_limit,
               const Eigen::VectorXf& upper_limit);

  /**
   * @brief Gets a minimal representation of the H-polyhedron.
   * @return A MinimalHPolyhedron struct representing the polyhedron.
   */
  const MinimalHPolyhedron GetMyMinimalHPolyhedron() const;

  /**
   * @brief Checks if a point is inside the polyhedron.
   * @param previous_sample The point to check.
   * @return True if the point is inside the polyhedron, false otherwise.
   */
  const bool PointInSet(const Eigen::VectorXf& previous_sample) const;

  /**
   * @brief Generates a uniform sample from the polyhedron using hit-and-run
   * sampling.
   * @param previous_sample The starting point for the sampling process, must be
   * contained.
   * @param mixing_steps The number of steps in the hit-and-run algorithm,
   * ~30-50 steps is a good choice.
   * @return A uniformly distributed point within the polyhedron.
   */
  const Eigen::VectorXf UniformSample(const Eigen::VectorXf& previous_sample,
                                      const int mixing_steps) const;

  /**
   * @brief Finds a feasible point within the polyhedron.
   * @return A point that satisfies all the constraints of the polyhedron.
   */
  const Eigen::VectorXf GetFeasiblePoint() const;

  /**
   * @brief Calculates the Chebyshev center of the polyhedron.
   * @return The coordinates of the Chebyshev center.
   */
  const Eigen::VectorXf ChebyshevCenter() const;

  /**
   * @brief Gets the ambient dimension of the polyhedron.
   * @return The dimension of the space in which the polyhedron is embedded.
   */
  const int ambient_dimension() const;

 private:
  Eigen::MatrixXf _A;      ///< Matrix of face normals
  Eigen::VectorXf _b;      ///< Right-hand side vector
  int _ambient_dimension;  ///< Dimension of the ambient space
};
}  // namespace csdecomp