#pragma once

#include <Eigen/Dense>

namespace csdecomp {

/**
 * @brief Checks for collision between two spheres.
 *
 * @param radiusA Radius of the first sphere
 * @param posA Position of the first sphere's center
 * @param radiusB Radius of the second sphere
 * @param posB Position of the second sphere's center
 * @return bool True if the spheres are not colliding, false otherwise
 */
__device__ bool sphereSphere(const float radiusA, const Eigen::Vector3f &posA,
                             const float radiusB, const Eigen::Vector3f &posB) {
  float distanceSquared = (posA - posB).squaredNorm();
  return distanceSquared >= (radiusA + radiusB) * (radiusA + radiusB);
}

/**
 * @brief Checks for collision between a sphere and a box.
 *
 * @param sphereRadius Radius of the sphere
 * @param spherePos Position of the sphere's center
 * @param boxDimensions Dimensions of the box (width, height, depth)
 * @param X_W_BOX Transformation matrix of the box in world frame
 * @return bool True if the sphere and box are not colliding, false otherwise
 */
__device__ bool sphereBox(const float sphereRadius,
                          const Eigen::Vector3f &spherePos,
                          const Eigen::Vector3f &boxDimensions,
                          const Eigen::Matrix4f &X_W_BOX) {
  // Extract positions in world frame
  const auto boxPos = X_W_BOX.block<3, 1>(0, 3);
  const auto boxRot = X_W_BOX.block<3, 3>(0, 0);

  Eigen::Vector3f boxHalfExtents = boxDimensions * 0.5f;

  // Transform sphere center to box local space
  Eigen::Vector3f sphereLocalPos = boxRot.transpose() * (spherePos - boxPos);

  // Find the closest point on the box to the sphere center
  Eigen::Vector3f closestPoint;
  for (int i = 0; i < 3; ++i) {
    closestPoint[i] = std::max(-boxHalfExtents[i],
                               std::min(sphereLocalPos[i], boxHalfExtents[i]));
  }

  // Check if the closest point is within the sphere's radius
  float distanceSquared = (closestPoint - sphereLocalPos).squaredNorm();

  return distanceSquared >= (sphereRadius * sphereRadius);
}

/**
 * @brief Checks for collision between two boxes using the Separating Axis
 * Theorem.
 *
 * @param boxADimensions Dimensions of the first box in (x,y,z)
 * @param X_W_GEOMA Transformation matrix of the first box in world frame
 * @param boxBDimensions Dimensions of the second box in (x,y,z)
 * @param X_W_GEOMB Transformation matrix of the second box in world frame
 * @return bool True if the boxes are not colliding, false otherwise
 */
__device__ bool boxBox(const Eigen::Vector3f &boxADimensions,
                       const Eigen::Matrix4f &X_W_GEOMA,
                       const Eigen::Vector3f &boxBDimensions,
                       const Eigen::Matrix4f &X_W_GEOMB) {
  // Extract positions in world frame
  const auto posA = X_W_GEOMA.block<3, 1>(0, 3);
  const auto posB = X_W_GEOMB.block<3, 1>(0, 3);

  // Extract rotation matrices
  const auto rotA = X_W_GEOMA.block<3, 3>(0, 0);
  const auto rotB = X_W_GEOMB.block<3, 3>(0, 0);

  Eigen::Vector3f geomA_ax1 = rotA.col(0);
  Eigen::Vector3f geomA_ax2 = rotA.col(1);
  Eigen::Vector3f geomA_ax3 = rotA.col(2);

  Eigen::Vector3f geomB_ax1 = rotB.col(0);
  Eigen::Vector3f geomB_ax2 = rotB.col(1);
  Eigen::Vector3f geomB_ax3 = rotB.col(2);

  float sep_axes_data[45];
  Eigen::Map<Eigen::Matrix3Xf> sep_axes(sep_axes_data, 3, 15);

  sep_axes.col(0) = geomA_ax1;
  sep_axes.col(1) = geomA_ax2;
  sep_axes.col(2) = geomA_ax3;

  sep_axes.col(3) = geomB_ax1;
  sep_axes.col(4) = geomB_ax2;
  sep_axes.col(5) = geomB_ax3;

  sep_axes.col(6) = geomA_ax1.cross(geomB_ax1);
  sep_axes.col(7) = geomA_ax1.cross(geomB_ax2);
  sep_axes.col(8) = geomA_ax1.cross(geomB_ax3);

  sep_axes.col(9) = geomA_ax2.cross(geomB_ax1);
  sep_axes.col(10) = geomA_ax2.cross(geomB_ax2);
  sep_axes.col(11) = geomA_ax2.cross(geomB_ax3);

  sep_axes.col(12) = geomA_ax3.cross(geomB_ax1);
  sep_axes.col(13) = geomA_ax3.cross(geomB_ax2);
  sep_axes.col(14) = geomA_ax3.cross(geomB_ax3);
  const Eigen::Vector3f p_A_B = posA - posB;
  for (size_t i = 0; i < 15; ++i) {
    const double lhs = std::abs(p_A_B.dot(sep_axes.col(i)));
    const double rhs =
        std::abs((boxADimensions[0] / 2) * geomA_ax1.dot(sep_axes.col(i))) +
        std::abs((boxADimensions[1] / 2) * geomA_ax2.dot(sep_axes.col(i))) +
        std::abs((boxADimensions[2] / 2) * geomA_ax3.dot(sep_axes.col(i))) +
        std::abs((boxBDimensions[0] / 2) * geomB_ax1.dot(sep_axes.col(i))) +
        std::abs((boxBDimensions[1] / 2) * geomB_ax2.dot(sep_axes.col(i))) +
        std::abs((boxBDimensions[2] / 2) * geomB_ax3.dot(sep_axes.col(i)));
    if (lhs > rhs) {
      // There exists a separating plane, so the pair is collision-free.
      return true;
    }
  }
  return false;
}
}  // namespace csdecomp