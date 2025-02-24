#pragma once
#include <Eigen/Dense>

#include "collision_geometry.h"
namespace csdecomp {

namespace {
constexpr float line_seg_cutoff = 1e-6;
}

Eigen::Vector3f closestPointOnLineSegment(const Eigen::Vector3f& line_start,
                                          const Eigen::Vector3f& line_end,
                                          const Eigen::Vector3f& point) {
  const Eigen::Vector3f dir = line_end - line_start;
  const double squared_norm = dir.squaredNorm();
  if (squared_norm < line_seg_cutoff) {
    return line_start;
  }
  const double t = (point - line_start).dot(dir) / squared_norm;
  return line_start + std::clamp<double>(t, 0, 1) * dir;
}

// TODO capsule sphere, capsule capsule, capsule box
bool cppCapsuleSphere(const Eigen::Matrix4f& X_W_cap, const float capRadius,
                      const float capLength, const Eigen::Vector3f& spherePos,
                      const float sphereRadius) {
  Eigen::Matrix3f rot = X_W_cap.block<3, 3>(0, 0);
  const Eigen::Vector3f cyl_base = rot * Eigen::Vector3f(0, 0, -capLength / 2);
  const Eigen::Vector3f cyl_tip = rot * Eigen::Vector3f(0, 0, capLength / 2);

  // closest point to sphere center along capsule line segment
  const Eigen::Vector3f capsule_point =
      closestPointOnLineSegment(cyl_base, cyl_tip, spherePos);
  float dist = (capsule_point - spherePos).norm();
  return dist > capRadius + sphereRadius;
}

bool cppSphereBox(const float sphereRadius, const Eigen::Vector3f& spherePos,
                  const Eigen::Vector3f& boxDimensions,
                  const Eigen::Matrix4f& X_W_BOX) {
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

bool cppBoxBox(const Eigen::Vector3f& boxADimensions,
               const Eigen::Matrix4f& X_W_GEOMA,
               const Eigen::Vector3f& boxBDimensions,
               const Eigen::Matrix4f& X_W_GEOMB) {
  const auto posA = X_W_GEOMA.block<3, 1>(0, 3);
  const auto posB = X_W_GEOMB.block<3, 1>(0, 3);
  float maxlen =
      std::max((boxADimensions).maxCoeff(), (boxBDimensions).maxCoeff());
  if ((posA - posB).norm() > 2 * sqrt(3) * maxlen) {
    return true;
  }
  Eigen::Matrix3f rotA = X_W_GEOMA.block<3, 3>(0, 0);
  Eigen::Matrix3f rotB = X_W_GEOMB.block<3, 3>(0, 0);
  Eigen::Vector3f geomA_ax1 = rotA.col(0);
  Eigen::Vector3f geomA_ax2 = rotA.col(1);
  Eigen::Vector3f geomA_ax3 = rotA.col(2);

  Eigen::Vector3f geomB_ax1 = rotB.col(0);
  Eigen::Vector3f geomB_ax2 = rotB.col(1);
  Eigen::Vector3f geomB_ax3 = rotB.col(2);
  std::vector<Eigen::Vector3f> sep_axes(15);
  sep_axes[0] = geomA_ax1;
  sep_axes[1] = geomA_ax2;
  sep_axes[2] = geomA_ax3;

  sep_axes[3] = geomB_ax1;
  sep_axes[4] = geomB_ax2;
  sep_axes[5] = geomB_ax3;

  sep_axes[6] = geomA_ax1.cross(geomB_ax1);
  sep_axes[7] = geomA_ax1.cross(geomB_ax2);
  sep_axes[8] = geomA_ax1.cross(geomB_ax3);

  sep_axes[9] = geomA_ax2.cross(geomB_ax1);
  sep_axes[10] = geomA_ax2.cross(geomB_ax2);
  sep_axes[11] = geomA_ax2.cross(geomB_ax3);

  sep_axes[12] = geomA_ax3.cross(geomB_ax1);
  sep_axes[13] = geomA_ax3.cross(geomB_ax2);
  sep_axes[14] = geomA_ax3.cross(geomB_ax3);

  const Eigen::Vector3f p_A_B = posA - posB;

  for (size_t i = 0; i < sep_axes.size(); ++i) {
    const double lhs = std::abs(p_A_B.dot(sep_axes[i]));
    const double rhs =
        std::abs((boxADimensions[0] / 2) * geomA_ax1.dot(sep_axes[i])) +
        std::abs((boxADimensions[1] / 2) * geomA_ax2.dot(sep_axes[i])) +
        std::abs((boxADimensions[2] / 2) * geomA_ax3.dot(sep_axes[i])) +
        std::abs((boxBDimensions[0] / 2) * geomB_ax1.dot(sep_axes[i])) +
        std::abs((boxBDimensions[1] / 2) * geomB_ax2.dot(sep_axes[i])) +
        std::abs((boxBDimensions[2] / 2) * geomB_ax3.dot(sep_axes[i]));
    if (lhs > rhs) {
      // There exists a separating plane, so the pair is collision-free.
      return true;
    }
  }
  return false;
}

}  // namespace csdecomp
