#pragma once
#include <Eigen/Dense>
#include <limits>

#include "collision_geometry.h"
namespace csdecomp {

namespace {
constexpr float line_seg_cutoff = 1e-6;

Eigen::Vector3f closestPointOnLineSegment(const Eigen::Vector3f& line_start,
                                          const Eigen::Vector3f& line_end,
                                          const Eigen::Vector3f& point) {
  const Eigen::Vector3f dir = line_end - line_start;
  const double squared_norm = dir.squaredNorm();
  if (squared_norm < line_seg_cutoff) {
    return line_start;
  }
  const double t = (point - line_start).dot(dir) / squared_norm;
  return line_start + std::clamp<float>(t, 0, 1) * dir;
}

float distanceLineSegmentBox(const Eigen::Vector3f& p1,
                             const Eigen::Vector3f& q1,
                             const Eigen::Vector3f& box_dimensions) {}

float distanceBetweenLineSegments(const Eigen::Vector3f& p1,
                                  const Eigen::Vector3f& q1,
                                  const Eigen::Vector3f& p2,
                                  const Eigen::Vector3f& q2) {
  const Eigen::Vector3f dir_1 = q1 - p1;
  const Eigen::Vector3f dir_2 = q2 - p2;
  // Get the segments' parameters
  const Eigen::Vector3f r = p1 - p2;

  const float a = dir_1.dot(dir_1);  // Length squared of segment 1
  const float b = dir_1.dot(dir_2);  // Dot product of the two directions
  const float c = dir_2.dot(dir_2);  // Length squared of segment 2
  const float d = dir_1.dot(r);
  const float e = dir_2.dot(r);

  const float f = a * c - b * b;  // Denominator

  // Parameters to compute the closest points
  float s = 0.0f;  // Parameter for first segment
  float t = 0.0f;  // Parameter for second segment

  // Compute the closest points parameters
  if (f < line_seg_cutoff) {
    // The lines are almost parallel
    // Force using point p1, and find the closest point on segment 2
    s = 0.0f;
    t = (b > c ? d / b : e / c);  // Use the largest denominator
  } else {
    // Get the closest points on the infinite lines
    s = (b * e - c * d) / f;
    t = (a * e - b * d) / f;
  }

  // Clamp the parameters to the segments
  s = std::clamp<float>(s, 0, 1);

  // Recompute t based on clamped s
  t = b * s + e;
  if (c > 0.0f) {
    t /= c;
  } else {
    t = 0.0f;
  }

  // Clamp t to the segment
  t = std::clamp<float>(t, 0, 1);

  // Recompute s based on clamped t
  if (a > 0.0f) {
    s = (b * t - d) / a;
    s = std::max(0.0f, std::min(1.0f, s));
  }

  // Compute the closest points on both segments
  const Eigen::Vector3f closest_point_1 = p1 + dir_1 * s;
  const Eigen::Vector3f closest_point_2 = p2 + dir_2 * t;

  // Return the distance between these points
  return (closest_point_1 - closest_point_2).norm();
}
}  // namespace

// TODO capsule box
bool cppCapsuleBox(const Eigen::Vector3f& capDimensions,
                   const Eigen::Matrix4f& X_W_CAP,
                   const Eigen::Vector3f& boxDimensions,
                   const Eigen::Matrix4f& X_W_BOX) {
  // THIS IS NOT EXACT, it is conservative

  // Express the linesegment defining the capsule in the frame of the box
  // 1. Compute X_BOX_CAP = (X_w_box)^-1X_w_Cap
  Eigen::Matrix3f rot_cap_inv = X_W_CAP.block<3, 3>(0, 0).transpose();
  Eigen::Vector3f pos_cap = X_W_CAP.block<3, 1>(0, 3);
  Eigen::Matrix3f rot_box = X_W_BOX.block<3, 3>(0, 0);
  Eigen::Vector3f pos_box = X_W_BOX.block<3, 1>(0, 3);
  Eigen::Matrix3f rot_box_cap = rot_cap_inv * rot_box;
  Eigen::Matrix3f pos_box_cap = rot_cap_inv * (pos_box - pos_cap);
  Eigen::Vector3f cyl_base =
      rot_box_cap * Eigen::Vector3f(0, 0, -capDimensions[1] / 2) + pos_box_cap;
  Eigen::Vector3f cyl_tip =
      rot_box_cap * Eigen::Vector3f(0, 0, capDimensions[1] / 2) + pos_box_cap;
  Eigen::Vector3f boxHalfExtents = boxDimensions * 0.5f;
  Eigen::Vector3f min = -boxHalfExtents;
  Eigen::Vector3f max = boxHalfExtents;
  float near = std::numeric_limits<float>::min();
  float far = std::numeric_limits<float>::max();

https://stackoverflow.com/questions/3106666/intersection-of-line-segment-with-axis-aligned-box-in-c-sharp#3115514
  /*
            +Z
            ^
       ---- |------
     /      |     /|
    /----------- / |
   |        +---|----> +Y
   | _ ___ /___ |/
          /
         /
        v
      +X

      3D XYZ Coordinate System
*/
  // Linesegment goes from p to q
  Eigen::Vector3f pProj;
  Eigen::Vector3f qProj;
  pProj.z() = boxHalfExtents[2];
  qProj.z() = boxHalfExtents[2];
  // crop XY planes
  // top plane
}

bool cppCapsuleCapsule(const Eigen::Vector3f& capADimensions,
                       const Eigen::Matrix4f& X_W_GEOMA,
                       const Eigen::Vector3f& capBDimensions,
                       const Eigen::Matrix4f& X_W_GEOMB) {
  Eigen::Matrix3f rot_A = X_W_GEOMA.block<3, 3>(0, 0);
  Eigen::Vector3f pos_A = X_W_GEOMA.block<3, 1>(0, 3);
  Eigen::Matrix3f rot_B = X_W_GEOMB.block<3, 3>(0, 0);
  Eigen::Vector3f pos_B = X_W_GEOMB.block<3, 1>(0, 3);

  const Eigen::Vector3f cyl_base_A =
      rot_A * Eigen::Vector3f(0, 0, -capADimensions[1] / 2) + pos_A;
  const Eigen::Vector3f cyl_tip_A =
      rot_A * Eigen::Vector3f(0, 0, capADimensions[1] / 2) + pos_A;
  const Eigen::Vector3f cyl_base_B =
      rot_B * Eigen::Vector3f(0, 0, -capADimensions[1] / 2) + pos_B;
  const Eigen::Vector3f cyl_tip_B =
      rot_B * Eigen::Vector3f(0, 0, capADimensions[1] / 2) + pos_B;
  float dist_ls =
      distanceBetweenLineSegments(cyl_base_A, cyl_tip_A, cyl_base_B, cyl_tip_B);
  return dist_ls > capADimensions[0] + capBDimensions[0];
}

bool cppCapsuleSphere(const Eigen::Matrix4f& X_W_cap, const float capRadius,
                      const float capLength, const Eigen::Vector3f& spherePos,
                      const float sphereRadius) {
  Eigen::Matrix3f rot = X_W_cap.block<3, 3>(0, 0);
  Eigen::Vector3f pos = X_W_cap.block<3, 1>(0, 3);
  const Eigen::Vector3f cyl_base =
      rot * Eigen::Vector3f(0, 0, -capLength / 2) + pos;
  const Eigen::Vector3f cyl_tip =
      rot * Eigen::Vector3f(0, 0, capLength / 2) + pos;

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
