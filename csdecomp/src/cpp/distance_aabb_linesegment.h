#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace csdecomp {
std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> DistanceLinesegmentAABB(
    const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
    const Eigen::VectorXd& box_min, const Eigen::VectorXd& box_max,
    const int maxit = 100, const double tol = 1e-9);

}  // namespace csdecomp
