#pragma once
#include <matplotlibcpp.h>

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>

#include "hpolyhedron.h"

using namespace csdecomp;

namespace PlottingUtils {
void inline setupTestPlottingEnv() { matplotlibcpp::backend("GTKAgg"); }
void inline cleanupPlotAndPauseForUser(float pause_time = 0.5) {
  matplotlibcpp::axis("equal");

  matplotlibcpp::draw();
  matplotlibcpp::pause(pause_time);
  matplotlibcpp::show(false);
  std::cout << "press ENTER to continue";
  matplotlibcpp::pause(0.1);
  std::getchar();
  matplotlibcpp::close();
}

std::pair<std::vector<double>, std::vector<double>>
get2DHPolyhedronLineVertices(const HPolyhedron& hpoly,
                             const int resolution = 200) {
  assert(hpoly.ambient_dimension() == 2);
  std::vector<double> x, y;
  Eigen::VectorXf center = hpoly.ChebyshevCenter();

  for (int i = 0; i < resolution + 1; ++i) {
    float angle = i * 2 * M_PI / (float)resolution;
    Eigen::Vector2f dir;
    dir << cos(angle), sin(angle);
    Eigen::VectorXf line_b = hpoly.b() - hpoly.A() * center;
    Eigen::VectorXf line_A = hpoly.A() * dir;

    float theta_max = 1000.0;
    for (int j = 0; j < line_A.size(); ++j) {
      if (line_A[j] > 0.0) {
        theta_max = std::min(theta_max, line_b[j] / line_A[j]);
      }
    }
    Eigen::Vector2f boundary_point;
    boundary_point = (center + theta_max * dir).head(2);
    x.push_back((double)boundary_point.x());
    y.push_back((double)boundary_point.y());
  }
  std::pair<std::vector<double>, std::vector<double>> res{x, y};
  return res;
}

void inline plot2DHPolyhedron(const HPolyhedron& hpoly,
                              const std::string& color = "b",
                              const int resolution = 100) {
  auto verts = get2DHPolyhedronLineVertices(hpoly, resolution);
  std::map<std::string, std::string> keywords;
  keywords["linewidth"] = "3";
  keywords["color"] = color;
  keywords["linestyle"] = "-";
  matplotlibcpp::plot(verts.first, verts.second, keywords);
}

bool inline plot(const Eigen::VectorXf& x, const Eigen::VectorXf& y,
                 const std::map<std::string, std::string>& keywords) {
  Eigen::VectorXd x_copy = x.cast<double>();
  Eigen::VectorXd y_copy = y.cast<double>();
  std::vector<double> samp_x(x_copy.data(), x_copy.data() + x_copy.size());
  std::vector<double> samp_y(y_copy.data(), y_copy.data() + y_copy.size());
  return matplotlibcpp::plot(samp_x, samp_y, keywords);
}

bool inline draw_circle(const float center_x, const float center_y,
                        const float radius, const std::string& color = "b",
                        const int resolution = 100,
                        const std::string& linewidth = "3") {
  Eigen::MatrixXf points(2, resolution + 1);
  for (int idx = 0; idx < resolution + 1; ++idx) {
    points(0, idx) =
        center_x + radius * cos(idx / (1.0 * resolution) * 2 * M_PI);
    points(1, idx) =
        center_y + radius * sin(idx / (1.0 * resolution) * 2 * M_PI);
  }
  std::map<std::string, std::string> keywords;
  keywords["linewidth"] = linewidth;
  keywords["color"] = color;
  keywords["linestyle"] = "-";
  return plot(points.row(0), points.row(1), keywords);
}

bool inline scatter(const Eigen::VectorXf& x, const Eigen::VectorXf& y, float s,
                    const std::map<std::string, std::string>& keywords) {
  Eigen::VectorXd x_copy = x.cast<double>();
  Eigen::VectorXd y_copy = y.cast<double>();
  std::vector<double> samp_x(x_copy.data(), x_copy.data() + x_copy.size());
  std::vector<double> samp_y(y_copy.data(), y_copy.data() + y_copy.size());
  return matplotlibcpp::scatter(samp_x, samp_y, s, keywords);
}

}  // namespace PlottingUtils