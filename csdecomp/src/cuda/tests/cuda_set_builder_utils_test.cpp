
#include "cuda_set_builder_utils.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <matplotlibcpp.h>

#include <chrono>
#include <iostream>
#include <random>

#include "hpolyhedron.h"
#include "plotting_utils.h"

namespace plt = matplotlibcpp;
using namespace PlottingUtils;
using namespace csdecomp;

GTEST_TEST(FCIUtilTest, ProjectionPlottingCuda) {
  int num_samples = 1000;
  Eigen::MatrixXf start_points(2, 1);
  Eigen::MatrixXf end_points(2, 1);

  // clang-format off
    start_points<< 0,// 1, 
                   1;// 1;
    end_points<<   2,// 2, 
                   1.8;// 1.8;
  // clang-format on

  std::vector<Eigen::MatrixXf> samples;
  // std::random_device rd;
  std::mt19937 gen(137);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Create a custom random number generator function
  auto random_generator = [&]() { return dis(gen); };
  Eigen::Vector2f min = {-3, -1}, max = {4, 5};
  Eigen::Vector2f diff = max - min;
  for (int i = 0; i < start_points.cols(); ++i) {
    Eigen::MatrixXf p_rand =
        Eigen::MatrixXf::NullaryExpr(2, num_samples, random_generator);
    p_rand = p_rand.array().colwise() * diff.array();
    p_rand.colwise() += min;
    samples.push_back(p_rand);
  }

  samples.at(0)(0, 0) = 0;
  samples.at(0)(1, 0) = 0;
  samples.at(0)(0, 1) = 3.5;
  samples.at(0)(1, 1) = 0;
  samples.at(0)(0, 2) = 1.5;
  samples.at(0)(1, 2) = 0;
  Eigen::Vector2f mid_proj{0.948276, 1.37931};
  const std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>>
      result =
          projectSamplesOntoLineSegmentsCuda(start_points, end_points, samples);

  EXPECT_TRUE(result.second.at(0).col(0).isApprox(start_points.col(0)));
  EXPECT_TRUE(result.second.at(0).col(1).isApprox(end_points.col(0)));
  EXPECT_LE((result.second.at(0).col(2) - mid_proj).norm(), 0.000001);
  EXPECT_EQ(result.first.at(0)[0],
            (samples.at(0).col(0) - start_points.col(0)).norm());
  EXPECT_EQ(result.first.at(0)[1],
            (samples.at(0).col(1) - end_points.col(0)).norm());
  EXPECT_NEAR((result.second.at(0).col(2) - samples.at(0).col(2)).norm() -
                  result.first.at(0)[2],
              0, 0.00001);

  if (std::getenv("BAZEL_TEST") == nullptr) {
    setupTestPlottingEnv();
    plt::figure_size(1000, 1000);
    std::map<std::string, std::string> keywords;

    // plt::plot({start_points(0, 1), end_points(0, 1)},
    //           {start_points(1, 1), end_points(1, 1)}, keywords);

    std::vector<std::string> colors = {"b", "g"};
    std::map<std::string, std::string> keywords_scatter;
    int idx = 0;
    for (auto s : samples) {
      keywords_scatter["color"] = colors.at(idx);

      Eigen::VectorXd row = s.row(0).cast<double>();
      std::vector<double> samp_x(row.data(), row.data() + num_samples);
      row = s.row(1).cast<double>();
      std::vector<double> samp_y(row.data(), row.data() + num_samples);
      plt::scatter(samp_x, samp_y, 10, keywords_scatter);

      Eigen::MatrixXd proj = result.second.at(idx).cast<double>();
      row = proj.row(0);
      std::vector<double> proj_x(row.data(), row.data() + num_samples);
      row = proj.row(1);
      std::vector<double> proj_y(row.data(), row.data() + num_samples);

      keywords["linewidth"] = "1";
      // keywords["alpha"] = "0.2";
      keywords["color"] = "k";

      for (int pair = 0; pair < s.cols(); ++pair) {
        plt::plot({s(0, pair), proj(0, pair)}, {s(1, pair), proj(1, pair)},
                  keywords);
      }
      keywords_scatter["color"] = "y";
      plt::scatter(proj_x, proj_y, 100, keywords_scatter);
      ++idx;
    }
    keywords["linewidth"] = "3";
    keywords["color"] = 'r';
    keywords["linestyle"] = "-";
    plt::plot({start_points(0, 0), end_points(0, 0)},
              {start_points(1, 0), end_points(1, 0)}, keywords);
    cleanupPlotAndPauseForUser();
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}