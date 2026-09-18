#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "map_closures/AlignRansac2D.hpp"

namespace {
int failures = 0;

void Check(const bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void TestMinimalSetProducesProperRotation() {
    const Eigen::Rotation2Dd rotation(M_PI_2);
    const Eigen::Vector2d translation(3.0, -4.0);
    const std::vector<map_closures::PointPair> pairs{
        {{0.0, 0.0}, translation},
        {{2.0, 0.0}, rotation * Eigen::Vector2d(2.0, 0.0) + translation},
    };
    const auto [transform, inliers] = map_closures::RansacAlignment2D(pairs);
    Check(inliers == pairs.size(), "minimal rigid set must retain both inliers");
    Check(std::abs(transform.linear().determinant() - 1.0) < 1e-12,
          "estimated rotation must be proper SO(2), never a reflection");
    for (const auto &pair : pairs) {
        Check((transform * pair.ref - pair.query).norm() < 1e-12,
              "minimal-set residual must be numerically zero");
    }
}

void TestDeterministicWithOutliers() {
    const Eigen::Rotation2Dd rotation(0.37);
    const Eigen::Vector2d translation(-2.0, 5.0);
    std::vector<map_closures::PointPair> pairs;
    for (const Eigen::Vector2d &point :
         {Eigen::Vector2d{-2.0, -1.0}, Eigen::Vector2d{0.0, 0.0},
          Eigen::Vector2d{1.0, 3.0}, Eigen::Vector2d{4.0, -2.0},
          Eigen::Vector2d{5.0, 2.0}}) {
        pairs.emplace_back(point, rotation * point + translation);
    }
    pairs.emplace_back(Eigen::Vector2d{30.0, 40.0}, Eigen::Vector2d{-50.0, 70.0});
    pairs.emplace_back(Eigen::Vector2d{-45.0, 25.0}, Eigen::Vector2d{60.0, -80.0});

    const auto first = map_closures::RansacAlignment2D(pairs);
    const auto second = map_closures::RansacAlignment2D(pairs);
    Check(first.second == 5U, "RANSAC must identify the five geometric inliers");
    Check(first.second == second.second, "inlier count must be reproducible");
    Check(first.first.matrix() == second.first.matrix(),
          "fixed-seed RANSAC transform must be bitwise reproducible");
    Check(std::abs(first.first.linear().determinant() - 1.0) < 1e-12,
          "refined transform must remain in SO(2)");
}

void TestInsufficientInputIsExplicit() {
    const std::vector<map_closures::PointPair> one_pair{
        {{0.0, 0.0}, {1.0, 1.0}},
    };
    const auto [transform, inliers] = map_closures::RansacAlignment2D(one_pair);
    Check(inliers == 0U, "insufficient input must report zero inliers");
    Check(transform.matrix().isApprox(Eigen::Matrix3d::Identity()),
          "insufficient input must return identity");
}
}  // namespace

int main() {
    TestMinimalSetProducesProperRotation();
    TestDeterministicWithOutliers();
    TestInsufficientInputIsExplicit();
    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "All deterministic SO(2) RANSAC tests passed\n";
    return 0;
}
