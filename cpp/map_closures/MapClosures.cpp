// MIT License
//
// Copyright (c) 2026 Saurabh Gupta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "MapClosures.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/core.hpp>
#include <set>
#include <opencv2/features2d.hpp>
#include <utility>
#include <vector>

#include "AlignRansac2D.hpp"
#include "DensityMap.hpp"
#include "GroundAlign.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

namespace {
constexpr int min_no_of_matches = 2;
constexpr int no_of_local_maps_to_skip = 3;
constexpr int self_similarity_threshold = 35;

// fixed parameters for OpenCV ORB Features
constexpr float scale_factor = 1.00f;
constexpr int n_levels = 1;
constexpr int first_level = 0;
constexpr int WTA_K = 2;
constexpr int nfeatures = 500;
constexpr int edge_threshold = 31;
constexpr int score_type = 0;
constexpr int patch_size = 31;
constexpr int fast_threshold = 35;
}  // namespace

namespace map_closures {
MapClosures::MapClosures() : config_(Config()) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
}

MapClosures::MapClosures(const Config &config) : config_(config) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
}

void MapClosures::MatchAndAddToDatabase(const int id,
                                        const std::vector<Eigen::Vector3d> &local_map) {
    Eigen::Matrix4d T_ground = AlignToLocalGround(local_map, config_.density_map_resolution);
    DensityMap density_map = GenerateDensityMap(local_map, T_ground, config_.density_map_resolution,
                                                config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_keypoints.reserve(nfeatures);
    orb_extractor_->detectAndCompute(density_map.grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);
    last_query_diagnostics_ = QueryDiagnostics{};
    last_reference_diagnostics_.clear();
    last_query_diagnostics_.query_id = id;
    last_query_diagnostics_.orb_descriptors =
        static_cast<std::size_t>(orb_descriptors.rows);

    std::vector<std::vector<cv::DMatch>> self_matches;
    self_matches.reserve(orb_keypoints.size());
    self_matcher_.knnMatch(orb_descriptors, orb_descriptors, self_matches, 2);

    std::vector<Matchable *> hbst_matchable;
    hbst_matchable.reserve(orb_descriptors.rows);
    std::for_each(
        self_matches.cbegin(), self_matches.cend(), [&](const std::vector<cv::DMatch> &self_match) {
            if (self_match.size() >= 2 && self_match[1].distance > self_similarity_threshold) {
                const int index_descriptor = self_match[0].queryIdx;
                cv::KeyPoint keypoint = orb_keypoints[index_descriptor];
                keypoint.pt.x = keypoint.pt.x + static_cast<float>(density_map.lower_bound.y());
                keypoint.pt.y = keypoint.pt.y + static_cast<float>(density_map.lower_bound.x());
                hbst_matchable.emplace_back(
                    new Matchable(keypoint, orb_descriptors.row(index_descriptor), id));
            }
        });

    last_query_diagnostics_.retained_descriptors = hbst_matchable.size();
    descriptor_matches_.clear();
    hbst_binary_tree_->matchAndAdd(hbst_matchable, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);

    density_maps_.emplace(id, std::move(density_map));
    ground_alignments_.emplace(id, std::move(T_ground));
}

void MapClosures::Match(const std::vector<Eigen::Vector3d> &local_map) {
    const Eigen::Matrix4d T_ground = AlignToLocalGround(local_map, config_.density_map_resolution);
    DensityMap density_map = GenerateDensityMap(local_map, T_ground, config_.density_map_resolution,
                                                config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_keypoints.reserve(nfeatures);
    orb_extractor_->detectAndCompute(density_map.grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);

    std::vector<std::vector<cv::DMatch>> self_matches;
    self_matches.reserve(orb_keypoints.size());
    self_matcher_.knnMatch(orb_descriptors, orb_descriptors, self_matches, 2);

    std::vector<Matchable *> hbst_matchable;
    hbst_matchable.reserve(orb_descriptors.rows);
    std::for_each(
        self_matches.cbegin(), self_matches.cend(), [&](const std::vector<cv::DMatch> &self_match) {
            if (self_match.size() >= 2 && self_match[1].distance > self_similarity_threshold) {
                const int index_descriptor = self_match[0].queryIdx;
                cv::KeyPoint keypoint = orb_keypoints[index_descriptor];
                keypoint.pt.x = keypoint.pt.x + static_cast<float>(density_map.lower_bound.y());
                keypoint.pt.y = keypoint.pt.y + static_cast<float>(density_map.lower_bound.x());
                hbst_matchable.emplace_back(
                    new Matchable(keypoint, orb_descriptors.row(index_descriptor)));
            }
        });
    descriptor_matches_.clear();
    hbst_binary_tree_->match(hbst_matchable, descriptor_matches_,
                             config_.hamming_distance_threshold);
}

ClosureCandidate MapClosures::ValidateClosure(const int reference_id, const int query_id) const {
    return ValidateClosure(reference_id, query_id, descriptor_matches_);
}

ClosureCandidate MapClosures::ValidateClosure(
    const int reference_id,
    const int query_id,
    const Tree::MatchVectorMap &matches_by_reference) const {
    const auto it = matches_by_reference.find(reference_id);
    if (it == matches_by_reference.end()) {
        return ClosureCandidate();
    }

    const Tree::MatchVector &matches = it->second;
    const size_t num_matches = matches.size();

    ClosureCandidate closure;
    if (num_matches > min_no_of_matches) {
        std::vector<PointPair> keypoint_pairs(num_matches);
        std::transform(matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
                       [&](const Tree::Match &match) {
                           const Eigen::Vector2d query_point =
                               Eigen::Vector2d(match.object_query.pt.y, match.object_query.pt.x);
                           const Eigen::Vector2d ref_point = Eigen::Vector2d(
                               match.object_references[0].pt.y, match.object_references[0].pt.x);
                           return PointPair(ref_point, query_point);
                       });

        const auto [pose2d, number_of_inliers] = RansacAlignment2D(keypoint_pairs);
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.pose.block<2, 2>(0, 0) = pose2d.linear();
        closure.pose.block<2, 1>(0, 3) = pose2d.translation() * config_.density_map_resolution;
        closure.pose = ground_alignments_.at(query_id).inverse() * closure.pose *
                       ground_alignments_.at(reference_id);
        closure.number_of_inliers = number_of_inliers;
    }
    return closure;
}

std::vector<ClosureCandidate> MapClosures::GetTopKClosures(
    const int query_id, const std::vector<Eigen::Vector3d> &local_map, const int k) {
    MatchAndAddToDatabase(query_id, local_map);
    auto compare_closure_candidates = [](const ClosureCandidate &a, const ClosureCandidate &b) {
        return a.number_of_inliers > b.number_of_inliers;
    };

    last_query_diagnostics_.database_references = descriptor_matches_.size();
    last_query_diagnostics_.retrieved_references =
        static_cast<std::size_t>(std::count_if(
            descriptor_matches_.cbegin(), descriptor_matches_.cend(),
            [](const auto &entry) { return !entry.second.empty(); }));

    std::vector<ClosureCandidate> closures;
    const int num_of_potential_closures = query_id - no_of_local_maps_to_skip;
    if (num_of_potential_closures > 0) {
        last_query_diagnostics_.eligible_references =
            static_cast<std::size_t>(num_of_potential_closures);
        closures.reserve(num_of_potential_closures);
        for (int ref_id = 0; ref_id < num_of_potential_closures; ++ref_id) {
            const auto matches_it = descriptor_matches_.find(ref_id);
            const std::size_t number_of_matches =
                matches_it == descriptor_matches_.end() ? 0U : matches_it->second.size();
            if (number_of_matches > last_query_diagnostics_.maximum_matches) {
                last_query_diagnostics_.maximum_matches = number_of_matches;
                last_query_diagnostics_.best_match_reference_id = ref_id;
            }
            if (number_of_matches > min_no_of_matches) {
                ++last_query_diagnostics_.references_with_enough_matches;
            }

            ClosureCandidate closure = ValidateClosure(ref_id, query_id);
            ReferenceDiagnostics reference_diagnostics;
            reference_diagnostics.reference_id = ref_id;
            reference_diagnostics.query_id = query_id;
            reference_diagnostics.number_of_matches = number_of_matches;
            reference_diagnostics.number_of_inliers = closure.number_of_inliers;
            reference_diagnostics.has_pose =
                closure.source_id == ref_id && closure.target_id == query_id;
            if (reference_diagnostics.has_pose) {
                reference_diagnostics.pose = closure.pose;
            }
            last_reference_diagnostics_.emplace_back(std::move(reference_diagnostics));
            if (closure.number_of_inliers > last_query_diagnostics_.maximum_inliers) {
                last_query_diagnostics_.maximum_inliers = closure.number_of_inliers;
                last_query_diagnostics_.best_inlier_reference_id = ref_id;
            }
            if (closure.number_of_inliers > min_no_of_matches) {
                ++last_query_diagnostics_.references_with_enough_inliers;
                closures.emplace_back(std::move(closure));
            }
        }
        if (k != -1 && !closures.empty()) {
            const int top_k = std::min(k, static_cast<int>(closures.size()));
            if (top_k < static_cast<int>(closures.size())) {
                const auto kth = closures.begin() + top_k;
                std::nth_element(closures.begin(), kth, closures.end(), compare_closure_candidates);
                closures.resize(top_k);
            }
            std::sort(closures.begin(), closures.end(), compare_closure_candidates);
        }
    }
    return closures;
}


ExactRerankResult MapClosures::GetClosuresExactRerank(
    const int query_id,
    const std::vector<Eigen::Vector3d> &local_map,
    const int exact_rerank_top_k) {
    ExactRerankResult result;

    Eigen::Matrix4d T_ground =
        AlignToLocalGround(local_map, config_.density_map_resolution);
    DensityMap density_map =
        GenerateDensityMap(local_map, T_ground, config_.density_map_resolution,
                           config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_keypoints.reserve(nfeatures);
    orb_extractor_->detectAndCompute(
        density_map.grid, cv::noArray(), orb_keypoints, orb_descriptors);

    last_query_diagnostics_ = QueryDiagnostics{};
    last_reference_diagnostics_.clear();
    last_query_diagnostics_.query_id = query_id;
    last_query_diagnostics_.orb_descriptors =
        static_cast<std::size_t>(orb_descriptors.rows);

    std::vector<std::vector<cv::DMatch>> self_matches;
    self_matches.reserve(orb_keypoints.size());
    self_matcher_.knnMatch(
        orb_descriptors, orb_descriptors, self_matches, 2);

    std::vector<Matchable *> hbst_matchable;
    hbst_matchable.reserve(orb_descriptors.rows);
    std::for_each(
        self_matches.cbegin(), self_matches.cend(),
        [&](const std::vector<cv::DMatch> &self_match) {
            if (self_match.size() >= 2 &&
                self_match[1].distance > self_similarity_threshold) {
                const int index_descriptor = self_match[0].queryIdx;
                cv::KeyPoint keypoint = orb_keypoints[index_descriptor];
                keypoint.pt.x +=
                    static_cast<float>(density_map.lower_bound.y());
                keypoint.pt.y +=
                    static_cast<float>(density_map.lower_bound.x());
                hbst_matchable.emplace_back(
                    new Matchable(
                        keypoint, orb_descriptors.row(index_descriptor), query_id));
            }
        });

    last_query_diagnostics_.retained_descriptors = hbst_matchable.size();
    descriptor_matches_.clear();
    Tree::MatchVectorMap final_matches;
    std::vector<std::uint64_t> exact_reranked_ids;

    const int num_of_potential_closures =
        query_id - no_of_local_maps_to_skip;
    const std::uint64_t eligible_reference_exclusive =
        num_of_potential_closures > 0 ?
        static_cast<std::uint64_t>(num_of_potential_closures) : 0U;
    const std::size_t rerank_top_k =
        exact_rerank_top_k > 0 ?
        static_cast<std::size_t>(exact_rerank_top_k) : 0U;

    if (!hbst_matchable.empty()) {
        hbst_binary_tree_->matchAndAdd(
            hbst_matchable,
            descriptor_matches_,
            config_.hamming_distance_threshold,
            srrg_hbst::SplittingStrategy::SplitEven,
            rerank_top_k > 0 ? &final_matches : nullptr,
            rerank_top_k,
            eligible_reference_exclusive,
            &exact_reranked_ids);
    }
    if (rerank_top_k == 0) {
        final_matches = descriptor_matches_;
    }

    density_maps_.emplace(query_id, std::move(density_map));
    ground_alignments_.emplace(query_id, std::move(T_ground));

    last_query_diagnostics_.database_references = descriptor_matches_.size();
    last_query_diagnostics_.retrieved_references =
        static_cast<std::size_t>(std::count_if(
            descriptor_matches_.cbegin(), descriptor_matches_.cend(),
            [](const auto &entry) { return !entry.second.empty(); }));

    if (num_of_potential_closures > 0) {
        last_query_diagnostics_.eligible_references =
            static_cast<std::size_t>(num_of_potential_closures);
        last_reference_diagnostics_.reserve(
            static_cast<std::size_t>(num_of_potential_closures));

        for (int ref_id = 0; ref_id < num_of_potential_closures; ++ref_id) {
            const auto matches_it = descriptor_matches_.find(ref_id);
            const std::size_t number_of_matches =
                matches_it == descriptor_matches_.end() ?
                0U : matches_it->second.size();
            if (number_of_matches >
                last_query_diagnostics_.maximum_matches) {
                last_query_diagnostics_.maximum_matches = number_of_matches;
                last_query_diagnostics_.best_match_reference_id = ref_id;
            }
            if (number_of_matches > min_no_of_matches) {
                ++last_query_diagnostics_.references_with_enough_matches;
            }

            const ClosureCandidate closure =
                ValidateClosure(ref_id, query_id, descriptor_matches_);
            ReferenceDiagnostics reference_diagnostics;
            reference_diagnostics.reference_id = ref_id;
            reference_diagnostics.query_id = query_id;
            reference_diagnostics.number_of_matches = number_of_matches;
            reference_diagnostics.number_of_inliers =
                closure.number_of_inliers;
            reference_diagnostics.has_pose =
                closure.source_id == ref_id &&
                closure.target_id == query_id;
            if (reference_diagnostics.has_pose) {
                reference_diagnostics.pose = closure.pose;
            }
            last_reference_diagnostics_.emplace_back(
                std::move(reference_diagnostics));

            if (closure.number_of_inliers >
                last_query_diagnostics_.maximum_inliers) {
                last_query_diagnostics_.maximum_inliers =
                    closure.number_of_inliers;
                last_query_diagnostics_.best_inlier_reference_id = ref_id;
            }
            if (closure.number_of_inliers > min_no_of_matches) {
                ++last_query_diagnostics_.references_with_enough_inliers;
            }
        }
    }

    result.ann_query_diagnostics = last_query_diagnostics_;
    result.final_query_diagnostics = last_query_diagnostics_;
    result.final_query_diagnostics.retrieved_references =
        static_cast<std::size_t>(std::count_if(
            final_matches.cbegin(), final_matches.cend(),
            [](const auto &entry) { return !entry.second.empty(); }));
    result.final_query_diagnostics.references_with_enough_matches = 0U;
    result.final_query_diagnostics.references_with_enough_inliers = 0U;
    result.final_query_diagnostics.maximum_matches = 0U;
    result.final_query_diagnostics.best_match_reference_id = -1;
    result.final_query_diagnostics.maximum_inliers = 0U;
    result.final_query_diagnostics.best_inlier_reference_id = -1;
    result.exact_reranked_references = exact_reranked_ids.size();

    std::vector<std::pair<int, std::size_t>> ann_ranking;
    ann_ranking.reserve(last_reference_diagnostics_.size());
    for (const auto &reference : last_reference_diagnostics_) {
        if (reference.number_of_matches > 0U) {
            ann_ranking.emplace_back(
                reference.reference_id, reference.number_of_matches);
        }
    }
    std::sort(
        ann_ranking.begin(), ann_ranking.end(),
        [](const auto &lhs, const auto &rhs) {
            return lhs.second != rhs.second ?
                   lhs.second > rhs.second :
                   lhs.first < rhs.first;
        });
    std::set<int> exact_reranked_set;
    for (const auto id : exact_reranked_ids) {
        exact_reranked_set.insert(static_cast<int>(id));
    }

    result.references.reserve(last_reference_diagnostics_.size());
    result.closures.reserve(last_reference_diagnostics_.size());
    for (const auto &ann_reference : last_reference_diagnostics_) {
        const int ref_id = ann_reference.reference_id;
        const auto final_it = final_matches.find(ref_id);
        const std::size_t final_number_of_matches =
            final_it == final_matches.end() ?
            0U : final_it->second.size();
        const ClosureCandidate final_closure =
            ValidateClosure(ref_id, query_id, final_matches);

        if (final_number_of_matches >
            result.final_query_diagnostics.maximum_matches) {
            result.final_query_diagnostics.maximum_matches =
                final_number_of_matches;
            result.final_query_diagnostics.best_match_reference_id =
                ref_id;
        }
        if (final_number_of_matches > min_no_of_matches) {
            ++result.final_query_diagnostics.references_with_enough_matches;
        }
        if (final_closure.number_of_inliers >
            result.final_query_diagnostics.maximum_inliers) {
            result.final_query_diagnostics.maximum_inliers =
                final_closure.number_of_inliers;
            result.final_query_diagnostics.best_inlier_reference_id =
                ref_id;
        }
        if (final_closure.number_of_inliers > min_no_of_matches) {
            ++result.final_query_diagnostics.references_with_enough_inliers;
            result.closures.emplace_back(final_closure);
        }

        ExactRerankReferenceDiagnostics reference;
        reference.reference_id = ref_id;
        reference.query_id = query_id;
        reference.ann_number_of_matches =
            ann_reference.number_of_matches;
        reference.ann_number_of_inliers =
            ann_reference.number_of_inliers;
        reference.number_of_matches = final_number_of_matches;
        reference.number_of_inliers =
            final_closure.number_of_inliers;
        reference.exact_reranked =
            exact_reranked_set.count(ref_id) != 0U;
        reference.has_pose =
            final_closure.source_id == ref_id &&
            final_closure.target_id == query_id;
        if (reference.has_pose) {
            reference.pose = final_closure.pose;
        }
        const auto rank_it = std::find_if(
            ann_ranking.cbegin(), ann_ranking.cend(),
            [&](const auto &entry) { return entry.first == ref_id; });
        reference.ann_rank =
            rank_it == ann_ranking.cend() ?
            0U :
            static_cast<std::size_t>(
                std::distance(ann_ranking.cbegin(), rank_it) + 1);
        result.references.emplace_back(std::move(reference));
    }

    return result;
}

}  // namespace map_closures
