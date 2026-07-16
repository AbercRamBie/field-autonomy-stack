#include "autonomy/lane_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace autonomy {
namespace {

double smooth(
    const double current,
    const double previous,
    const double alpha
) {
    return alpha * current + (1.0 - alpha) * previous;
}

bool isFiniteGeometry(
    const LaneObservation& observation
) {
    return
        std::isfinite(observation.left_bottom_x) &&
        std::isfinite(observation.right_bottom_x) &&
        std::isfinite(observation.left_lookahead_x) &&
        std::isfinite(observation.right_lookahead_x) &&

        std::isfinite(observation.left_curve.a) &&
        std::isfinite(observation.left_curve.b) &&
        std::isfinite(observation.left_curve.c) &&

        std::isfinite(observation.right_curve.a) &&
        std::isfinite(observation.right_curve.b) &&
        std::isfinite(observation.right_curve.c);
}

}  // namespace

LaneTracker::LaneTracker(LaneTrackerConfig config)
    : config_(config) {
    if (config_.smoothing_alpha <= 0.0 ||
        config_.smoothing_alpha > 1.0 ||
        config_.maximum_boundary_jump_px <= 0.0 ||
        config_.maximum_lane_width_change_px <= 0.0 ||
        config_.warped_pixels_per_metre <= 0.0 ||
        config_.warped_bottom_y_px <= config_.warped_lookahead_y_px ||
        config_.maximum_missed_frames < 0 ||
        config_.confidence_decay <= 0.0 ||
        config_.confidence_decay > 1.0) {
        throw std::invalid_argument(
            "Invalid lane tracker configuration"
        );
    }
}

LaneObservation LaneTracker::update(
    const LaneObservation& detection
) {
    LaneObservation candidate = detection;

    if (candidate.left_lookahead_x >
        candidate.right_lookahead_x) {
        std::swap(
            candidate.left_lookahead_x,
            candidate.right_lookahead_x
        );
    }

    if (!isAcceptable(candidate)) {
        ++missed_frames_;

        if (!initialized_ ||
            missed_frames_ > config_.maximum_missed_frames) {
            initialized_ = false;
            previous_ = LaneObservation{};

            LaneObservation invalid = candidate;
            invalid.valid = false;
            invalid.confidence = 0.0;
            return invalid;
        }

        LaneObservation held = previous_;

        held.left_detected = candidate.left_detected;
        held.right_detected = candidate.right_detected;
        held.processing_time_ms = candidate.processing_time_ms;

        held.confidence =
            previous_.confidence *
            std::pow(
                config_.confidence_decay,
                static_cast<double>(missed_frames_)
            );

        held.valid =
            held.confidence >= config_.minimum_confidence;

        return held;
    }

    missed_frames_ = 0;

    if (!initialized_) {
        previous_ = candidate;
        recalculateErrors(previous_);

        initialized_ = true;
        return previous_;
    }

    LaneObservation filtered = candidate;
    const double alpha = config_.smoothing_alpha;

    filtered.left_bottom_x = smooth(
        candidate.left_bottom_x,
        previous_.left_bottom_x,
        alpha
    );

    filtered.right_bottom_x = smooth(
        candidate.right_bottom_x,
        previous_.right_bottom_x,
        alpha
    );

    filtered.left_lookahead_x = smooth(
        candidate.left_lookahead_x,
        previous_.left_lookahead_x,
        alpha
    );

    filtered.right_lookahead_x = smooth(
        candidate.right_lookahead_x,
        previous_.right_lookahead_x,
        alpha
    );

    filtered.left_curve.a = smooth(
        candidate.left_curve.a,
        previous_.left_curve.a,
        alpha
    );

    filtered.left_curve.b = smooth(
        candidate.left_curve.b,
        previous_.left_curve.b,
        alpha
    );

    filtered.left_curve.c = smooth(
        candidate.left_curve.c,
        previous_.left_curve.c,
        alpha
    );

    filtered.right_curve.a = smooth(
        candidate.right_curve.a,
        previous_.right_curve.a,
        alpha
    );

    filtered.right_curve.b = smooth(
        candidate.right_curve.b,
        previous_.right_curve.b,
        alpha
    );

    filtered.right_curve.c = smooth(
        candidate.right_curve.c,
        previous_.right_curve.c,
        alpha
    );

    filtered.confidence = std::clamp(
        smooth(
            candidate.confidence,
            previous_.confidence,
            alpha
        ),
        0.0,
        1.0
    );

    filtered.valid =
        filtered.confidence >= config_.minimum_confidence;

    recalculateErrors(filtered);

    previous_ = filtered;
    return filtered;
}

bool LaneTracker::isAcceptable(
    const LaneObservation& detection
) const noexcept {
    if (!detection.valid ||
        !detection.left_detected ||
        !detection.right_detected ||
        !isFiniteGeometry(detection)) {
        return false;
    }

    if (detection.left_bottom_x >= detection.right_bottom_x ||
        detection.left_lookahead_x >=
            detection.right_lookahead_x) {
        return false;
    }

    if (!initialized_) {
        return true;
    }

    const double left_bottom_jump = std::abs(
        detection.left_bottom_x -
        previous_.left_bottom_x
    );

    const double right_bottom_jump = std::abs(
        detection.right_bottom_x -
        previous_.right_bottom_x
    );

    const double left_lookahead_jump = std::abs(
        detection.left_lookahead_x -
        previous_.left_lookahead_x
    );

    const double right_lookahead_jump = std::abs(
        detection.right_lookahead_x -
        previous_.right_lookahead_x
    );

    if (left_bottom_jump >
            config_.maximum_boundary_jump_px ||
        right_bottom_jump >
            config_.maximum_boundary_jump_px ||
        left_lookahead_jump >
            config_.maximum_boundary_jump_px ||
        right_lookahead_jump >
            config_.maximum_boundary_jump_px) {
        return false;
    }

    const double current_lane_width =
        detection.right_bottom_x -
        detection.left_bottom_x;

    const double previous_lane_width =
        previous_.right_bottom_x -
        previous_.left_bottom_x;

    for (const double y : {240.0, 300.0, 360.0, 420.0, 479.0}) {
        const double left_x =
            detection.left_curve.a * y * y +
            detection.left_curve.b * y +
            detection.left_curve.c;

        const double right_x =
            detection.right_curve.a * y * y +
            detection.right_curve.b * y +
            detection.right_curve.c;

        const double lane_width = right_x - left_x;

        if (lane_width < 170.0 || lane_width > 450.0) {
            return false;
        }
    }

    return std::abs(
        current_lane_width - previous_lane_width
    ) <= config_.maximum_lane_width_change_px;
}

void LaneTracker::recalculateErrors(
    LaneObservation& observation
) const {
    const double bottom_y =
        static_cast<double>(
            config_.warped_bottom_y_px
        );

    const double lookahead_y =
        static_cast<double>(
            config_.warped_lookahead_y_px
        );

    const double left_bottom_x =
        observation.left_curve.a * bottom_y * bottom_y +
        observation.left_curve.b * bottom_y +
        observation.left_curve.c;

    const double right_bottom_x =
        observation.right_curve.a * bottom_y * bottom_y +
        observation.right_curve.b * bottom_y +
        observation.right_curve.c;

    const double left_lookahead_x =
        observation.left_curve.a *
            lookahead_y * lookahead_y +
        observation.left_curve.b * lookahead_y +
        observation.left_curve.c;

    const double right_lookahead_x =
        observation.right_curve.a *
            lookahead_y * lookahead_y +
        observation.right_curve.b * lookahead_y +
        observation.right_curve.c;

    const double lane_centre_bottom =
        0.5 *
        (
            left_bottom_x +
            right_bottom_x
        );

    const double lane_centre_lookahead =
        0.5 *
        (
            left_lookahead_x +
            right_lookahead_x
        );

    observation.lateral_error_m =
        (
            config_.warped_vehicle_centre_x -
            lane_centre_bottom
        ) /
        config_.warped_pixels_per_metre;

    const double vertical_distance =
        bottom_y - lookahead_y;

    observation.heading_error_rad =
        std::atan2(
            lane_centre_bottom -
                lane_centre_lookahead,
            vertical_distance
        );
}

void LaneTracker::reset() noexcept {
    previous_ = LaneObservation{};
    initialized_ = false;
    missed_frames_ = 0;
}

}  // namespace autonomy
