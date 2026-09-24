#pragma once

#include "autonomy/types.hpp"

namespace autonomy {

struct LaneTrackerConfig {
    // Polynomial smoothing is performed by LaneDetector using the
    // rolling ten-frame mean from the reference implementation.
    double smoothing_alpha{1.0};
    double maximum_boundary_jump_px{50.0};
    double maximum_lane_width_change_px{70.0};
    double warped_pixels_per_metre{320.0 / 3.5};
    double warped_vehicle_centre_x{320.0};
    int warped_bottom_y_px{479};
    int warped_lookahead_y_px{280};
    int maximum_missed_frames{5};
    double confidence_decay{0.80};
    double minimum_confidence{0.35};
    double maximum_absolute_lateral_error_m{0.75};
    double maximum_absolute_heading_error_rad{0.20};
    double maximum_relative_lateral_error_m{0.50};
    double maximum_relative_heading_error_rad{0.15};   
};

class LaneTracker {
public:
    explicit LaneTracker(LaneTrackerConfig config = {});

    [[nodiscard]] LaneObservation update(
        const LaneObservation& detection
    );

    void reset() noexcept;

private:
    [[nodiscard]] bool isAcceptable(
        const LaneObservation& detection
    ) const noexcept;

    void recalculateErrors(
        LaneObservation& observation
    ) const;

    LaneTrackerConfig config_;
    LaneObservation previous_{};

    bool initialized_{false};
    int missed_frames_{0};
};

}  // namespace autonomy
