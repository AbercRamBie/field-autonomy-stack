#pragma once
#include "autonomy/types.hpp"
#include <opencv2/core.hpp>
#include <deque>

namespace autonomy {

enum class LaneDetectorProfile{
    Synthetic,
    RecordedVideo
};

class LaneDetector {
public:
    LaneDetector(
        double pixels_per_metre = 75.0,
        int horizon_y_px = 300,
        LaneDetectorProfile profile = LaneDetectorProfile::RecordedVideo
    );

    [[nodiscard]] LaneObservation detect(
        const cv::Mat& frame,
        cv::Mat* debug_frame = nullptr
    );

private:
    cv::Mat createBinaryImage(
        const cv::Mat& warped_frame
    ) const;

    double pixels_per_metre_;
    int horizon_y_px_;

    cv::Mat perspective_transform_;
    cv::Mat inverse_perspective_transform_;

    bool perspective_initialized_{false};

    std::deque<LanePolynomial> left_curve_history_;
    std::deque<LanePolynomial> right_curve_history_;
    int missed_detection_frames_{0};

    LaneDetectorProfile profile_;
};

}  // namespace autonomy
