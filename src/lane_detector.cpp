// Placeholder
#include "autonomy/lane_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace autonomy {
namespace {

struct WeightedLine {
    double weighted_a{0.0};
    double weighted_b{0.0};
    double total_weight{0.0};

    void add(double a, double b, double weight) {
        weighted_a += a * weight;
        weighted_b += b * weight;
        total_weight += weight;
    }

    [[nodiscard]] bool valid() const noexcept {
        return total_weight > 0.0;
    }

    [[nodiscard]] double a() const {
        return weighted_a / total_weight;
    }

    [[nodiscard]] double b() const {
        return weighted_b / total_weight;
    }
};

}  // namespace

LaneDetector::LaneDetector(
    const double pixels_per_metre,
    const int horizon_y_px
)
    : pixels_per_metre_(pixels_per_metre),
      horizon_y_px_(horizon_y_px) {
    if (pixels_per_metre_ <= 0.0 || horizon_y_px_ <= 0) {
        throw std::invalid_argument(
            "Invalid lane detector configuration"
        );
    }
}

LaneObservation LaneDetector::detect(
    const cv::Mat& frame,
    cv::Mat* debug_frame
) const {
    const auto start_time = std::chrono::steady_clock::now();

    LaneObservation observation;

    if (frame.empty()) {
        return observation;
    }

    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
    cv::Canny(blurred, edges, 60.0, 160.0);

    cv::Mat mask = cv::Mat::zeros(edges.size(), edges.type());

    std::vector<cv::Point> polygon{
        {0, edges.rows - 1},
        {edges.cols - 1, edges.rows - 1},
        {static_cast<int>(0.62 * edges.cols), horizon_y_px_},
        {static_cast<int>(0.38 * edges.cols), horizon_y_px_}
    };

    cv::fillConvexPoly(mask, polygon, cv::Scalar(255));

    cv::Mat roi_edges;
    cv::bitwise_and(edges, mask, roi_edges);

    std::vector<cv::Vec4i> segments;

    cv::HoughLinesP(
        roi_edges,
        segments,
        1.0,
        CV_PI / 180.0,
        30,
        35.0,
        25.0
    );

    WeightedLine left_line;
    WeightedLine right_line;

    double total_line_length = 0.0;

    const double image_centre_x =
        static_cast<double>(frame.cols) / 2.0;

    cv::Mat local_debug = frame.clone();

    for (const auto& segment : segments) {
        const double x1 = segment[0];
        const double y1 = segment[1];
        const double x2 = segment[2];
        const double y2 = segment[3];

        const double dy = y2 - y1;

        if (std::abs(dy) < 1.0) {
            continue;
        }

        const double a = (x2 - x1) / dy;
        const double b = x1 - a * y1;
        const double length = std::hypot(x2 - x1, y2 - y1);
        const double midpoint_x = 0.5 * (x1 + x2);

        if (std::abs(a) < 0.15) {
            continue;
        }

        if (a < 0.0 && midpoint_x < image_centre_x) {
            left_line.add(a, b, length);
            total_line_length += length;
            cv::line(
                local_debug,
                {segment[0], segment[1]},
                {segment[2], segment[3]},
                cv::Scalar(255, 0, 0),
                2
            );
        } else if (a > 0.0 && midpoint_x > image_centre_x) {
            right_line.add(a, b, length);
            total_line_length += length;
            cv::line(
                local_debug,
                {segment[0], segment[1]},
                {segment[2], segment[3]},
                cv::Scalar(0, 0, 255),
                2
            );
        }
    }

    observation.left_detected = left_line.valid();
    observation.right_detected = right_line.valid();

    if (!left_line.valid() || !right_line.valid()) {
        observation.valid = false;

        const auto end_time = std::chrono::steady_clock::now();
        observation.processing_time_ms =
            std::chrono::duration<double, std::milli>(
                end_time - start_time
            ).count();

        if (debug_frame != nullptr) {
            *debug_frame = local_debug;
        }

        return observation;
    }

    const int bottom_y = frame.rows - 1;
    const int lookahead_y =
        static_cast<int>(0.60 * frame.rows);

    const double left_bottom_x =
        left_line.a() * bottom_y + left_line.b();

    const double right_bottom_x =
        right_line.a() * bottom_y + right_line.b();

    const double left_lookahead_x =
        left_line.a() * lookahead_y + left_line.b();

    const double right_lookahead_x =
        right_line.a() * lookahead_y + right_line.b();

    const double lane_centre_bottom =
        0.5 * (left_bottom_x + right_bottom_x);

    const double lane_centre_lookahead =
        0.5 * (left_lookahead_x + right_lookahead_x);

    observation.lateral_error_m =
        (image_centre_x - lane_centre_bottom) /
        pixels_per_metre_;

    const double vertical_distance =
        static_cast<double>(bottom_y - lookahead_y);

    observation.heading_error_rad = std::atan2(
        lane_centre_bottom - lane_centre_lookahead,
        vertical_distance
    );

    const double length_score = std::clamp(
        total_line_length / 800.0,
        0.0,
        0.3
    );

    observation.confidence = 0.7 + length_score;
    observation.valid = observation.confidence >= 0.35;

    cv::circle(
        local_debug,
        {
            static_cast<int>(lane_centre_bottom),
            bottom_y - 15
        },
        8,
        cv::Scalar(0, 255, 0),
        cv::FILLED
    );

    const auto end_time = std::chrono::steady_clock::now();

    observation.processing_time_ms =
        std::chrono::duration<double, std::milli>(
            end_time - start_time
        ).count();

    if (debug_frame != nullptr) {
        *debug_frame = local_debug;
    }

    return observation;
}

}  // namespace autonomy