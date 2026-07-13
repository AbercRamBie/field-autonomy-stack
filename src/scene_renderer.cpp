// Placeholder
#include "autonomy/scene_renderer.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <stdexcept>
#include <vector>

namespace autonomy {

SceneRenderer::SceneRenderer(RenderConfig config)
    : config_(config) {
    if (config_.width_px <= 0 ||
        config_.height_px <= 0 ||
        config_.horizon_y_px <= 0 ||
        config_.horizon_y_px >= config_.height_px) {
        throw std::invalid_argument("Invalid render configuration");
    }
}

cv::Mat SceneRenderer::render(
    const VehicleState& state,
    const double lane_width_m,
    const FaultState& faults
) {
    cv::Mat frame(
        config_.height_px,
        config_.width_px,
        CV_8UC3,
        cv::Scalar(35, 35, 35)
    );

    const int bottom_y = config_.height_px - 1;
    const int horizon_y = config_.horizon_y_px;
    const double image_centre_x =
        static_cast<double>(config_.width_px) / 2.0;

    const double bottom_centre_x =
        image_centre_x -
        state.lateral_error_m * config_.pixels_per_metre;

    const double perspective_height =
        static_cast<double>(bottom_y - horizon_y);

    const double horizon_centre_x =
        bottom_centre_x -
        std::tan(state.heading_error_rad) * perspective_height;

    const double bottom_half_width =
        0.5 * lane_width_m * config_.pixels_per_metre;

    const double horizon_half_width = 18.0;

    const cv::Point left_bottom(
        static_cast<int>(bottom_centre_x - bottom_half_width),
        bottom_y
    );

    const cv::Point left_horizon(
        static_cast<int>(horizon_centre_x - horizon_half_width),
        horizon_y
    );

    const cv::Point right_bottom(
        static_cast<int>(bottom_centre_x + bottom_half_width),
        bottom_y
    );

    const cv::Point right_horizon(
        static_cast<int>(horizon_centre_x + horizon_half_width),
        horizon_y
    );

    std::vector<cv::Point> road_polygon{
        left_bottom,
        left_horizon,
        right_horizon,
        right_bottom
    };

    cv::fillConvexPoly(
        frame,
        road_polygon,
        cv::Scalar(70, 70, 70)
    );

    cv::line(
        frame,
        left_bottom,
        left_horizon,
        cv::Scalar(240, 240, 240),
        7,
        cv::LINE_AA
    );

    cv::line(
        frame,
        right_bottom,
        right_horizon,
        cv::Scalar(240, 240, 240),
        7,
        cv::LINE_AA
    );

    if (faults.camera_noisy) {
        cv::Mat noise(frame.size(), frame.type());

        cv::randn(
            noise,
            cv::Scalar::all(0),
            cv::Scalar::all(35)
        );

        cv::add(frame, noise, frame);
    }

    if (faults.lane_occluded) {
        cv::rectangle(
            frame,
            cv::Rect(
                frame.cols / 4,
                frame.rows / 2,
                frame.cols / 2,
                frame.rows / 3
            ),
            cv::Scalar(20, 20, 20),
            cv::FILLED
        );
    }

    return frame;
}

}  // namespace autonomy