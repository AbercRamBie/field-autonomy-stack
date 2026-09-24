#include "autonomy/lane_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace autonomy {
namespace {

constexpr int kExpectedFrameWidth = 640;
constexpr int kExpectedFrameHeight = 480;

struct Polynomial {
    double a{0.0};
    double b{0.0};
    double c{0.0};
    bool valid{false};

    [[nodiscard]] double xAt(const double y) const noexcept {
        return a * y * y + b * y + c;
    }
};

struct SlidingWindowResult {
    std::vector<cv::Point2f> left_pixels;
    std::vector<cv::Point2f> right_pixels;

    int left_base_x{0};
    int right_base_x{0};

    bool valid{false};
};

std::vector<cv::Point2f> createSourcePoints(
    const cv::Size& image_size,
    const LaneDetectorProfile profile
) {
    const float scale_x =
        static_cast<float>(image_size.width) /
        static_cast<float>(kExpectedFrameWidth);

    const float scale_y =
        static_cast<float>(image_size.height) /
        static_cast<float>(kExpectedFrameHeight);

    if(profile == LaneDetectorProfile::Synthetic){

        return {
            {302.0F * scale_x, 150.0F * scale_y},
            {338.0F * scale_x, 150.0F * scale_y},
            {189.0F * scale_x, 479.0F * scale_y},
            {451.0F * scale_x, 479.0F * scale_y}
        };
    }


    /*
    * Calibration for the included recorded-video camera.
    */
    return {
        {312.0F * scale_x, 335.0F * scale_y},
        {328.0F * scale_x, 335.0F * scale_y},
        {158.0F * scale_x, 479.0F * scale_y},
        {425.0F * scale_x, 479.0F * scale_y}
    };

}

std::vector<cv::Point2f> createRoiPoints(
    const cv::Size& image_size,
    const LaneDetectorProfile profile
) {
    const float scale_x =
        static_cast<float>(image_size.width) /
        static_cast<float>(kExpectedFrameWidth);

    const float scale_y =
        static_cast<float>(image_size.height) /
        static_cast<float>(kExpectedFrameHeight);

    if (profile == LaneDetectorProfile::Synthetic) {
        /*
         * Wider than the nominal synthetic lane so boundaries
         * remain visible during lateral and heading errors.
         */
        return {
            {220.0F * scale_x, 140.0F * scale_y},
            {420.0F * scale_x, 140.0F * scale_y},
            {80.0F * scale_x, 479.0F * scale_y},
            {560.0F * scale_x, 479.0F * scale_y}
        };
    }

    /*
     * Initially retain the calibrated video trapezoid.
     * We will tune its ROI independently using video evidence.
     */
    return createSourcePoints(
        image_size,
        profile
    );
}

std::vector<cv::Point2f> createDestinationPoints(
    const cv::Size& image_size
) {
    const float scale_x =
        static_cast<float>(image_size.width) /
        static_cast<float>(kExpectedFrameWidth);

    const float bottom_y =
        static_cast<float>(image_size.height - 1);

    return {
        {160.0F * scale_x, 0.0F},
        {480.0F * scale_x, 0.0F},
        {160.0F * scale_x, bottom_y},
        {480.0F * scale_x, bottom_y}
    };
}

cv::Mat gradientBinary(
    const cv::Mat& channel,
    const double minimum_threshold
) {
    if (channel.empty()) {
        return {};
    }

    cv::Mat sobel;

    cv::Sobel(
        channel,
        sobel,
        CV_32F,
        1,
        0,
        3
    );

    cv::Mat absolute_sobel;
    cv::absdiff(
        sobel,
        cv::Scalar::all(0),
        absolute_sobel
    );

    double maximum_gradient = 0.0;
    cv::minMaxLoc(
        absolute_sobel,
        nullptr,
        &maximum_gradient
    );

    cv::Mat scaled;

    if (maximum_gradient > 0.0) {
        cv::convertScaleAbs(
            absolute_sobel,
            scaled,
            255.0 / maximum_gradient
        );
    } else {
        scaled = cv::Mat::zeros(
            channel.size(),
            CV_8UC1
        );
    }

    cv::Mat binary;

    cv::inRange(
        scaled,
        cv::Scalar(minimum_threshold),
        cv::Scalar(255),
        binary
    );

    return binary;
}

Polynomial averagePolynomialHistory(
    const std::deque<LanePolynomial>& history
) {
    Polynomial average;

    if (history.empty()) {
        return average;
    }

    for (const LanePolynomial& curve : history) {
        average.a += curve.a;
        average.b += curve.b;
        average.c += curve.c;
    }

    const double count =
        static_cast<double>(history.size());

    average.a /= count;
    average.b /= count;
    average.c /= count;
    average.valid =
        std::isfinite(average.a) &&
        std::isfinite(average.b) &&
        std::isfinite(average.c);

    return average;
}

Polynomial fitPolynomial(
    const std::vector<cv::Point2f>& points
) {
    Polynomial result;

    constexpr std::size_t minimum_points = 100;

    if (points.size() < minimum_points) {
        return result;
    }

    cv::Mat design(
        static_cast<int>(points.size()),
        3,
        CV_64F
    );

    cv::Mat values(
        static_cast<int>(points.size()),
        1,
        CV_64F
    );

    for (std::size_t index = 0; index < points.size(); ++index) {
        const double y =
            static_cast<double>(points[index].y);

        const double x =
            static_cast<double>(points[index].x);

        design.at<double>(
            static_cast<int>(index),
            0
        ) = y * y;

        design.at<double>(
            static_cast<int>(index),
            1
        ) = y;

        design.at<double>(
            static_cast<int>(index),
            2
        ) = 1.0;

        values.at<double>(
            static_cast<int>(index),
            0
        ) = x;
    }

    cv::Mat coefficients;

    const bool solved = cv::solve(
        design,
        values,
        coefficients,
        cv::DECOMP_SVD
    );

    if (!solved) {
        return result;
    }

    result.a = coefficients.at<double>(0, 0);
    result.b = coefficients.at<double>(1, 0);
    result.c = coefficients.at<double>(2, 0);

    result.valid =
        std::isfinite(result.a) &&
        std::isfinite(result.b) &&
        std::isfinite(result.c);

    return result;
}

SlidingWindowResult findLanePixels(
    const cv::Mat& binary,
    cv::Mat* debug_frame = nullptr
) {
    SlidingWindowResult result;

    if (binary.empty() || binary.type() != CV_8UC1) {
        return result;
    }

    constexpr int number_of_windows = 9;
    constexpr int margin_px = 75;
    constexpr int minimum_pixels_to_recenter = 1;
    constexpr std::size_t minimum_lane_pixels = 100;

    const int width = binary.cols;
    const int height = binary.rows;
    const int left_search_start =
        static_cast<int>(0.18 * static_cast<double>(width));
    const int left_search_end =
        static_cast<int>(0.38 * static_cast<double>(width));
    const int right_search_start =
        static_cast<int>(0.62 * static_cast<double>(width));
    const int right_search_end =
        static_cast<int>(0.82 * static_cast<double>(width));

    if (width < 2 || height < number_of_windows) {
        return result;
    }

    const cv::Mat bottom_half = binary(
        cv::Rect(
            0,
            height / 2,
            width,
            height - height / 2
        )
    );

    cv::Mat histogram;

    cv::reduce(
        bottom_half,
        histogram,
        0,
        cv::REDUCE_SUM,
        CV_32S
    );

    cv::Point left_peak;
    cv::Point right_peak;

    cv::minMaxLoc(
        histogram(
            cv::Rect(
                left_search_start,
                0,
                left_search_end - left_search_start,
                1
            )
        ),
        nullptr,
        nullptr,
        nullptr,
        &left_peak
    );

    cv::minMaxLoc(
        histogram(
            cv::Rect(
                right_search_start,
                0,
                right_search_end - right_search_start,
                1
            )
        ),
        nullptr,
        nullptr,
        nullptr,
        &right_peak
    );

    result.left_base_x =
        left_peak.x + left_search_start;
    result.right_base_x =
        right_peak.x + right_search_start;

    int current_left_x = result.left_base_x;
    int current_right_x = result.right_base_x;

    std::vector<cv::Point> nonzero_pixels;
    cv::findNonZero(binary, nonzero_pixels);

    const int window_height =
        height / number_of_windows;

    for (
        int window = 0;
        window < number_of_windows;
        ++window
    ) {
        const int y_high =
            height - window * window_height;

        const int y_low =
            height - (window + 1) * window_height;

        const int left_x_low =
            current_left_x - margin_px;

        const int left_x_high =
            current_left_x + margin_px;

        const int right_x_low =
            current_right_x - margin_px;

        const int right_x_high =
            current_right_x + margin_px;

        std::vector<cv::Point> current_left_pixels;
        std::vector<cv::Point> current_right_pixels;

        for (const cv::Point& point : nonzero_pixels) {
            if (point.y < y_low || point.y >= y_high) {
                continue;
            }

            if (
                point.x >= left_x_low &&
                point.x < left_x_high
            ) {
                current_left_pixels.push_back(point);

                result.left_pixels.emplace_back(
                    static_cast<float>(point.x),
                    static_cast<float>(point.y)
                );
            }

            if (
                point.x >= right_x_low &&
                point.x < right_x_high
            ) {
                current_right_pixels.push_back(point);

                result.right_pixels.emplace_back(
                    static_cast<float>(point.x),
                    static_cast<float>(point.y)
                );
            }
        }

        if (
            static_cast<int>(current_left_pixels.size()) >=
            minimum_pixels_to_recenter
        ) {
            double sum_x = 0.0;

            for (const cv::Point& point : current_left_pixels) {
                sum_x += static_cast<double>(point.x);
            }

            current_left_x = static_cast<int>(
                std::lround(
                    sum_x /
                    static_cast<double>(
                        current_left_pixels.size()
                    )
                )
            );
        }

        if (
            static_cast<int>(current_right_pixels.size()) >=
            minimum_pixels_to_recenter
        ) {
            double sum_x = 0.0;

            for (const cv::Point& point : current_right_pixels) {
                sum_x += static_cast<double>(point.x);
            }

            current_right_x = static_cast<int>(
                std::lround(
                    sum_x /
                    static_cast<double>(
                        current_right_pixels.size()
                    )
                )
            );
        }

        if (debug_frame != nullptr) {
            cv::rectangle(
                *debug_frame,
                cv::Point(left_x_low, y_low),
                cv::Point(left_x_high, y_high),
                cv::Scalar(255, 0, 0),
                2
            );

            cv::rectangle(
                *debug_frame,
                cv::Point(right_x_low, y_low),
                cv::Point(right_x_high, y_high),
                cv::Scalar(0, 0, 255),
                2
            );
        }
    }

    result.valid =
        result.left_pixels.size() >= minimum_lane_pixels &&
        result.right_pixels.size() >= minimum_lane_pixels;

    return result;
}

bool validateLane(
    const Polynomial& left_curve,
    const Polynomial& right_curve,
    const cv::Size& image_size
) {
    if (!left_curve.valid || !right_curve.valid) {
        return false;
    }

    const double width_scale =
        static_cast<double>(image_size.width) /
        static_cast<double>(kExpectedFrameWidth);

    const double minimum_lane_width_px =
        170.0 * width_scale;

    const double maximum_lane_width_px =
        450.0 * width_scale;

    const double maximum_width_variation_px =
        180.0 * width_scale;

    const double boundary_margin =
        100.0 * width_scale;

    const std::vector<double> sample_rows{
        0.50 * static_cast<double>(image_size.height - 1),
        0.65 * static_cast<double>(image_size.height - 1),
        0.75 * static_cast<double>(image_size.height - 1),
        static_cast<double>(image_size.height - 1)
    };

    double minimum_observed_width =
        std::numeric_limits<double>::max();

    double maximum_observed_width = 0.0;

    for (const double y : sample_rows) {
        const double left_x = left_curve.xAt(y);
        const double right_x = right_curve.xAt(y);

        if (
            !std::isfinite(left_x) ||
            !std::isfinite(right_x)
        ) {
            return false;
        }

        if (left_x >= right_x) {
            return false;
        }

        const double lane_width =
            right_x - left_x;

        if (
            lane_width < minimum_lane_width_px ||
            lane_width > maximum_lane_width_px
        ) {
            return false;
        }

        if (
            left_x < -boundary_margin ||
            right_x >
                static_cast<double>(image_size.width) +
                boundary_margin
        ) {
            return false;
        }

        minimum_observed_width =
            std::min(
                minimum_observed_width,
                lane_width
            );

        maximum_observed_width =
            std::max(
                maximum_observed_width,
                lane_width
            );
    }

    /*
     * The calibrated bird's-eye lane bases are near 25% and
     * 75% of the image width. Reject fits that have drifted to
     * an adjacent lane, shoulder edge, or guardrail.
     */
    const double bottom_y =
        static_cast<double>(image_size.height - 1);
    const double left_bottom_x = left_curve.xAt(bottom_y);
    const double right_bottom_x = right_curve.xAt(bottom_y);
    const double minimum_right_base =
        0.55 * static_cast<double>(image_size.width);
    const double maximum_right_base =
        0.90 * static_cast<double>(image_size.width);

    if (right_bottom_x < minimum_right_base ||
        right_bottom_x > maximum_right_base ||
        left_bottom_x < -boundary_margin ||
        left_bottom_x > 0.45 * static_cast<double>(image_size.width)) {
        return false;
    }

    return
        maximum_observed_width -
        minimum_observed_width <=
        maximum_width_variation_px;
}

std::vector<cv::Point> createCurvePoints(
    const Polynomial& curve,
    const int image_height,
    const int step = 2
) {
    std::vector<cv::Point> points;

    if (!curve.valid || image_height <= 0 || step <= 0) {
        return points;
    }

    points.reserve(
        static_cast<std::size_t>(
            image_height / step + 1
        )
    );

    for (int y = 0; y < image_height; y += step) {
        points.emplace_back(
            static_cast<int>(
                std::lround(curve.xAt(y))
            ),
            y
        );
    }

    if (points.empty() ||
        points.back().y != image_height - 1) {
        const int bottom_y = image_height - 1;

        points.emplace_back(
            static_cast<int>(
                std::lround(curve.xAt(bottom_y))
            ),
            bottom_y
        );
    }

    return points;
}

std::vector<cv::Point2f> createCurvePointsFloat(
    const Polynomial& curve,
    const int image_height
) {
    std::vector<cv::Point2f> points;

    if (!curve.valid || image_height <= 0) {
        return points;
    }

    points.reserve(
        static_cast<std::size_t>(image_height)
    );

    for (int y = 0; y < image_height; ++y) {
        points.emplace_back(
            static_cast<float>(curve.xAt(y)),
            static_cast<float>(y)
        );
    }

    return points;
}

double closestXAtY(
    const std::vector<cv::Point2f>& points,
    const double target_y
) {
    if (points.empty()) {
        return 0.0;
    }

    const auto closest = std::min_element(
        points.begin(),
        points.end(),
        [target_y](
            const cv::Point2f& left,
            const cv::Point2f& right
        ) {
            return
                std::abs(
                    static_cast<double>(left.y) -
                    target_y
                ) <
                std::abs(
                    static_cast<double>(right.y) -
                    target_y
                );
        }
    );

    return static_cast<double>(closest->x);
}

cv::Mat createLaneOverlay(
    const cv::Mat& frame,
    const Polynomial& left_curve,
    const Polynomial& right_curve,
    const cv::Mat& inverse_perspective_matrix
) {
    const std::vector<cv::Point> left_points =
        createCurvePoints(
            left_curve,
            frame.rows
        );

    const std::vector<cv::Point> right_points =
        createCurvePoints(
            right_curve,
            frame.rows
        );

    if (left_points.empty() || right_points.empty()) {
        return frame.clone();
    }

    std::vector<cv::Point> lane_polygon;
    lane_polygon.reserve(
        left_points.size() +
        right_points.size()
    );

    lane_polygon.insert(
        lane_polygon.end(),
        left_points.begin(),
        left_points.end()
    );

    for (
        auto iterator = right_points.rbegin();
        iterator != right_points.rend();
        ++iterator
    ) {
        lane_polygon.push_back(*iterator);
    }

    cv::Mat warped_overlay = cv::Mat::zeros(
        frame.size(),
        CV_8UC3
    );

    cv::fillPoly(
        warped_overlay,
        std::vector<std::vector<cv::Point>>{
            lane_polygon
        },
        cv::Scalar(0, 160, 0)
    );

    cv::polylines(
        warped_overlay,
        left_points,
        false,
        cv::Scalar(255, 0, 0),
        6,
        cv::LINE_AA
    );

    cv::polylines(
        warped_overlay,
        right_points,
        false,
        cv::Scalar(0, 0, 255),
        6,
        cv::LINE_AA
    );

    cv::Mat camera_overlay;

    cv::warpPerspective(
        warped_overlay,
        camera_overlay,
        inverse_perspective_matrix,
        frame.size(),
        cv::INTER_LINEAR
    );

    cv::Mat result;

    cv::addWeighted(
        frame,
        1.0,
        camera_overlay,
        0.35,
        0.0,
        result
    );

    return result;
}

}  // namespace

LaneDetector::LaneDetector(
    const double pixels_per_metre,
    const int horizon_y_px,
    const LaneDetectorProfile profile
)
    : pixels_per_metre_(pixels_per_metre),
      horizon_y_px_(horizon_y_px),
      profile_(profile) {
    if (
        pixels_per_metre_ <= 0.0 ||
        horizon_y_px_ <= 0
    ) {
        throw std::invalid_argument(
            "Invalid lane detector configuration"
        );
    }

    const cv::Size expected_size{
        kExpectedFrameWidth,
        kExpectedFrameHeight
    };

    const std::vector<cv::Point2f> source_points =
        createSourcePoints(expected_size, profile_);

    const std::vector<cv::Point2f> destination_points =
        createDestinationPoints(expected_size);

    perspective_transform_ =
        cv::getPerspectiveTransform(
            source_points,
            destination_points
        );

    inverse_perspective_transform_ =
        cv::getPerspectiveTransform(
            destination_points,
            source_points
        );

    perspective_initialized_ = true;
}

cv::Mat LaneDetector::createBinaryImage(
    const cv::Mat& frame
) const {
    if (frame.empty()) {
        return {};
    }

    cv::Mat hls;

    cv::cvtColor(
        frame,
        hls,
        cv::COLOR_BGR2HLS
    );

    std::vector<cv::Mat> channels;
    cv::split(hls, channels);

    if (channels.size() < 3) {
        return {};
    }

    const cv::Mat lightness_binary =
        gradientBinary(
            channels[1],
            15.0
        );

    cv::Mat saturation_binary;
    cv::inRange(
        channels[2],
        cv::Scalar(100),
        cv::Scalar(255),
        saturation_binary
    );

    cv::Mat white_binary;
    cv::inRange(
        channels[1],
        cv::Scalar(200),
        cv::Scalar(255),
        white_binary
    );

    cv::Mat binary;

    cv::bitwise_or(
        lightness_binary,
        saturation_binary,
        binary
    );

    cv::bitwise_or(
        binary,
        white_binary,
        binary
    );

    return binary;
}

LaneObservation LaneDetector::detect(
    const cv::Mat& frame,
    cv::Mat* debug_frame
) {
    const auto start_time =
        std::chrono::steady_clock::now();

    LaneObservation observation;

    if (frame.empty()) {
        return observation;
    }

    auto finish = [&](
        LaneObservation result,
        const cv::Mat& display_frame
    ) {
        const auto end_time =
            std::chrono::steady_clock::now();

        result.processing_time_ms =
            std::chrono::duration<double, std::milli>(
                end_time - start_time
            ).count();

        if (debug_frame != nullptr) {
            if (display_frame.empty()) {
                *debug_frame = frame.clone();
            } else {
                *debug_frame = display_frame.clone();
            }
        }

        return result;
    };

    auto record_missed_detection = [&]() {
        constexpr int history_reset_frames = 10;

        ++missed_detection_frames_;
        if (missed_detection_frames_ >= history_reset_frames) {
            left_curve_history_.clear();
            right_curve_history_.clear();
        }
    };

    cv::Mat perspective_matrix;
    cv::Mat inverse_perspective_matrix;

    if (
        perspective_initialized_ &&
        frame.cols == kExpectedFrameWidth &&
        frame.rows == kExpectedFrameHeight
    ) {
        perspective_matrix =
            perspective_transform_;

        inverse_perspective_matrix =
            inverse_perspective_transform_;
    } else {
        const std::vector<cv::Point2f> source_points =
            createSourcePoints(frame.size(), profile_);

        const std::vector<cv::Point2f> destination_points =
            createDestinationPoints(frame.size());

        perspective_matrix =
            cv::getPerspectiveTransform(
                source_points,
                destination_points
            );

        inverse_perspective_matrix =
            cv::getPerspectiveTransform(
                destination_points,
                source_points
            );
    }

    /*
     * Step 1: Create the HLS/Sobel mask in camera space. The
     * reference implementation thresholds before warping.
     */
    const cv::Mat camera_binary =
        createBinaryImage(frame);

    if (camera_binary.empty()) {
        return finish(
            observation,
            frame
        );
    }

    const std::vector<cv::Point2f> roi_source_points =
        createRoiPoints(frame.size(), profile_);

    const auto toPoint = [](
        const cv::Point2f& point
    ) {
        return cv::Point{
            cvRound(point.x),
            cvRound(point.y)
        };
    };

    const std::vector<cv::Point> roi_polygon{
        toPoint(roi_source_points[0]),
        toPoint(roi_source_points[1]),
        toPoint(roi_source_points[3]),
        toPoint(roi_source_points[2])
    };

    cv::Mat roi_mask = cv::Mat::zeros(
        frame.size(),
        CV_8UC1
    );

    cv::fillConvexPoly(
        roi_mask,
        roi_polygon,
        cv::Scalar(255),
        cv::LINE_AA
    );

    cv::Mat masked_camera_binary;

    cv::bitwise_and(
        camera_binary,
        roi_mask,
        masked_camera_binary
    );

    cv::Mat masked_frame = cv::Mat::zeros(
        frame.size(),
        frame.type()
    );

    frame.copyTo(
        masked_frame,
        roi_mask
    );

    /*
     * Step 2: Warp both the display frame and binary mask.
     */
    cv::Mat warped;

    cv::warpPerspective(
        masked_frame,
        warped,
        perspective_matrix,
        frame.size(),
        cv::INTER_LINEAR
    );

    if (warped.empty()) {
        return finish(
            observation,
            frame
        );
    }

    if (debug_frame != nullptr) {
        cv::imshow("1 - Warped", warped);
    }

    cv::Mat binary;
    cv::warpPerspective(
        masked_camera_binary,
        binary,
        perspective_matrix,
        frame.size(),
        cv::INTER_NEAREST
    );

    if (debug_frame != nullptr) {
        cv::imshow("2 - Binary", binary);
    }

    /*
     * Step 3: Histogram and sliding-window search.
     */
    cv::Mat sliding_window_debug;

    cv::cvtColor(
        binary,
        sliding_window_debug,
        cv::COLOR_GRAY2BGR
    );

    const SlidingWindowResult lane_pixels =
        findLanePixels(
            binary,
            &sliding_window_debug
        );

    if (debug_frame != nullptr) {
        cv::imshow(
            "3 - Sliding windows",
            sliding_window_debug
        );
    }

    if (!lane_pixels.valid) {
        record_missed_detection();

        return finish(
            observation,
            frame
        );
    }

    /*
     * Step 4: Fit quadratic lane curves.
     */
    const Polynomial measured_left_curve =
        fitPolynomial(
            lane_pixels.left_pixels
        );

    const Polynomial measured_right_curve =
        fitPolynomial(
            lane_pixels.right_pixels
        );

    observation.left_detected =
        measured_left_curve.valid;

    observation.right_detected =
        measured_right_curve.valid;

    /*
     * Step 5: Validate the detected lane geometry.
     */
    if (!measured_left_curve.valid ||
        !measured_right_curve.valid) {
        record_missed_detection();

        return finish(
            observation,
            frame
        );
    }

    missed_detection_frames_ = 0;

    constexpr std::size_t history_length = 10;

    left_curve_history_.push_back(
        LanePolynomial{
            measured_left_curve.a,
            measured_left_curve.b,
            measured_left_curve.c
        }
    );
    right_curve_history_.push_back(
        LanePolynomial{
            measured_right_curve.a,
            measured_right_curve.b,
            measured_right_curve.c
        }
    );

    while (left_curve_history_.size() > history_length) {
        left_curve_history_.pop_front();
    }
    while (right_curve_history_.size() > history_length) {
        right_curve_history_.pop_front();
    }

    const Polynomial left_curve =
        averagePolynomialHistory(left_curve_history_);
    const Polynomial right_curve =
        averagePolynomialHistory(right_curve_history_);

    const bool lane_geometry_valid =
        validateLane(
            left_curve,
            right_curve,
            binary.size()
        );

    if (!lane_geometry_valid) {
        left_curve_history_.pop_back();
        right_curve_history_.pop_back();
        record_missed_detection();

        return finish(
            observation,
            frame
        );
    }

    /*
    * Copy the independently fitted curves into LaneObservation.
    */

    observation.left_curve.a = left_curve.a;
    observation.left_curve.b = left_curve.b;
    observation.left_curve.c = left_curve.c;

    observation.right_curve.a = right_curve.a;
    observation.right_curve.b = right_curve.b;
    observation.right_curve.c = right_curve.c;

    /*
     * Step 6: Transform sampled curve points back into
     * camera-image coordinates.
     */
    const std::vector<cv::Point2f> left_warped_points =
        createCurvePointsFloat(
            left_curve,
            frame.rows
        );

    const std::vector<cv::Point2f> right_warped_points =
        createCurvePointsFloat(
            right_curve,
            frame.rows
        );

    std::vector<cv::Point2f> left_camera_points;
    std::vector<cv::Point2f> right_camera_points;

    cv::perspectiveTransform(
        left_warped_points,
        left_camera_points,
        inverse_perspective_matrix
    );

    cv::perspectiveTransform(
        right_warped_points,
        right_camera_points,
        inverse_perspective_matrix
    );

    const int bottom_y =
        frame.rows - 1;

    const int lookahead_y =
        static_cast<int>(
            0.70 *
            static_cast<double>(frame.rows)
        );

    observation.left_bottom_x =
        closestXAtY(
            left_camera_points,
            static_cast<double>(bottom_y)
        );

    observation.right_bottom_x =
        closestXAtY(
            right_camera_points,
            static_cast<double>(bottom_y)
        );

    observation.left_lookahead_x =
        closestXAtY(
            left_camera_points,
            static_cast<double>(lookahead_y)
        );

    observation.right_lookahead_x =
        closestXAtY(
            right_camera_points,
            static_cast<double>(lookahead_y)
        );

    if (
        observation.left_bottom_x >=
            observation.right_bottom_x ||
        observation.left_lookahead_x >=
            observation.right_lookahead_x
    ) {
        return finish(
            observation,
            frame
        );
    }

    /*
     * Step 7: Calculate camera-space lateral and heading
     * measurements for the existing tracker/controller.
     */
    const double image_centre_x =
        0.5 * static_cast<double>(frame.cols);

    const double lane_centre_bottom =
        0.5 *
        (
            observation.left_bottom_x +
            observation.right_bottom_x
        );

    const double lane_centre_lookahead =
        0.5 *
        (
            observation.left_lookahead_x +
            observation.right_lookahead_x
        );

    observation.lateral_error_m =
        (
            image_centre_x -
            lane_centre_bottom
        ) /
        pixels_per_metre_;

    const double vertical_distance =
        static_cast<double>(
            bottom_y - lookahead_y
        );

    observation.heading_error_rad =
        std::atan2(
            lane_centre_bottom -
                lane_centre_lookahead,
            vertical_distance
        );

    /*
     * Use the amount of supporting lane pixels as an
     * initial confidence measurement.
     */
    const std::size_t supporting_pixels =
        std::min(
            lane_pixels.left_pixels.size(),
            lane_pixels.right_pixels.size()
        );

    const double support_score =
        std::clamp(
            static_cast<double>(supporting_pixels) /
                2000.0,
            0.0,
            1.0
        );

    observation.confidence =
        0.4 + 0.6 * support_score;

    observation.valid =
        observation.confidence >= 0.35;

    /*
     * Step 8: Produce the curved lane overlay.
     */
    const cv::Mat overlay =
        createLaneOverlay(
            frame,
            left_curve,
            right_curve,
            inverse_perspective_matrix
        );

    return finish(
        observation,
        overlay
    );
}

}  // namespace autonomy
