#include "autonomy/health_monitor.hpp"
#include "autonomy/lane_detector.hpp"
#include "autonomy/lane_tracker.hpp"
#include "autonomy/stanley_controller.hpp"
#include "autonomy/types.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr int kFrameWidth = 640;
constexpr int kFrameHeight = 480;

constexpr const char* kMainWindowName =
    "Real-video autonomy check";

std::string formatValue(
    const double value,
    const int precision = 3
) {
    std::ostringstream stream;

    stream
        << std::fixed
        << std::setprecision(precision)
        << value;

    return stream.str();
}

bool isMainWindowClosed() {
    return cv::getWindowProperty(
        kMainWindowName,
        cv::WND_PROP_VISIBLE
    ) < 1.0;
}

void drawTrackedLane(
    cv::Mat& frame,
    const autonomy::LaneObservation& observation
) {
    if (!observation.valid) {
        return;
    }

    const int bottom_y = frame.rows - 1;

    const int lookahead_y =
        static_cast<int>(
            0.70 * static_cast<double>(frame.rows)
        );

    const cv::Point left_bottom{
        static_cast<int>(
            std::lround(observation.left_bottom_x)
        ),
        bottom_y
    };

    const cv::Point left_lookahead{
        static_cast<int>(
            std::lround(observation.left_lookahead_x)
        ),
        lookahead_y
    };

    const cv::Point right_bottom{
        static_cast<int>(
            std::lround(observation.right_bottom_x)
        ),
        bottom_y
    };

    const cv::Point right_lookahead{
        static_cast<int>(
            std::lround(observation.right_lookahead_x)
        ),
        lookahead_y
    };

    cv::line(
        frame,
        left_bottom,
        left_lookahead,
        cv::Scalar(255, 0, 0),
        5,
        cv::LINE_AA
    );

    cv::line(
        frame,
        right_bottom,
        right_lookahead,
        cv::Scalar(0, 0, 255),
        5,
        cv::LINE_AA
    );

    const double lane_centre_bottom =
        0.5 *
        (
            observation.left_bottom_x +
            observation.right_bottom_x
        );

    cv::circle(
        frame,
        {
            static_cast<int>(
                std::lround(lane_centre_bottom)
            ),
            bottom_y - 15
        },
        8,
        cv::Scalar(0, 255, 0),
        cv::FILLED,
        cv::LINE_AA
    );
}

void drawStatus(
    cv::Mat& frame,
    const autonomy::LaneObservation& observation,
    const autonomy::AutonomyMode mode,
    const double steering_rad,
    const bool paused
) {
    cv::rectangle(
        frame,
        cv::Rect(0, 0, frame.cols, 115),
        cv::Scalar(0, 0, 0),
        cv::FILLED
    );

    const cv::Scalar mode_colour =
        mode == autonomy::AutonomyMode::Normal
            ? cv::Scalar(0, 220, 0)
            : mode == autonomy::AutonomyMode::Degraded
                ? cv::Scalar(0, 200, 255)
                : cv::Scalar(0, 0, 255);

    cv::putText(
        frame,
        "Mode: " + autonomy::toString(mode),
        {12, 25},
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        mode_colour,
        2,
        cv::LINE_AA
    );

    cv::putText(
        frame,
        "Confidence: " +
            formatValue(observation.confidence) +
            "  lateral: " +
            formatValue(
                observation.lateral_error_m
            ) +
            " m",
        {12, 52},
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
    );

    cv::putText(
        frame,
        "Heading: " +
            formatValue(
                observation.heading_error_rad
            ) +
            " rad  steering: " +
            formatValue(steering_rad) +
            " rad",
        {12, 78},
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
    );

    cv::putText(
        frame,
        paused
            ? "PAUSED - Space/p: resume, q/Esc: quit"
            : "Space/p: pause, q/Esc: quit",
        {12, 103},
        cv::FONT_HERSHEY_SIMPLEX,
        0.50,
        paused
            ? cv::Scalar(0, 200, 255)
            : cv::Scalar(180, 180, 180),
        1,
        cv::LINE_AA
    );
}

/*
 * Wait while paused.
 *
 * Returns true when the user requests exit or closes the
 * main window. Returns false when playback should resume.
 */
bool waitWhilePaused(
    cv::Mat& debug_frame,
    const autonomy::LaneObservation& observation,
    const autonomy::AutonomyMode mode,
    const double steering_rad
) {
    drawStatus(
        debug_frame,
        observation,
        mode,
        steering_rad,
        true
    );

    cv::imshow(
        kMainWindowName,
        debug_frame
    );

    std::cout
        << "Paused. Press Space or p to resume; "
        << "q or Escape to quit.\n";

    while (true) {
        const int key = cv::waitKey(30);

        if (
            key == 27 ||
            key == 'q' ||
            isMainWindowClosed()
        ) {
            return true;
        }

        if (key == ' ' || key == 'p') {
            return false;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr
            << "Usage: run_video "
            << "<video-file> [assumed-speed-mps]\n";

        return 1;
    }

    const std::string video_path = argv[1];

    const double assumed_speed_mps =
        argc == 3
            ? std::stod(argv[2])
            : 8.0;

    if (assumed_speed_mps < 0.0) {
        throw std::invalid_argument(
            "Assumed speed must not be negative"
        );
    }

    cv::VideoCapture video(video_path);

    if (!video.isOpened()) {
        std::cerr
            << "Could not open video: "
            << video_path
            << '\n';

        return 2;
    }

    double fps = video.get(cv::CAP_PROP_FPS);

    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 30.0;
    }

    const auto frame_period =
        std::chrono::duration<double>(
            1.0 / fps
        );

    autonomy::LaneDetector lane_detector(
        75.0,
        300,
        autonomy::LaneDetectorProfile::RecordedVideo
    );
    autonomy::LaneTracker lane_tracker;
    autonomy::HealthMonitor health_monitor;
    autonomy::StanleyController controller;

    std::size_t frame_index = 0;
    std::size_t valid_observations = 0;

    cv::Mat source_frame;
    cv::Mat last_debug_frame;

    autonomy::LaneObservation last_observation;
    autonomy::AutonomyMode last_mode =
        autonomy::AutonomyMode::Normal;

    double last_steering_rad = 0.0;

    bool user_requested_exit = false;

    while (video.read(source_frame)) {
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<
                std::chrono::steady_clock::duration
            >(frame_period);

        cv::Mat frame;

        cv::resize(
            source_frame,
            frame,
            cv::Size(
                kFrameWidth,
                kFrameHeight
            ),
            0.0,
            0.0,
            cv::INTER_AREA
        );

        cv::Mat debug_frame;

        const autonomy::LaneObservation raw_observation =
            lane_detector.detect(
                frame,
                &debug_frame
            );

        const autonomy::LaneObservation observation =
            lane_tracker.update(
                raw_observation
            );

        if (observation.valid) {
            ++valid_observations;
        }

        const autonomy::AutonomyMode mode =
            health_monitor.update(
                observation
            );

        const double target_speed_mps =
            mode == autonomy::AutonomyMode::SafeStop
                ? 0.0
                : mode == autonomy::AutonomyMode::Degraded
                    ? std::min(
                        assumed_speed_mps,
                        4.0
                    )
                    : assumed_speed_mps;

        const autonomy::ControlCommand command =
            controller.calculate(
                observation,
                assumed_speed_mps,
                target_speed_mps
            );

        drawStatus(
            debug_frame,
            observation,
            mode,
            command.requested_steering_rad,
            false
        );

        cv::imshow(
            kMainWindowName,
            debug_frame
        );

        last_debug_frame = debug_frame.clone();
        last_observation = observation;
        last_mode = mode;
        last_steering_rad =
            command.requested_steering_rad;

        ++frame_index;

        const int key = cv::waitKey(1);

        if (key == ' ' || key == 'p') {
            user_requested_exit =
                waitWhilePaused(
                    last_debug_frame,
                    last_observation,
                    last_mode,
                    last_steering_rad
                );

            if (user_requested_exit) {
                break;
            }
        }

        if (
            key == 27 ||
            key == 'q' ||
            isMainWindowClosed()
        ) {
            user_requested_exit = true;
            break;
        }

        std::this_thread::sleep_until(deadline);
    }

    /*
     * Keep the final processed frame visible after EOF.
     */
    if (
        !user_requested_exit &&
        !last_debug_frame.empty()
    ) {
        drawStatus(
            last_debug_frame,
            last_observation,
            last_mode,
            last_steering_rad,
            true
        );

        cv::imshow(
            kMainWindowName,
            last_debug_frame
        );

        while (!isMainWindowClosed()) {
            const int key = cv::waitKey(30);

            if (key == 27 || key == 'q') {
                break;
            }
        }
    }

    const double valid_percentage =
        frame_index == 0
            ? 0.0
            : 100.0 *
                static_cast<double>(
                    valid_observations
                ) /
                static_cast<double>(
                    frame_index
                );

    cv::destroyAllWindows();

    return frame_index == 0 ? 3 : 0;
}