#include "autonomy/lane_detector.hpp"
#include "autonomy/lane_tracker.hpp"
#include "autonomy/types.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr int kFrameWidth = 640;
constexpr int kFrameHeight = 480;

void writeHeader(std::ofstream& output) {
  output 
        << "frame_index,"
        << "video_time_s,"
        << "raw_valid,"
        << "tracked_valid,"
        << "tracked_fresh,"
        << "left_detected,"
        << "right_detected,"
        << "lateral_error_m,"
        << "heading_error_rad,"
        << "confidence,"
        << "processing_time_ms,"
        << "left_a,"
        << "left_b,"
        << "left_c,"
        << "right_a,"
        << "right_b,"
        << "right_c\n";
}

void writeObservation(
    std::ofstream& output,
    const std::size_t frame_index,
    const double video_time_s,
    const autonomy::LaneObservation& raw,
    const autonomy::LaneObservation& tracked
) {
    output
        << frame_index << ','
        << video_time_s << ','
        << static_cast<int>(raw.valid) << ','
        << static_cast<int>(tracked.valid) << ','
        << static_cast<int>(tracked.fresh) << ','
        << static_cast<int>(tracked.left_detected) << ','
        << static_cast<int>(tracked.right_detected) << ','
        << tracked.lateral_error_m << ','
        << tracked.heading_error_rad << ','
        << tracked.confidence << ','
        << tracked.processing_time_ms << ','
        << tracked.left_curve.a << ','
        << tracked.left_curve.b << ','
        << tracked.left_curve.c << ','
        << tracked.right_curve.a << ','
        << tracked.right_curve.b << ','
        << tracked.right_curve.c
        << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "Usage: extract_real_observations "
            << "<video-file> <output-csv>\n";
        return 1;
    }

    const std::string video_path = argv[1];
    const std::string output_path = argv[2];

    cv::VideoCapture video(video_path);

    if (!video.isOpened()) {
        std::cerr
            << "Could not open video: "
            << video_path
            << '\n';
        return 2;
    }

    std::ofstream output(output_path);

    if (!output.is_open()) {
        std::cerr
            << "Could not open output CSV: "
            << output_path
            << '\n';
        return 3;
    }

    double fps = video.get(cv::CAP_PROP_FPS);

    if (!std::isfinite(fps) || fps <= 0.0) {
        fps = 20.0;
    }

    output << std::setprecision(17);
    writeHeader(output);

    autonomy::LaneDetector detector(
        75.0,
        300,
        autonomy::LaneDetectorProfile::RecordedVideo
    );
    autonomy::LaneTracker tracker;

    std::size_t frame_index = 0;
    std::size_t raw_valid_count = 0;
    std::size_t tracked_valid_count = 0;

    cv::Mat source_frame;

    while (video.read(source_frame)) {
        cv::Mat frame;

        cv::resize(
            source_frame,
            frame,
            cv::Size(kFrameWidth, kFrameHeight),
            0.0,
            0.0,
            cv::INTER_AREA
        );

        const autonomy::LaneObservation raw =
            detector.detect(frame, nullptr);

        const autonomy::LaneObservation tracked =
            tracker.update(raw);

        if (raw.valid) {
            ++raw_valid_count;
        }

        if (tracked.valid) {
            ++tracked_valid_count;
        }

        const double video_time_s =
            static_cast<double>(frame_index) / fps;

        writeObservation(
            output,
            frame_index,
            video_time_s,
            raw,
            tracked
        );

        ++frame_index;
    }

    std::cout
        << "Processed frames: " << frame_index << '\n'
        << "Raw valid observations: " << raw_valid_count << '\n'
        << "Tracked valid observations: "
        << tracked_valid_count << '\n'
        << "Video FPS: " << fps << '\n'
        << "Output: " << output_path << '\n';

    return frame_index == 0 ? 4 : 0;
}
