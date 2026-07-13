// Placeholder
#include "autonomy/csv_logger.hpp"

#include <stdexcept>

namespace autonomy {

CsvLogger::CsvLogger(const std::string& output_path)
    : output_(output_path) {
    if (!output_.is_open()) {
        throw std::runtime_error(
            "Could not open telemetry file: " + output_path
        );
    }

    output_
        << "frame,"
        << "time_s,"
        << "distance_m,"
        << "true_lateral_error_m,"
        << "true_heading_error_rad,"
        << "measured_lateral_error_m,"
        << "measured_heading_error_rad,"
        << "confidence,"
        << "speed_mps,"
        << "requested_steering_rad,"
        << "applied_steering_rad,"
        << "target_speed_mps,"
        << "curvature,"
        << "mode,"
        << "frame_dropped,"
        << "lane_occluded,"
        << "camera_noisy,"
        << "steering_delayed,"
        << "processing_time_ms\n";
}

void CsvLogger::write(const TelemetryRecord& record) {
    output_
        << record.frame_index << ','
        << record.simulation_time_s << ','
        << record.true_state.distance_m << ','
        << record.true_state.lateral_error_m << ','
        << record.true_state.heading_error_rad << ','
        << record.observation.lateral_error_m << ','
        << record.observation.heading_error_rad << ','
        << record.observation.confidence << ','
        << record.true_state.speed_mps << ','
        << record.control.requested_steering_rad << ','
        << record.control.applied_steering_rad << ','
        << record.control.target_speed_mps << ','
        << record.road_curvature << ','
        << toString(record.mode) << ','
        << record.faults.frame_dropped << ','
        << record.faults.lane_occluded << ','
        << record.faults.camera_noisy << ','
        << record.faults.steering_delayed << ','
        << record.observation.processing_time_ms
        << '\n';

    if (record.frame_index % 20 == 0) {
        output_.flush();
    }
}

}  // namespace autonomy