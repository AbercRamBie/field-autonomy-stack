#pragma once

#include <cstddef>
#include <string>

namespace autonomy {

constexpr double kPi = 3.14159265358979323846;

struct VehicleState {
    double distance_m{0.0};
    double lateral_error_m{0.0};
    double heading_error_rad{0.0};
    double speed_mps{8.0};
    double steering_rad{0.0};
};

struct LanePolynomial {
    double a{0.0};
    double b{0.0};
    double c{0.0};
};

struct LaneObservation {
    bool valid{false};
    bool left_detected{false};
    bool right_detected{false};

    // True only when LaneTracker accepts the current frame.
    // False for held or invalid tracker observations.
    bool fresh{false};

    double lateral_error_m{0.0};
    double heading_error_rad{0.0};
    double confidence{0.0};

    double processing_time_ms{0.0};

    // Camera-space points.
    double left_bottom_x{0.0};
    double right_bottom_x{0.0};
    double left_lookahead_x{0.0};
    double right_lookahead_x{0.0};

    // Bird's-eye polynomial coefficients.
    LanePolynomial left_curve{};
    LanePolynomial right_curve{};
};

struct ControlCommand {
    double requested_steering_rad{0.0};
    double applied_steering_rad{0.0};
    double target_speed_mps{0.0};
};

enum class AutonomyMode {
    Normal,
    Degraded,
    SafeStop
};

inline std::string toString(const AutonomyMode mode) {
    switch (mode) {
        case AutonomyMode::Normal:
            return "NORMAL";
        case AutonomyMode::Degraded:
            return "DEGRADED";
        case AutonomyMode::SafeStop:
            return "SAFE_STOP";
    }

    return "UNKNOWN";
}

struct FaultState {
    bool frame_dropped{false};
    bool lane_occluded{false};
    bool camera_noisy{false};
    bool steering_delayed{false};
};

struct TelemetryRecord {
    std::size_t frame_index{0};
    double simulation_time_s{0.0};

    VehicleState true_state{};
    LaneObservation observation{};
    ControlCommand control{};

    double road_curvature{0.0};
    AutonomyMode mode{AutonomyMode::Normal};
    FaultState faults{};
};

}  // namespace autonomy
