// Placeholder
#include "autonomy/health_monitor.hpp"

#include <iostream>

namespace autonomy {

AutonomyMode HealthMonitor::update(
    const LaneObservation& observation
) {
    const bool healthy =
        observation.valid &&
        observation.confidence >= 0.70;

    const bool failed =
        !observation.valid ||
        observation.confidence < 0.35;

    if (healthy) {
        ++recovery_frames_;
        low_confidence_frames_ = 0;
    } else {
        recovery_frames_ = 0;
        ++low_confidence_frames_;
    }

    const AutonomyMode previous_mode = mode_;

    if (mode_ == AutonomyMode::Normal) {
        if (low_confidence_frames_ >= degraded_threshold_frames_) {
            mode_ = AutonomyMode::Degraded;
        }
    } else if (mode_ == AutonomyMode::Degraded) {
        if (failed &&
            low_confidence_frames_ >= safe_stop_threshold_frames_) {
            mode_ = AutonomyMode::SafeStop;
        } else if (
            recovery_frames_ >= recovery_threshold_frames_
        ) {
            mode_ = AutonomyMode::Normal;
        }
    } else if (mode_ == AutonomyMode::SafeStop) {
        if (recovery_frames_ >= recovery_threshold_frames_) {
            mode_ = AutonomyMode::Degraded;
        }
    }

    if (mode_ != previous_mode) {
        std::cout
            << "[HEALTH] "
            << toString(previous_mode)
            << " -> "
            << toString(mode_)
            << '\n';
    }

    return mode_;
}

AutonomyMode HealthMonitor::mode() const noexcept {
    return mode_;
}

}  // namespace autonomy