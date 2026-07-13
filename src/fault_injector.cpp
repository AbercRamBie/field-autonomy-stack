// Placeholder
#include "autonomy/fault_injector.hpp"

#include <algorithm>
#include <cmath>

namespace autonomy {

FaultInjector::FaultInjector(
    const double dt_s,
    const unsigned int seed
)
    : dt_s_(dt_s),
      random_engine_(seed) {}

FaultState FaultInjector::update(
    const std::size_t,
    const double simulation_time_s
) {
    current_faults_ = FaultState{};

    if (simulation_time_s >= 8.0 &&
        simulation_time_s < 11.0) {
        current_faults_.camera_noisy = true;
    }

    if (simulation_time_s >= 14.0 &&
        simulation_time_s < 16.0) {
        current_faults_.frame_dropped =
            uniform_(random_engine_) < 0.35;
    }

    if (simulation_time_s >= 19.0 &&
        simulation_time_s < 23.0) {
        current_faults_.lane_occluded = true;
    }

    if (simulation_time_s >= 26.0 &&
        simulation_time_s < 29.0) {
        current_faults_.steering_delayed = true;
    }

    return current_faults_;
}

double FaultInjector::applySteeringFault(
    const double requested_steering_rad
) {
    steering_queue_.push_back(requested_steering_rad);

    if (!current_faults_.steering_delayed) {
        steering_queue_.clear();
        return requested_steering_rad;
    }

    const int delay_frames = std::max(
        1,
        static_cast<int>(std::round(0.20 / dt_s_))
    );

    while (
        static_cast<int>(steering_queue_.size()) >
        delay_frames + 1
    ) {
        steering_queue_.pop_front();
    }

    if (
        static_cast<int>(steering_queue_.size()) <=
        delay_frames
    ) {
        return 0.0;
    }

    return steering_queue_.front();
}

}  // namespace autonomy