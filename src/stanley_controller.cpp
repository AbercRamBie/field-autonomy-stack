// Placeholder
#include "autonomy/stanley_controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace autonomy {

StanleyController::StanleyController(
    const double lateral_gain,
    const double speed_softening,
    const double max_steering_rad
)
    : lateral_gain_(lateral_gain),
      speed_softening_(speed_softening),
      max_steering_rad_(max_steering_rad) {
    if (lateral_gain_ <= 0.0 ||
        speed_softening_ <= 0.0 ||
        max_steering_rad_ <= 0.0) {
        throw std::invalid_argument(
            "Controller parameters must be positive"
        );
    }
}

ControlCommand StanleyController::calculate(
    const LaneObservation& observation,
    const double speed_mps,
    const double target_speed_mps
) const {
    ControlCommand command;
    command.target_speed_mps = target_speed_mps;

    if (!observation.valid) {
        command.requested_steering_rad = 0.0;
        return command;
    }

    const double lateral_correction = std::atan2(
        lateral_gain_ * observation.lateral_error_m,
        speed_mps + speed_softening_
    );

    const double steering =
        -(observation.heading_error_rad + lateral_correction);

    command.requested_steering_rad = std::clamp(
        steering,
        -max_steering_rad_,
        max_steering_rad_
    );

    return command;
}

}  // namespace autonomy