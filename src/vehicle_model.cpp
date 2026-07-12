// Placeholder
#include "autonomy/vehicle_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace autonomy {

VehicleModel::VehicleModel(const double wheelbase_m)
    : wheelbase_m_(wheelbase_m) {
    if (wheelbase_m_ <= 0.0) {
        throw std::invalid_argument("Wheelbase must be positive");
    }
}

void VehicleModel::reset(const VehicleState& state) {
    state_ = state;
}

void VehicleModel::step(
    const double steering_rad,
    const double target_speed_mps,
    const double road_curvature,
    const double dt_s
) {
    if (dt_s <= 0.0) {
        throw std::invalid_argument("Timestep must be positive");
    }

    const double steering = std::clamp(
        steering_rad,
        -max_steering_rad_,
        max_steering_rad_
    );

    const double speed_error = target_speed_mps - state_.speed_mps;
    const double requested_acceleration = speed_error / dt_s;

    const double acceleration = std::clamp(
        requested_acceleration,
        -max_acceleration_mps2_,
        max_acceleration_mps2_
    );

    state_.speed_mps = std::max(
        0.0,
        state_.speed_mps + acceleration * dt_s
    );

    const double lateral_rate =
        state_.speed_mps * std::sin(state_.heading_error_rad);

    const double heading_rate =
        (state_.speed_mps / wheelbase_m_) * std::tan(steering) -
        state_.speed_mps * road_curvature;

    state_.lateral_error_m += lateral_rate * dt_s;
    state_.heading_error_rad += heading_rate * dt_s;
    state_.distance_m += state_.speed_mps * dt_s;
    state_.steering_rad = steering;
}

const VehicleState& VehicleModel::state() const noexcept {
    return state_;
}

}  // namespace autonomy