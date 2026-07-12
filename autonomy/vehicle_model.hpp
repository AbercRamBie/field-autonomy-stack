#pragma once

#include "autonomy/types.hpp"

namespace autonomy {

class VehicleModel {
public:
    explicit VehicleModel(double wheelbase_m = 2.7);

    void reset(const VehicleState& state);

    void step(
        double steering_rad,
        double target_speed_mps,
        double road_curvature,
        double dt_s
    );

    [[nodiscard]] const VehicleState& state() const noexcept;

private:
    VehicleState state_{};
    double wheelbase_m_{2.7};
    double max_steering_rad_{0.45};
    double max_acceleration_mps2_{2.0};
};

}  // namespace autonomy