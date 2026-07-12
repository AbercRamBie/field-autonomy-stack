#include "autonomy/track.hpp"
#include "autonomy/vehicle_model.hpp"

#include <iostream>

int main() {
    autonomy::Track track;
    autonomy::VehicleModel vehicle;

    autonomy::VehicleState initial_state;
    initial_state.lateral_error_m = 0.5;
    initial_state.heading_error_rad = 0.05;
    initial_state.speed_mps = 8.0;

    vehicle.reset(initial_state);

    constexpr double dt_s = 0.05;

    for (int frame = 0; frame < 200; ++frame) {
        const auto& state = vehicle.state();
        const double curvature = track.curvatureAt(state.distance_m);

        vehicle.step(
            0.0,
            8.0,
            curvature,
            dt_s
        );

        std::cout
            << frame << ','
            << vehicle.state().lateral_error_m << ','
            << vehicle.state().heading_error_rad
            << '\n';
    }

    return 0;
}