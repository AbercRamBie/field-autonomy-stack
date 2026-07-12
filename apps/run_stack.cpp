#include "autonomy/stanley_controller.hpp"
#include "autonomy/track.hpp"
#include "autonomy/vehicle_model.hpp"

#include <iostream>

int main() {
    autonomy::Track track;
    autonomy::VehicleModel vehicle;
    autonomy::StanleyController controller;

    autonomy::VehicleState initial_state;
    initial_state.lateral_error_m = 0.5;
    initial_state.heading_error_rad = 0.05;
    initial_state.speed_mps = 8.0;

    vehicle.reset(initial_state);

    constexpr double dt_s = 0.05;

    for (int frame = 0; frame < 600; ++frame) {
        const auto& state = vehicle.state();

        const double curvature =
            track.curvatureAt(state.distance_m);

        // For now, use the simulator's true state as the lane observation.
        // Later, this will be replaced by the OpenCV lane detector output.
        autonomy::LaneObservation observation;
        observation.valid = true;
        observation.confidence = 1.0;
        observation.lateral_error_m =
            state.lateral_error_m;
        observation.heading_error_rad =
            state.heading_error_rad;

        const auto command = controller.calculate(
            observation,
            state.speed_mps,
            8.0
        );

        vehicle.step(
            command.requested_steering_rad,
            command.target_speed_mps,
            curvature,
            dt_s
        );

        const auto& updated_state = vehicle.state();

        std::cout
            << frame << ','
            << updated_state.lateral_error_m << ','
            << updated_state.heading_error_rad << ','
            << command.requested_steering_rad
            << '\n';
    }

    return 0;
}