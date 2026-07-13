#include "autonomy/vehicle_model.hpp"

#include <gtest/gtest.h>

TEST(VehicleModelTest, ZeroSpeedDoesNotChangePoseErrors) {
    autonomy::VehicleModel model;

    autonomy::VehicleState state;
    state.speed_mps = 0.0;
    state.lateral_error_m = 0.5;
    state.heading_error_rad = 0.1;

    model.reset(state);
    model.step(0.2, 0.0, 0.01, 0.05);

    EXPECT_NEAR(
        model.state().lateral_error_m,
        0.5,
        1e-9
    );

    EXPECT_NEAR(
        model.state().heading_error_rad,
        0.1,
        1e-9
    );
}

TEST(VehicleModelTest, RejectsInvalidTimestep) {
    autonomy::VehicleModel model;

    EXPECT_THROW(
        model.step(0.0, 0.0, 0.0, 0.0),
        std::invalid_argument
    );
}