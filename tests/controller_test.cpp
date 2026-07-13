#include "autonomy/stanley_controller.hpp"

#include <gtest/gtest.h>

#include <cmath>

TEST(StanleyControllerTest, CorrectsPositiveLateralError) {
    autonomy::StanleyController controller;

    autonomy::LaneObservation observation;
    observation.valid = true;
    observation.lateral_error_m = 0.5;
    observation.heading_error_rad = 0.0;

    const auto command =
        controller.calculate(
            observation,
            8.0,
            8.0
        );

    EXPECT_LT(
        command.requested_steering_rad,
        0.0
    );
}

TEST(StanleyControllerTest, RespectsSteeringLimit) {
    autonomy::StanleyController controller;

    autonomy::LaneObservation observation;
    observation.valid = true;
    observation.lateral_error_m = 100.0;
    observation.heading_error_rad = 2.0;

    const auto command =
        controller.calculate(
            observation,
            1.0,
            8.0
        );

    EXPECT_LE(
        std::abs(command.requested_steering_rad),
        0.45
    );
}

TEST(StanleyControllerTest, InvalidObservationReturnsNeutralSteering) {
    autonomy::StanleyController controller;

    autonomy::LaneObservation observation;
    observation.valid = false;

    const auto command =
        controller.calculate(
            observation,
            8.0,
            4.0
        );

    EXPECT_DOUBLE_EQ(
        command.requested_steering_rad,
        0.0
    );
}