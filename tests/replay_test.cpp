#include "autonomy/stanley_controller.hpp"

#include <gtest/gtest.h>

TEST(ReplayTest, SameInputProducesSameOutput) {
    autonomy::StanleyController controller;

    autonomy::LaneObservation observation;
    observation.valid = true;
    observation.lateral_error_m = 0.3;
    observation.heading_error_rad = 0.04;
    observation.confidence = 1.0;

    const auto first =
        controller.calculate(
            observation,
            8.0,
            8.0
        );

    const auto second =
        controller.calculate(
            observation,
            8.0,
            8.0
        );

    EXPECT_DOUBLE_EQ(
        first.requested_steering_rad,
        second.requested_steering_rad
    );
}