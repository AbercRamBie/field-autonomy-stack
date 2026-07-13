#include "autonomy/lane_detector.hpp"
#include "autonomy/scene_renderer.hpp"
#include "autonomy/types.hpp"

#include <gtest/gtest.h>

TEST(LaneDetectorTest, DetectsCleanSyntheticLane) {
    autonomy::SceneRenderer renderer;
    autonomy::LaneDetector detector;

    autonomy::VehicleState state;
    state.lateral_error_m = 0.4;
    state.heading_error_rad = 0.03;

    autonomy::FaultState faults;

    const auto frame =
        renderer.render(
            state,
            3.5,
            faults
        );

    const auto observation =
        detector.detect(frame);

    EXPECT_TRUE(observation.valid);
    EXPECT_TRUE(observation.left_detected);
    EXPECT_TRUE(observation.right_detected);
    EXPECT_GT(observation.confidence, 0.7);
}