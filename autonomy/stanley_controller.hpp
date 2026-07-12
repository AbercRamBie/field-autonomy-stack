#pragma once

#include "autonomy/types.hpp"

namespace autonomy {

class StanleyController {
public:
    StanleyController(
        double lateral_gain = 1.2,
        double speed_softening = 1.0,
        double max_steering_rad = 0.45
    );

    [[nodiscard]] ControlCommand calculate(
        const LaneObservation& observation,
        double speed_mps,
        double target_speed_mps
    ) const;

private:
    double lateral_gain_;
    double speed_softening_;
    double max_steering_rad_;
};

}  // namespace autonomy