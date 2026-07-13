#pragma once

#include "autonomy/types.hpp"

namespace autonomy {

class HealthMonitor {
public:
    [[nodiscard]] AutonomyMode update(
        const LaneObservation& observation
    );

    [[nodiscard]] AutonomyMode mode() const noexcept;

private:
    AutonomyMode mode_{AutonomyMode::Normal};

    int low_confidence_frames_{0};
    int recovery_frames_{0};

    int degraded_threshold_frames_{4};
    int safe_stop_threshold_frames_{12};
    int recovery_threshold_frames_{8};
};

}  // namespace autonomy