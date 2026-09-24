#pragma once

#include "autonomy/types.hpp"
#include <cstddef>
#include <deque>
#include <random>

namespace autonomy {

class FaultInjector {
public:
    explicit FaultInjector(
        double dt_s,
        unsigned int seed = 42
    );

    [[nodiscard]] FaultState update(
        std::size_t frame_index,
        double simulation_time_s
    );

    [[nodiscard]] double applySteeringFault(
        double requested_steering_rad
    );

private:
    double dt_s_;
    FaultState current_faults_{};

    std::mt19937 random_engine_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};

    std::deque<double> steering_queue_;
};

}  // namespace autonomy