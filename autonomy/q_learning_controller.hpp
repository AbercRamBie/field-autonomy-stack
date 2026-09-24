#pragma once

#include "autonomy/types.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace autonomy {

class QLearningController {
public:
    static constexpr std::size_t kLateralBins = 11;
    static constexpr std::size_t kHeadingBins = 11;
    static constexpr std::size_t kPreviousSteeringBins = 5;

    static constexpr std::size_t kStateCount =
        kLateralBins *
        kHeadingBins *
        kPreviousSteeringBins;

    static constexpr std::size_t kActionCount = 3;

    explicit QLearningController(
        double learning_rate = 0.10,
        double discount_factor = 0.98,
        double epsilon = 1.0,
        unsigned int seed = 42
    );

    [[nodiscard]] std::size_t stateIndex(
        const LaneObservation& observation,
        double previous_steering_rad
    ) const;

    [[nodiscard]] std::size_t selectAction(
        std::size_t state_index,
        bool allow_exploration
    );

    [[nodiscard]] double residualForAction(
        std::size_t action_index
    ) const;

    void update(
        std::size_t state_index,
        std::size_t action_index,
        double reward,
        std::size_t next_state_index,
        bool terminated
    );

    void setEpsilon(double epsilon);

    void decayEpsilon(
        double decay_factor,
        double minimum_epsilon
    );

    [[nodiscard]] double epsilon() const noexcept;

    [[nodiscard]] double qValue(
        std::size_t state_index,
        std::size_t action_index
    ) const;

    void save(const std::string& output_path) const;
    void load(const std::string& input_path);

private:
    [[nodiscard]] std::size_t greedyAction(
        std::size_t state_index
    ) const;

    [[nodiscard]] double maximumQ(
        std::size_t state_index
    ) const;

    [[nodiscard]] std::size_t tableIndex(
        std::size_t state_index,
        std::size_t action_index
    ) const;

    double learning_rate_;
    double discount_factor_;
    double epsilon_;

    std::vector<double> q_table_;

    std::mt19937 random_engine_;
    std::uniform_real_distribution<double> unit_distribution_{
        0.0,
        1.0
    };
};

}  // namespace autonomy