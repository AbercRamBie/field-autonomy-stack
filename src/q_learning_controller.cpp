#include "autonomy/q_learning_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace autonomy {
namespace {

constexpr std::array<double, 10> kLateralEdges{
    -1.25, -0.80, -0.50, -0.25, -0.10,
     0.10,  0.25,  0.50,  0.80,  1.25
};

constexpr std::array<double, 10> kHeadingEdges{
    -0.25, -0.15, -0.08, -0.04, -0.015,
     0.015, 0.04, 0.08, 0.15, 0.25
};

constexpr std::array<double, 4> kPreviousSteeringEdges{
    -0.20, -0.05, 0.05, 0.20
};

constexpr std::array<double, 3> kResidualActions{
    -0.06,
     0.00,
     0.06
};

template <std::size_t EdgeCount>
std::size_t findBin(
    const double value,
    const std::array<double, EdgeCount>& edges
) {
    const auto iterator = std::upper_bound(
        edges.begin(),
        edges.end(),
        value
    );

    return static_cast<std::size_t>(
        std::distance(edges.begin(), iterator)
    );
}

void validateProbability(
    const double value,
    const std::string& name
) {
    if (
        !std::isfinite(value) ||
        value < 0.0 ||
        value > 1.0
    ) {
        throw std::invalid_argument(
            name + " must be between 0 and 1"
        );
    }
}

}  // namespace

QLearningController::QLearningController(
    const double learning_rate,
    const double discount_factor,
    const double epsilon,
    const unsigned int seed
)
    : learning_rate_(learning_rate),
      discount_factor_(discount_factor),
      epsilon_(epsilon),
      q_table_(kStateCount * kActionCount, 0.0),
      random_engine_(seed) {
    if (
        !std::isfinite(learning_rate_) ||
        learning_rate_ <= 0.0 ||
        learning_rate_ > 1.0
    ) {
        throw std::invalid_argument(
            "Learning rate must be in (0, 1]"
        );
    }

    validateProbability(
        discount_factor_,
        "Discount factor"
    );

    validateProbability(
        epsilon_,
        "Epsilon"
    );
}

std::size_t QLearningController::stateIndex(
    const LaneObservation& observation,
    const double previous_steering_rad
) const {
    if (
        !std::isfinite(observation.lateral_error_m) ||
        !std::isfinite(observation.heading_error_rad) ||
        !std::isfinite(previous_steering_rad)
    ) {
        throw std::invalid_argument(
            "State values must be finite"
        );
    }

    const std::size_t lateral_bin = findBin(
        observation.lateral_error_m,
        kLateralEdges
    );

    const std::size_t heading_bin = findBin(
        observation.heading_error_rad,
        kHeadingEdges
    );

    const std::size_t previous_steering_bin = findBin(
        previous_steering_rad,
        kPreviousSteeringEdges
    );

    return (
        lateral_bin * kHeadingBins +
        heading_bin
    ) * kPreviousSteeringBins +
        previous_steering_bin;
}

std::size_t QLearningController::selectAction(
    const std::size_t state_index,
    const bool allow_exploration
) {
    if (state_index >= kStateCount) {
        throw std::out_of_range(
            "Q-learning state index is out of range"
        );
    }

    const bool explore =
        allow_exploration &&
        unit_distribution_(random_engine_) < epsilon_;

    if (explore) {
        std::uniform_int_distribution<std::size_t>
            action_distribution(
                0,
                kActionCount - 1
            );

        return action_distribution(random_engine_);
    }

    return greedyAction(state_index);
}

double QLearningController::residualForAction(
    const std::size_t action_index
) const {
    if (action_index >= kActionCount) {
        throw std::out_of_range(
            "Q-learning action index is out of range"
        );
    }

    return kResidualActions[action_index];
}

void QLearningController::update(
    const std::size_t state_index,
    const std::size_t action_index,
    const double reward,
    const std::size_t next_state_index,
    const bool terminated
) {
    if (!std::isfinite(reward)) {
        throw std::invalid_argument(
            "Reward must be finite"
        );
    }

    const std::size_t current_table_index =
        tableIndex(state_index, action_index);

    if (next_state_index >= kStateCount) {
        throw std::out_of_range(
            "Next state index is out of range"
        );
    }

    double target = reward;

    if (!terminated) {
        target +=
            discount_factor_ *
            maximumQ(next_state_index);
    }

    double& current_q = q_table_[current_table_index];

    current_q +=
        learning_rate_ *
        (target - current_q);
}

void QLearningController::setEpsilon(
    const double epsilon
) {
    validateProbability(epsilon, "Epsilon");
    epsilon_ = epsilon;
}

void QLearningController::decayEpsilon(
    const double decay_factor,
    const double minimum_epsilon
) {
    validateProbability(
        decay_factor,
        "Epsilon decay factor"
    );

    validateProbability(
        minimum_epsilon,
        "Minimum epsilon"
    );

    epsilon_ = std::max(
        minimum_epsilon,
        epsilon_ * decay_factor
    );
}

double QLearningController::epsilon() const noexcept {
    return epsilon_;
}

double QLearningController::qValue(
    const std::size_t state_index,
    const std::size_t action_index
) const {
    return q_table_[
        tableIndex(state_index, action_index)
    ];
}

std::size_t QLearningController::greedyAction(
    const std::size_t state_index
) const {
    // Prefer the neutral residual when Q-values are tied.
    std::size_t best_action = 1;
    double best_value = qValue(state_index, best_action);

    for (
        std::size_t action_index = 0;
        action_index < kActionCount;
        ++action_index
    ) {
        const double candidate =
            qValue(state_index, action_index);

        if (candidate > best_value) {
            best_value = candidate;
            best_action = action_index;
        }
    }

    return best_action;
}

double QLearningController::maximumQ(
    const std::size_t state_index
) const {
    const std::size_t first =
        tableIndex(state_index, 0);

    const auto begin = q_table_.begin() +
        static_cast<std::ptrdiff_t>(first);

    const auto end = begin +
        static_cast<std::ptrdiff_t>(kActionCount);

    return *std::max_element(begin, end);
}

std::size_t QLearningController::tableIndex(
    const std::size_t state_index,
    const std::size_t action_index
) const {
    if (state_index >= kStateCount) {
        throw std::out_of_range(
            "Q-learning state index is out of range"
        );
    }

    if (action_index >= kActionCount) {
        throw std::out_of_range(
            "Q-learning action index is out of range"
        );
    }

    return state_index * kActionCount + action_index;
}

void QLearningController::save(
    const std::string& output_path
) const {
    std::ofstream output(output_path);

    if (!output.is_open()) {
        throw std::runtime_error(
            "Could not open Q-table output: " +
            output_path
        );
    }

    output
        << "FIELD_AUTONOMY_Q_TABLE_V1\n"
        << kStateCount << ' '
        << kActionCount << '\n'
        << std::setprecision(17);

    for (const double value : q_table_) {
        output << value << '\n';
    }
}

void QLearningController::load(
    const std::string& input_path
) {
    std::ifstream input(input_path);

    if (!input.is_open()) {
        throw std::runtime_error(
            "Could not open Q-table input: " +
            input_path
        );
    }

    std::string format;
    std::getline(input, format);

    if (format != "FIELD_AUTONOMY_Q_TABLE_V1") {
        throw std::runtime_error(
            "Unsupported Q-table format"
        );
    }

    std::size_t state_count = 0;
    std::size_t action_count = 0;

    input >> state_count >> action_count;

    if (
        state_count != kStateCount ||
        action_count != kActionCount
    ) {
        throw std::runtime_error(
            "Q-table dimensions do not match"
        );
    }

    std::vector<double> loaded_table(
        kStateCount * kActionCount,
        0.0
    );

    for (double& value : loaded_table) {
        if (!(input >> value) || !std::isfinite(value)) {
            throw std::runtime_error(
                "Q-table contains invalid data"
            );
        }
    }

    q_table_ = std::move(loaded_table);
}

}  // namespace autonomy