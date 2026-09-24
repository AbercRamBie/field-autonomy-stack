#include "autonomy/q_learning_controller.hpp"
#include "autonomy/stanley_controller.hpp"
#include "autonomy/track.hpp"
#include "autonomy/types.hpp"
#include "autonomy/vehicle_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

constexpr double kDtS = 0.05;
constexpr double kTargetSpeedMps = 8.0;
constexpr double kLaneDepartureM = 1.75;
constexpr double kMaximumSteeringRad = 0.45;
constexpr std::size_t kEpisodeCount = 3000;
constexpr std::size_t kStepsPerEpisode = 400;

autonomy::LaneObservation makeObservation(const autonomy::VehicleState& state) 
{
    autonomy::LaneObservation observation;

    observation.valid = true;
    observation.fresh = true;
    observation.confidence = 1.0;
    observation.lateral_error_m =
        state.lateral_error_m;
    observation.heading_error_rad =
        state.heading_error_rad;

    return observation;
}

double calculateReward(const autonomy::VehicleState& next_state,const double steering_rad,const double previous_steering_rad) 
{
    const double normalized_lane_error =
        next_state.lateral_error_m /
        kLaneDepartureM;

    const double normalized_heading_error =
        next_state.heading_error_rad /
        0.35;

    const double normalized_steering_change =
        (steering_rad - previous_steering_rad) /
        kMaximumSteeringRad;

    return
        1.0
        - 2.0 *
            normalized_lane_error *
            normalized_lane_error
        - 0.5 *
            normalized_heading_error *
            normalized_heading_error
        - 0.05 *
            normalized_steering_change *
            normalized_steering_change;
}

}  // namespace

int main() {
    std::filesystem::create_directories("models");
    std::filesystem::create_directories("logs");

    std::ofstream training_log(
        "logs/q_learning_training.csv"
    );

    if (!training_log.is_open()) {
        throw std::runtime_error(
            "Could not open training log"
        );
    }

    training_log
        << std::setprecision(17)
        << "episode,"
        << "steps,"
        << "episode_return,"
        << "mean_absolute_lateral_error_m,"
        << "epsilon,"
        << "lane_departure\n";

    autonomy::Track track;
    autonomy::VehicleModel vehicle;
    autonomy::StanleyController stanley_controller;

    autonomy::QLearningController q_controller(
        0.10,
        0.98,
        1.0,
        42
    );

    std::mt19937 reset_engine(2026);

    std::uniform_real_distribution<double>
        distance_distribution(0.0, 150.0);

    std::uniform_real_distribution<double>
        lateral_distribution(-1.2, 1.2);

    std::uniform_real_distribution<double>
        heading_distribution(-0.15, 0.15);

    for (
        std::size_t episode = 0;
        episode < kEpisodeCount;
        ++episode
    ) {
        autonomy::VehicleState initial_state;

        initial_state.distance_m =
            distance_distribution(reset_engine);

        initial_state.lateral_error_m =
            lateral_distribution(reset_engine);

        initial_state.heading_error_rad =
            heading_distribution(reset_engine);

        initial_state.speed_mps =
            kTargetSpeedMps;

        initial_state.steering_rad = 0.0;

        vehicle.reset(initial_state);

        double previous_steering_rad = 0.0;
        double episode_return = 0.0;
        double absolute_error_sum = 0.0;

        std::size_t completed_steps = 0;
        bool lane_departure = false;

        for (
            std::size_t step = 0;
            step < kStepsPerEpisode;
            ++step
        ) {
            const autonomy::VehicleState current_state =
                vehicle.state();

            const autonomy::LaneObservation observation =
                makeObservation(current_state);

            const std::size_t state_index =
                q_controller.stateIndex(
                    observation,
                    previous_steering_rad
                );

            const autonomy::ControlCommand
                stanley_command =
                    stanley_controller.calculate(
                        observation,
                        current_state.speed_mps,
                        kTargetSpeedMps
                    );

            const std::size_t action_index =
                q_controller.selectAction(
                    state_index,
                    true
                );

            const double residual_steering_rad =
                q_controller.residualForAction(
                    action_index
                );

            const double requested_steering_rad =
                std::clamp(
                    stanley_command
                            .requested_steering_rad +
                        residual_steering_rad,
                    -kMaximumSteeringRad,
                    kMaximumSteeringRad
                );

            const double curvature =
                track.curvatureAt(
                    current_state.distance_m
                );

            vehicle.step(
                requested_steering_rad,
                kTargetSpeedMps,
                curvature,
                kDtS
            );

            const autonomy::VehicleState next_state =
                vehicle.state();

            const autonomy::LaneObservation
                next_observation =
                    makeObservation(next_state);

            const std::size_t next_state_index =
                q_controller.stateIndex(
                    next_observation,
                    requested_steering_rad
                );

            double reward = calculateReward(
                next_state,
                requested_steering_rad,
                previous_steering_rad
            );

            lane_departure =
                std::abs(
                    next_state.lateral_error_m
                ) > kLaneDepartureM;

            if (lane_departure) {
                reward -= 10.0;
            }

            q_controller.update(
                state_index,
                action_index,
                reward,
                next_state_index,
                lane_departure
            );

            previous_steering_rad =
                requested_steering_rad;

            episode_return += reward;

            absolute_error_sum += std::abs(
                next_state.lateral_error_m
            );

            completed_steps = step + 1;

            if (lane_departure) {
                break;
            }
        }

        q_controller.decayEpsilon(
            0.995,
            0.05
        );

        const double mean_absolute_error =
            completed_steps > 0
                ? absolute_error_sum /
                    static_cast<double>(
                        completed_steps
                    )
                : 0.0;

        training_log
            << episode << ','
            << completed_steps << ','
            << episode_return << ','
            << mean_absolute_error << ','
            << q_controller.epsilon() << ','
            << static_cast<int>(lane_departure)
            << '\n';
    }

    q_controller.save("models/q_table.txt");

    return 0;
}