#include "autonomy/stanley_controller.hpp"
#include "autonomy/types.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(
    const std::string& line,
    const char delimiter
) {
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string item;

    while (std::getline(stream, item, delimiter)) {
        values.push_back(item);
    }

    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: replay_controller <telemetry.csv>\n";
        return 1;
    }

    std::ifstream input(argv[1]);

    if (!input.is_open()) {
        throw std::runtime_error(
            "Could not open telemetry file"
        );
    }

    autonomy::StanleyController controller;

    std::string line;
    std::getline(input, line);

    std::size_t replayed_frames = 0;
    std::size_t matched_commands = 0;
    double maximum_difference = 0.0;

    constexpr double tolerance = 1e-9;

    while (std::getline(input, line)) {
        const auto columns = split(line, ',');

        if (columns.size() < 19) {
            continue;
        }

        autonomy::LaneObservation observation;
        observation.lateral_error_m =
            std::stod(columns[5]);
        observation.heading_error_rad =
            std::stod(columns[6]);
        observation.confidence =
            std::stod(columns[7]);
        observation.valid =
            observation.confidence >= 0.35;

        const double speed_mps =
            std::stod(columns[8]);

        const double recorded_steering =
            std::stod(columns[9]);

        const double target_speed_mps =
            std::stod(columns[11]);

        const auto command = controller.calculate(
            observation,
            speed_mps,
            target_speed_mps
        );

        const double difference = std::abs(
            command.requested_steering_rad -
            recorded_steering
        );

        maximum_difference =
            std::max(maximum_difference, difference);

        ++replayed_frames;

        if (difference <= tolerance) {
            ++matched_commands;
        }
    }

    const bool passed =
        replayed_frames > 0 &&
        replayed_frames == matched_commands &&
        maximum_difference <= tolerance;

    return passed ? 0 : 2;
}