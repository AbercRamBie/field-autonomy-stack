#include "autonomy/csv_logger.hpp"
#include "autonomy/fault_injector.hpp"
#include "autonomy/health_monitor.hpp"
#include "autonomy/lane_detector.hpp"
#include "autonomy/scene_renderer.hpp"
#include "autonomy/stanley_controller.hpp"
#include "autonomy/track.hpp"
#include "autonomy/types.hpp"
#include "autonomy/vehicle_model.hpp"

#include <opencv2/highgui.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    constexpr double dt_s = 0.05;
    constexpr std::size_t frame_count = 700;

    std::filesystem::create_directories("logs");

    autonomy::Track track;
    autonomy::VehicleModel vehicle;
    autonomy::SceneRenderer renderer;
    autonomy::LaneDetector lane_detector;
    autonomy::StanleyController controller;
    autonomy::FaultInjector fault_injector(dt_s, 42);
    autonomy::HealthMonitor health_monitor;
    autonomy::CsvLogger logger("logs/latest.csv");

    autonomy::VehicleState initial_state;
    initial_state.lateral_error_m = 0.8;
    initial_state.heading_error_rad = 0.08;
    initial_state.speed_mps = 8.0;

    vehicle.reset(initial_state);

    for (std::size_t frame_index = 0;
         frame_index < frame_count;
         ++frame_index) {

        const auto frame_deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<
               std::chrono::steady_clock::duration
            >(std::chrono::duration<double>(dt_s));

        const double simulation_time_s =
            static_cast<double>(frame_index) * dt_s;

        const autonomy::VehicleState true_state =
            vehicle.state();

        const double curvature =
            track.curvatureAt(true_state.distance_m);

        const autonomy::FaultState faults =
            fault_injector.update(
                frame_index,
                simulation_time_s
            );

        cv::Mat camera_frame =
            renderer.render(
                true_state,
                track.laneWidthM(),
                faults
            );

        cv::Mat debug_frame = camera_frame.clone();

        autonomy::LaneObservation observation;

        if (!faults.frame_dropped) {
            observation =
                lane_detector.detect(
                    camera_frame,
                    &debug_frame
                );
        }

        const autonomy::AutonomyMode mode =
            health_monitor.update(observation);

        double target_speed_mps = 8.0;

        if (mode == autonomy::AutonomyMode::Degraded) {
            target_speed_mps = 4.0;
        } else if (
            mode == autonomy::AutonomyMode::SafeStop
        ) {
            target_speed_mps = 0.0;
        }

        autonomy::ControlCommand command =
            controller.calculate(
                observation,
                true_state.speed_mps,
                target_speed_mps
            );

        command.applied_steering_rad =
            fault_injector.applySteeringFault(
                command.requested_steering_rad
            );

        vehicle.step(
            command.applied_steering_rad,
            command.target_speed_mps,
            curvature,
            dt_s
        );

        autonomy::TelemetryRecord record;
        record.frame_index = frame_index;
        record.simulation_time_s = simulation_time_s;
        record.true_state = true_state;
        record.observation = observation;
        record.control = command;
        record.road_curvature = curvature;
        record.mode = mode;
        record.faults = faults;

        logger.write(record);

        std::cout
            << "frame=" << frame_index
            << " mode=" << autonomy::toString(mode)
            << " true_error="
            << true_state.lateral_error_m
            << " measured_error="
            << observation.lateral_error_m
            << " confidence="
            << observation.confidence
            << " steering="
            << command.applied_steering_rad
            << '\n';

        cv::imshow("Field autonomy stack", debug_frame);

        const int key = cv::waitKey(1);

        if (key == 27 || key == 'q') {
            break;
        }

        std::this_thread::sleep_until(frame_deadline);
    }

    return 0;
}