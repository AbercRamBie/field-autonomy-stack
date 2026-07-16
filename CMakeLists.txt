cmake_minimum_required(VERSION 3.20)

project(field_autonomy_stack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
    calib3d
)

add_library(autonomy
    src/track.cpp
    src/vehicle_model.cpp
    src/scene_renderer.cpp
    src/lane_detector.cpp
    src/lane_tracker.cpp
    src/stanley_controller.cpp
    src/q_learning_controller.cpp
    src/fault_injector.cpp
    src/health_monitor.cpp
    src/csv_logger.cpp
)

target_include_directories(autonomy
    PUBLIC
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(autonomy
    PUBLIC
        ${OpenCV_LIBS}
)

target_compile_options(autonomy
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
)

add_executable(run_stack apps/run_stack.cpp)
target_link_libraries(run_stack PRIVATE autonomy)

add_executable(run_video apps/run_video.cpp)
target_link_libraries(run_video PRIVATE autonomy)

add_executable(replay_controller apps/replay_controller.cpp)
target_link_libraries(replay_controller PRIVATE autonomy)

add_executable(train_q_learning apps/train_q_learning.cpp)
target_link_libraries(train_q_learning PRIVATE autonomy)

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(googletest)

    add_executable(autonomy_tests
        tests/vehicle_model_test.cpp
        tests/controller_test.cpp
        tests/detector_test.cpp
        tests/replay_test.cpp
    )

    target_link_libraries(autonomy_tests
        PRIVATE
            autonomy
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(autonomy_tests)
endif()
