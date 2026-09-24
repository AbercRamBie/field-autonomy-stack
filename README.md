# Field Autonomy Stack

A C++17 lane-detection and lateral-control project with a synthetic closed-loop
simulation, a recorded-video demonstration, and an experimental real-driving
data preparation pipeline.

## Dependencies

- CMake 3.20 or newer
- A C++17 compiler
- OpenCV with `core`, `imgproc`, `highgui`, `videoio`, and `calib3d`
- Python 3 with NumPy, pandas, and Matplotlib for the Python utilities
- Internet access during the first test configuration so CMake can fetch
  GoogleTest

## Build

From the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel 2
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Main simulation

Run the synthetic lane-following simulation with the Stanley controller:

```bash
./build/run_stack stanley
```

To use the residual Q-learning controller, train its table first:

```bash
./build/train_q_learning
./build/run_stack residual_q
```

## Recorded-video demonstration

Run lane detection and control visualization on the included video. The final
argument is the assumed vehicle speed in metres per second.

```bash
./build/run_video data/videos/lane_detection_test_01.mp4 8.0
```

## Experimental real-data pipeline

The pipeline under `python/rl/` prepares synchronized observations and
behavior-cloning samples from a comma2k19 segment containing road video,
timestamps, CAN signals, and IMU data. It is an experimental data-preparation
path, not a complete offline-RL trainer or a vehicle-ready control system.

First audit the recorded steering sign and timing:

```bash
python3 python/rl/audit_steering_dynamics.py \
    --segment-dir /path/to/comma2k19/segment
```

Then process the segment using the verified steering ratio and sign:

```bash
python3 python/rl/process_segment.py \
    --segment-dir /path/to/comma2k19/segment \
    --route-id route_001 \
    --output-dir data/processed/route_001 \
    --steering-ratio 16.88 \
    --steering-sign 1 \
    --extractor build/extract_observations
```

This wrapper extracts lane observations, aligns them with the recorded vehicle
signals, and writes behavior-cloning samples plus a reproducibility manifest.
