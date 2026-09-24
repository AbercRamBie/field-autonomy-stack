# Field Autonomy Stack: Codebase Guide

This document explains the current codebase as it exists today. It is intended
to help you read, debug, and extend the project confidently before adding
reinforcement learning.

The project contains two related workflows:

1. A synthetic closed-loop lane-following simulation (`run_stack`).
2. A real-video perception and controller demonstration (`run_video`).

They share data types, lane detection, health monitoring, and the Stanley
controller, but they are not the same experiment. The synthetic application
updates a vehicle model from steering commands. The video application only
calculates and displays commands; it cannot change what happens in a recorded
video.

## 1. The shortest useful mental model

The intended autonomy loop is:

```text
world/road state
      |
      v
camera image --> lane detector --> lane observation --> health monitor
                                                        |
                                                        v
vehicle model <-- steering command <-- controller <-- operating mode
      |
      +---------------------- next simulation frame --------------------+
```

For the real-video program, the last feedback connection does not exist:

```text
recorded video frame
      |
      v
lane detector --> lane tracker --> health monitor --> Stanley controller
      |                                                   |
      +---------------- rendered overlay/status <---------+

The steering command is displayed, but it does not affect the recorded video.
```

The main value passed between perception and control is `LaneObservation`.
Understanding that structure is the best starting point for understanding the
project.

## 2. Repository layout

```text
fieldautonomystack/
├── apps/                 Executable entry points (`main` functions)
│   ├── run_stack.cpp     Synthetic closed-loop simulation
│   ├── run_video.cpp     Recorded-video lane detection demo
│   ├── replay_controller.cpp
│   └── train_q_learning.cpp       Currently a placeholder
├── autonomy/             Public headers and shared data structures
├── src/                  Implementations of the autonomy modules
├── tests/                GoogleTest unit tests
├── python/               Offline telemetry analysis
├── data/videos/          Input video data
├── results/              Generated plots
└── CMakeLists.txt        Build graph and targets
```

The normal C++ convention used here is:

- `autonomy/example.hpp` declares a class or function.
- `src/example.cpp` implements it.
- `apps/*.cpp` assembles the classes into runnable programs.

Most classes are inside the `autonomy` namespace. This prevents their names
from colliding with names from OpenCV or the C++ standard library.

## 3. Build targets

`CMakeLists.txt` first builds one static library named `autonomy`. The library
contains the reusable modules:

```text
Track
VehicleModel
SceneRenderer
LaneDetector
LaneTracker
StanleyController
FaultInjector
HealthMonitor
CsvLogger
```

Each program then links against this library:

| Target | Entry point | Current purpose |
|---|---|---|
| `run_stack` | `apps/run_stack.cpp` | Synthetic closed-loop simulation |
| `run_video` | `apps/run_video.cpp` | Process and display a recorded video |
| `replay_controller` | `apps/replay_controller.cpp` | Recalculate commands from a CSV log |
| `train_q_learning` | `apps/train_q_learning.cpp` | Placeholder; training is not implemented |

Typical build commands are:

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
cmake --build build -j4
```

The project uses C++17 and OpenCV. Warning flags such as `-Wconversion` and
`-Wshadow` are applied to the `autonomy` library.

## 4. Shared types: the language between modules

All important cross-module messages are declared in
`autonomy/types.hpp`.

### `VehicleState`

```cpp
struct VehicleState {
    double distance_m;
    double lateral_error_m;
    double heading_error_rad;
    double speed_mps;
    double steering_rad;
};
```

This is the simulated vehicle's state relative to the road centreline.

- `distance_m`: distance travelled along the road.
- `lateral_error_m`: sideways displacement from the desired lane centre.
- `heading_error_rad`: difference between vehicle direction and road direction.
- `speed_mps`: current longitudinal speed.
- `steering_rad`: applied front-wheel steering angle.

The sign convention is important. The Stanley controller negates the combined
heading and lateral corrections, so changing the meaning of positive lateral
error requires changing the controller and tests together.

### `LanePolynomial`

Each lane boundary is represented in bird's-eye coordinates as:

```text
x(y) = a*y^2 + b*y + c
```

`y` is the image row and `x(y)` is the expected horizontal position of the
lane boundary at that row. This form is convenient because road markings are
mostly vertical after perspective warping.

### `LaneObservation`

This is the output of perception and the input to tracking, health monitoring,
and control.

It contains:

- validity flags for the full observation and each boundary;
- lateral and heading errors;
- a confidence value;
- processing latency;
- bottom and lookahead boundary positions in camera space;
- left and right bird's-eye polynomials.

One subtlety: `LaneDetector` initially calculates errors from camera-space
points, but `LaneTracker::recalculateErrors()` replaces those errors using the
bird's-eye polynomial geometry. Therefore, `run_video` ultimately controls from
the tracker's bird's-eye calculation.

### `ControlCommand`

- `requested_steering_rad`: controller output before actuator faults.
- `applied_steering_rad`: steering that reaches the simulated vehicle.
- `target_speed_mps`: desired speed selected from the operating mode.

In `run_video`, only `requested_steering_rad` is meaningful because there is no
vehicle or actuator to apply the command.

### `AutonomyMode`

The health state machine has three states:

```text
NORMAL --> DEGRADED --> SAFE_STOP
   ^          ^            |
   |          +------------+
   +-----------------------+
```

Recovery occurs one state at a time; `SAFE_STOP` recovers to `DEGRADED`, then
`DEGRADED` recovers to `NORMAL`.

### `FaultState` and `TelemetryRecord`

`FaultState` describes injected simulation faults. `TelemetryRecord` bundles
one complete simulation timestep for CSV logging.

## 5. Executable 1: `run_stack`

`apps/run_stack.cpp` is the closest thing to a complete autonomy loop.

### Objects constructed at startup

```text
Track             supplies lane width and road curvature
VehicleModel      stores and updates the simulated vehicle state
SceneRenderer     turns state into a synthetic camera image
LaneDetector      estimates the lane from that image
StanleyController converts errors into steering
FaultInjector     injects camera, frame, occlusion, and steering faults
HealthMonitor     chooses NORMAL, DEGRADED, or SAFE_STOP
CsvLogger         writes one row per timestep
```

The simulation uses:

- timestep: `0.05 s` (20 Hz);
- maximum frames: `700`;
- initial lateral error: `0.8 m`;
- initial heading error: `0.08 rad`;
- initial speed: `8 m/s`.

### Per-frame call sequence

For every frame, `main()` performs these calls in order:

```text
vehicle.state()
    |
track.curvatureAt(distance)
    |
fault_injector.update(frame, time)
    |
renderer.render(true_state, lane_width, faults)
    |
lane_detector.detect(camera_frame)       skipped if frame_dropped
    |
health_monitor.update(observation)
    |
controller.calculate(observation, current_speed, target_speed)
    |
fault_injector.applySteeringFault(requested_steering)
    |
vehicle.step(applied_steering, target_speed, curvature, dt)
    |
logger.write(record)
```

The important feedback step is `vehicle.step()`. Its new state is read on the
next iteration, so a steering decision can improve or worsen future errors.
That feedback makes `run_stack` a potential starting point for an RL
environment.

`run_stack` currently does not use `LaneTracker`. It sends the detector output
straight to the health monitor and controller.

## 6. Executable 2: `run_video`

`apps/run_video.cpp` processes a video file at approximately its recorded
frame rate.

Run it with:

```bash
./build/run_video data/videos/lane_detection_test_01.mp4
```

An optional second argument supplies assumed speed:

```bash
./build/run_video data/videos/lane_detection_test_01.mp4 8.0
```

### Startup

1. Validate the command-line arguments.
2. Open the file with `cv::VideoCapture`.
3. Read FPS, falling back to 30 FPS if it is invalid.
4. Construct `LaneDetector`, `LaneTracker`, `HealthMonitor`, and
   `StanleyController`.

### Per-frame call sequence

```text
video.read(source_frame)
    |
cv::resize(source_frame, 640 x 480)
    |
lane_detector.detect(frame, &debug_frame)
    |                  raw_observation
    v
lane_tracker.update(raw_observation)
    |                  tracked observation
    v
health_monitor.update(observation)
    |
select target speed from mode
    |
controller.calculate(observation, assumed_speed, target_speed)
    |
drawStatus(debug_frame, observation, mode, steering)
    |
cv::imshow(...)
```

Every 30 frames, the program prints both `raw_valid` and `valid`:

- `raw_valid`: whether `LaneDetector` accepted this frame.
- `valid`: whether `LaneTracker` supplied an acceptable current or briefly
  held observation.

At end-of-file, the last window stays visible until `q`, Escape, or window
close. This is why the application does not immediately exit when the video
finishes.

### What this application does not do

It does not move a simulated or physical vehicle. `StanleyController` computes
a steering request, but the request is only shown in the window and terminal.
Consequently, this application cannot evaluate whether steering would actually
keep the vehicle in the lane.

## 7. Lane detection in detail

The public interface is small:

```cpp
LaneObservation LaneDetector::detect(
    const cv::Mat& frame,
    cv::Mat* debug_frame
);
```

Most of the implementation is in the anonymous namespace at the top of
`src/lane_detector.cpp`. Anonymous-namespace helpers are private to that source
file; other translation units cannot call them.

### 7.1 Constructor and persistent state

`LaneDetector` constructs forward and inverse perspective matrices for the
expected `640 x 480` image size.

It also retains:

- the latest ten accepted left polynomials;
- the latest ten accepted right polynomials;
- a missed-detection count.

The rolling history implements the smoothing used by the referenced
`Curved-Lane-Lines` notebook. After ten consecutive misses, the history is
cleared so an old road segment is not mixed into a newly acquired lane.

### 7.2 Perspective points

`createSourcePoints()` defines a trapezoid around the lane in camera space.
`createDestinationPoints()` defines its bird's-eye rectangle.

```text
camera image                         bird's-eye image

       top-left  top-right           left       right
          \        /                  |           |
           \ lane /                   |   lane    |
            \    /                    |           |
       bottom-left bottom-right       |           |
```

The source points are calibration parameters, not universal constants. If the
camera position, resolution, crop, or aspect ratio changes, these points need
recalibration.

### 7.3 `createBinaryImage()`

This converts the BGR camera frame to HLS and creates three masks:

1. A normalized Sobel gradient mask from the Lightness channel.
2. A Saturation mask for saturated/yellow markings.
3. A high-Lightness mask for white markings.

The masks are combined with logical OR:

```text
binary = lightness_gradient OR saturation_mask OR white_mask
```

The binary mask is created before perspective warping. It is then warped with
nearest-neighbour interpolation so binary values stay binary.

### 7.4 Histogram and sliding windows

`findLanePixels()` first sums the bottom half of the warped binary image by
column. Strong column totals suggest the base of a lane marking.

The search then uses nine windows from bottom to top:

```text
          [L]                         [R]
          [L]                         [R]
       [L]                            [R]
       [L]                         [R]
    [L]                            [R]
---------------- image bottom ----------------
```

For each window:

1. Collect non-zero pixels within its x/y bounds.
2. Append them to the left or right point set.
3. Recenter the next window on the mean x coordinate of those pixels.

The initial histogram ranges are constrained for the ego lane. This adaptation
is necessary because the input is a multi-lane highway; an unrestricted half-
image maximum can select the yellow road edge or an adjacent marking.

### 7.5 `fitPolynomial()`

For each boundary, the code solves this least-squares system:

```text
[y0^2 y0 1] [a]   [x0]
[y1^2 y1 1] [b] = [x1]
[ ...     ] [c]   [...]
```

`cv::solve(..., cv::DECOMP_SVD)` produces `a`, `b`, and `c`. SVD is used
because it is more numerically tolerant than directly inverting the matrix.

The left and right boundaries are fitted independently.

### 7.6 Ten-frame rolling mean

The measured coefficients are appended to two deques. The detector averages
up to ten values for every coefficient:

```text
smoothed_a = mean(a[t], a[t-1], ..., a[t-9])
smoothed_b = mean(b[t], b[t-1], ..., b[t-9])
smoothed_c = mean(c[t], c[t-1], ..., c[t-9])
```

This removes frame-to-frame jitter, at the cost of several frames of response
delay on a changing curve.

### 7.7 `validateLane()`

The averaged curves are rejected if:

- coefficients or evaluated points are not finite;
- the left boundary crosses the right boundary;
- lane width falls outside configured pixel limits;
- lane width changes too much from top to bottom;
- a boundary base moves into an implausible region.

This validation is stricter than the reference notebook, which assumes a clean
input video. It prevents a smooth but incorrect pair of curves from being sent
to control.

### 7.8 Returning to camera coordinates

The detector samples both bird's-eye curves and calls
`cv::perspectiveTransform()` with the inverse matrix. It then finds boundary
positions near the bottom and camera-space lookahead row.

From those positions it calculates:

```text
lane centre = (left_x + right_x) / 2

lateral error = (image centre - lane centre) / pixels_per_metre

heading error = atan2(
    lane_centre_bottom - lane_centre_lookahead,
    vertical lookahead distance
)
```

### 7.9 Confidence

Confidence is based on the smaller supporting-pixel count:

```text
support_score = clamp(min(left_pixels, right_pixels) / 2000, 0, 1)
confidence    = 0.4 + 0.6 * support_score
```

This is a heuristic, not a calibrated probability. A value of `0.9` does not
mean there is a scientifically measured 90% probability that the lane is
correct.

### 7.10 Overlay

`createLaneOverlay()`:

1. samples the left and right polynomials;
2. creates a polygon between them;
3. fills the polygon in bird's-eye space;
4. draws blue and red boundaries;
5. inverse-warps the overlay;
6. blends it with the source frame.

The overlay is diagnostic output. The controller uses numeric values from
`LaneObservation`, not pixels from the rendered green polygon.

## 8. Lane tracking

`LaneTracker::update()` receives the raw detector observation.

It performs three jobs:

1. Reject impossible or discontinuous geometry.
2. Briefly hold the last accepted observation during missed frames.
3. Recalculate control errors from the bird's-eye polynomials.

The default tracker smoothing alpha is currently `1.0`, meaning it does not
add another exponential filter. The detector already applies a ten-frame
rolling mean.

### Accepted detection

If the candidate is acceptable:

- missed-frame count returns to zero;
- geometry is copied;
- lateral and heading errors are recalculated;
- it becomes `previous_` for the next frame.

### Rejected detection

If a candidate is rejected:

- `missed_frames_` increases;
- the previous observation can be held for up to five frames;
- confidence decays by `0.80^missed_frames`;
- after the limit, tracker state is reset and an invalid observation is
  returned.

Holding a lane is useful for one or two damaged frames, but it is not a new
detection. That distinction is why `run_video` prints raw and tracked validity.

## 9. Health monitor

`HealthMonitor::update()` converts perception quality into an operating mode.

Definitions:

```text
healthy: observation is valid and confidence >= 0.70
failed:  observation is invalid or confidence < 0.35
```

Transitions use consecutive-frame counters:

- 4 low-confidence frames: `NORMAL -> DEGRADED`.
- 12 failed frames while degraded: `DEGRADED -> SAFE_STOP`.
- 8 healthy frames: recover by one state.

In `run_stack` and `run_video`, mode selects target speed:

| Mode | Target speed behavior |
|---|---|
| `NORMAL` | Requested normal speed |
| `DEGRADED` | Limited to approximately `4 m/s` |
| `SAFE_STOP` | `0 m/s` |

The video still plays at its normal rate because target speed does not control
video playback.

## 10. Stanley controller

The controller is deterministic and stateless. Its main calculation is:

```text
lateral_correction = atan2(
    lateral_gain * lateral_error,
    speed + speed_softening
)

steering = -(heading_error + lateral_correction)
```

The result is clamped to `[-0.45, +0.45] rad`.

Why speed is in the denominator: at high speed, a very aggressive steering
correction is undesirable. `speed_softening` also prevents division by zero
and excessive correction near zero speed.

If the lane observation is invalid, requested steering is zero. The target
speed is still copied into the returned command.

## 11. Synthetic world modules

### `Track`

`Track::curvatureAt(distance)` combines two sine waves:

```text
curvature = 0.012*sin(distance/24) + 0.004*sin(distance/7)
```

This creates long and short bends without loading a map. Lane width is fixed at
`3.5 m`.

### `VehicleModel`

The model is a simplified kinematic bicycle/error model.

Speed moves toward the target with acceleration limited to `2 m/s^2`.

```text
lateral_rate = speed * sin(heading_error)

heading_rate =
    (speed / wheelbase) * tan(steering)
    - speed * road_curvature
```

Euler integration updates state:

```text
state += rate * dt
```

This model is intentionally simple. It does not model tyre slip, actuator
dynamics, roll, road friction, or steering rate limits.

### `SceneRenderer`

The renderer converts simulated lateral and heading errors into a simple road
image:

- lateral error moves the road centre horizontally;
- heading error moves the horizon centre;
- perspective narrows the lane toward the horizon;
- faults can add noise or cover part of the lane.

This is a visual sensor model, not a photorealistic simulator.

## 12. Fault injection

`FaultInjector` uses simulation time to activate deterministic fault windows:

| Time | Fault |
|---|---|
| 8–11 s | Gaussian camera noise |
| 14–16 s | 35% random frame-drop probability |
| 19–23 s | Lane occlusion rectangle |
| 26–29 s | 0.20 s steering delay |

The random generator uses a default seed of 42, making runs repeatable.

The steering-delay fault uses a queue. During the fault, an older command is
returned instead of the latest command.

## 13. Logging, replay, and analysis

### CSV logging

`CsvLogger` writes `logs/latest.csv` during `run_stack`. Each row contains:

- time and frame;
- true vehicle state;
- measured errors and confidence;
- requested and applied steering;
- target speed and road curvature;
- autonomy mode;
- active faults;
- perception processing time.

The file is flushed every 20 frames to reduce data loss without flushing on
every row.

### Controller replay

Run:

```bash
./build/replay_controller logs/latest.csv
```

The replay program reads recorded perception inputs, reruns
`StanleyController::calculate()`, and compares the new steering result with the
recorded requested steering using a tolerance of `1e-9`.

This tests determinism of the controller calculation. It does not replay the
vehicle model, renderer, detector, faults, or health-state transitions.

### Python analysis

Run:

```bash
python3 python/analyse_run.py logs/latest.csv
```

The script prints performance statistics and generates:

- `results/lateral_error.png`;
- `results/steering.png`;
- `results/confidence.png`.

It uses pandas for CSV loading and Matplotlib for plotting.

## 14. Tests

Current tests cover:

- controller sign and steering limits;
- neutral steering for invalid perception;
- deterministic repeated controller output;
- basic vehicle-model behavior;
- lane detection on the synthetic renderer.

Important current test-build issue: `CMakeLists.txt` refers to
`tests/vehicle_model_test.cpp`, but the file in the repository is named
`tests/vehicle_model_tests.cpp`. A fresh test-enabled configuration will need
that mismatch corrected.

The detector test was written for the synthetic detector/renderer combination.
Because the detector is now calibrated around the real-video perspective, that
test should be revalidated rather than assumed to pass.

There are currently no focused tests for:

- `LaneTracker` rejection and hold behavior;
- health-monitor state transitions;
- fault timing and steering delay;
- CSV schema/replay compatibility;
- the ten-frame polynomial history;
- video end-of-file behavior.

## 15. Current reinforcement-learning status

Reinforcement learning is not implemented yet.

- `apps/train_q_learning.cpp` only prints `Field autonomy stack`.
- `src/q_learning_controller.cpp` contains only a placeholder comment.
- There is no Q-learning header, state encoder, action set, Q-table, reward
  function, episode loop, policy persistence, or evaluation program.

This is useful to know before beginning: you are designing the RL subsystem,
not merely tuning an existing one.

## 16. A sensible RL boundary

Start from the synthetic closed loop, not recorded video. Recorded video cannot
respond to an action, so it cannot provide the action-state-next-state cycle
needed for standard online reinforcement learning.

The environment step should look like:

```text
state_t
   |
agent chooses action_t
   |
VehicleModel::step(action_t, ...)
   |
new simulated state state_t+1
   |
calculate reward and termination
```

### Suggested first state

Keep the first version small and discretized:

```text
lateral-error bin
heading-error bin
speed bin
road-curvature bin
```

Do not begin with raw camera pixels and tabular Q-learning. The state space
would be far too large.

### Suggested actions

A small discrete steering set is sufficient for tabular Q-learning:

```text
{-0.45, -0.30, -0.15, 0.0, +0.15, +0.30, +0.45} rad
```

### Suggested reward components

For example:

```text
reward =
    + progress_reward
    - lateral_error_weight * abs(lateral_error)
    - heading_error_weight * abs(heading_error)
    - steering_change_weight * abs(action - previous_action)
    - large_lane_departure_penalty
```

The steering-change penalty matters because an agent can otherwise learn a
jittery left-right policy that stays near the centre but is not driveable.

### Episode termination

Terminate an episode when any of these happens:

- absolute lateral error exceeds half the lane width;
- a maximum duration or distance is reached;
- state becomes non-finite;
- optionally, the health monitor enters `SAFE_STOP`.

### Baseline comparison

Keep `StanleyController` as the baseline. Evaluate both controllers on the same
initial states, track curvature, duration, and random seeds. Compare:

- mean and 95th-percentile lateral error;
- lane departures;
- steering smoothness;
- distance travelled;
- recovery from disturbances.

Without a baseline, an increasing Q value does not prove that driving quality
improved.

## 17. Important technical debt before RL

These issues should be understood and preferably addressed before trusting RL
results:

1. **RL files are placeholders.** The training interface must be designed.
2. **Test filename mismatch.** Test-enabled CMake refers to the wrong vehicle
   test filename.
3. **Detector and synthetic renderer calibration differ.** The current
   perspective points were tuned for the real highway video.
4. **`run_stack` bypasses `LaneTracker`.** Simulation and video therefore use
   different perception-to-control paths.
5. **The detector opens GUI windows internally.** A training loop should be
   able to run headlessly and much faster than real time.
6. **Recorded video is resized from its original aspect ratio to 640 x 480.**
   This distorts geometry and ties calibration to that resize.
7. **No source-camera calibration is applied.** Lens distortion is not removed.
8. **Confidence is heuristic.** It should not be interpreted as a probability.
9. **Some configuration values are unused.** `LaneDetector::horizon_y_px_` and
   `RenderConfig::heading_pixels_per_rad` currently do not influence their
   implementations.
10. **Several source files still say `// Placeholder` despite containing real
    implementations.** Those comments are stale.

RL can exploit simulator or reward mistakes. Fixing the experiment boundary
and tests is more important than choosing a sophisticated learning algorithm.

## 18. Recommended reading order

Read the code in this order and trace values with a debugger or temporary
logging:

1. `autonomy/types.hpp`
2. `apps/run_stack.cpp`
3. `autonomy/vehicle_model.hpp` and `src/vehicle_model.cpp`
4. `autonomy/stanley_controller.hpp` and `src/stanley_controller.cpp`
5. `autonomy/health_monitor.hpp` and `src/health_monitor.cpp`
6. `autonomy/scene_renderer.hpp` and `src/scene_renderer.cpp`
7. `autonomy/lane_detector.hpp` and `src/lane_detector.cpp`
8. `autonomy/lane_tracker.hpp` and `src/lane_tracker.cpp`
9. `apps/run_video.cpp`
10. fault injection, logging, replay, and analysis
11. tests

`src/lane_detector.cpp` is much easier to understand after the shared types and
main application flows are already clear.

## 19. Practical learning exercises

Try these before writing RL code:

1. Put a breakpoint on `StanleyController::calculate()` and inspect the
   observation and command for ten frames.
2. Manually calculate one Stanley steering command and compare it with the
   debugger value.
3. Disable one fault window and predict how the CSV plots will change.
4. Change the initial lateral error in `run_stack` and observe recovery.
5. Print raw and tracked errors together in `run_video` to see when tracking
   holds an older detection.
6. Plot the left/right polynomial x positions at y = 280 and y = 479.
7. Add a unit test for every health-monitor transition.
8. Add a unit test proving the vehicle moves toward the lane centre for the
   controller's expected sign convention.
9. Make rendering optional so a simulation can run faster than real time.
10. Only then define the RL environment API.

## 20. Glossary

| Term | Meaning in this project |
|---|---|
| Camera space | Normal front-facing image coordinates |
| Bird's-eye space | Perspective-warped top-down road coordinates |
| Lateral error | Sideways distance from desired lane centre |
| Heading error | Angular difference from the road direction |
| Polynomial fit | Quadratic approximation of a lane boundary |
| Sliding window | Bottom-to-top pixel search around a lane marking |
| Tracker | Temporal validation and brief missed-frame handling |
| Controller | Converts errors into a steering request |
| Plant/model | System changed by the command; here, `VehicleModel` |
| Closed loop | Commands influence the next observed state |
| Open loop | Commands are calculated but cannot affect the input video |
| Episode | One RL rollout from reset until termination |
| Policy | Rule mapping a state to an action |
| Q value | Estimated long-term return for a state-action pair |

## 21. The most important distinction before RL

Lane detection answers:

> Where does the camera think the lane is?

Tracking answers:

> Is this detection believable and temporally consistent?

Health monitoring answers:

> Is perception reliable enough to continue normally?

Control answers:

> Given the measured error, what steering should be requested?

The vehicle model answers:

> After that steering action, what is the next state?

Reinforcement learning replaces or augments the action-selection part. It does
not remove the need for a clear environment, state definition, safety limits,
tracking, evaluation, or a trustworthy next-state transition.
