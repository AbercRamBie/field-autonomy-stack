# Field Autonomy Stack: Interview Preparation

## 1. The project in one sentence

This is a small, modular lane-following autonomy stack that turns camera images
into lane geometry, converts that geometry into steering, monitors perception
health, applies the command to a simple vehicle model in simulation, and records
enough telemetry to replay and analyse the result.

Do not describe it as a production self-driving system. It is an educational
prototype for reasoning about the interfaces, feedback, failure handling, and
validation of an autonomy pipeline.

## 2. Answers at three interview depths

### 15-second answer

> I built a C++17/OpenCV lane-following stack. It uses classical vision to fit
> lane-boundary polynomials, a Stanley controller for lateral control, a bicycle
> model for closed-loop simulation, and a health state machine for degraded and
> safe-stop behaviour. I also added deterministic fault injection, telemetry,
> replay, and a small residual Q-learning experiment.

### 45-second answer

> The main data contract is a `LaneObservation`, which decouples perception from
> control. In the closed-loop application, a renderer creates a camera frame
> from vehicle state, the detector estimates lateral and heading error, the
> health monitor chooses normal, degraded, or safe-stop mode, and Stanley turns
> those errors into a bounded steering command. The bicycle model advances the
> vehicle, closing the feedback loop. A second application runs the same
> perception and controller over recorded video, but only in open loop because
> its commands cannot change a recording. The project taught me that confidence,
> calibration, timing, and fallback behaviour matter at least as much as the
> nominal controller.

### Two-minute answer

Explain it in this order:

1. The problem is lane centring, not general autonomous driving.
2. The architecture is modular and joined by explicit data structures.
3. The synthetic program is closed loop; the video program is not.
4. Classical vision makes the perception path inspectable.
5. Stanley is a simple, explainable baseline controller.
6. A state machine reduces speed or stops when perception is unreliable.
7. Fault injection, logging, replay, and tests provide validation hooks.
8. Tabular Q-learning adds only a small residual to Stanley, limiting learned
   authority and preserving a deterministic fallback.
9. The current calibration and validation are prototype quality. The next step
   is scenario-based testing and properly calibrated real-vehicle data, not a
   claim that it is road-ready.

## 3. The most important distinction

There are two related but different experiments.

### Synthetic closed-loop path: `run_stack`

```text
Track curvature + vehicle state
              |
              v
       SceneRenderer -- camera/frame/occlusion faults
              |
              v
        LaneDetector
              |
              v
        HealthMonitor ----> target speed: 8 / 4 / 0 m/s
              |
              v
    StanleyController + optional Q residual
              |
              v
        actuator-delay fault
              |
              v
         VehicleModel
              |
              +---------- next timestep ----------+
```

This is closed loop because the steering command changes the next vehicle
state, which changes the next rendered image and therefore the next command.

### Recorded-video path: `run_video`

```text
recorded frame -> detector -> tracker -> health monitor -> controller
       |                                                   |
       +--------------- display overlay/status <----------+
```

This is open loop. The steering output is a counterfactual display value; it
cannot affect later frames in a video that has already been recorded. Therefore
video replay can evaluate perception availability and command plausibility, but
not closed-loop driving performance.

## 4. The shared data contract

`autonomy/types.hpp` is the best file to read first.

- `VehicleState` is the simulator's ground truth: progress, lateral error,
  heading error, speed, and steering.
- `LanePolynomial` represents a boundary in bird's-eye space as
  `x(y) = ay^2 + by + c`. After perspective warping, lane lines are primarily
  vertical, so predicting horizontal position `x` from row `y` is convenient.
- `LaneObservation` is the boundary between perception and downstream logic.
  It contains validity, freshness, confidence, errors, latency, boundary points,
  and both polynomials.
- `ControlCommand` distinguishes requested steering from applied steering. That
  distinction makes actuator delay observable and testable.
- `TelemetryRecord` captures one whole simulation step for offline diagnosis.

Why use plain structures? They make module boundaries visible, logging easy,
and unit tests small. The simplest alternative is to pass several primitive
arguments. That works initially but becomes error-prone as the interface grows.
A production system would add timestamps, coordinate-frame identifiers, units,
schema/version metadata, and stronger type safety.

## 5. One frame, step by step

At 20 Hz (`dt = 0.05 s`), `run_stack` does the following:

1. Read the current `VehicleState`.
2. Ask `Track` for road curvature at the current distance.
3. Activate any scheduled fault for this time.
4. Render a 640x480 synthetic road image.
5. Detect lane boundaries unless the frame was dropped.
6. Update the health state from observation validity and confidence.
7. Select a target speed: 8 m/s normally, 4 m/s degraded, or 0 m/s in safe
   stop.
8. Calculate nominal Stanley steering.
9. Optionally add one learned residual action and clamp total steering.
10. Apply any actuator-delay fault.
11. Step the vehicle model.
12. Log truth, measurement, mode, faults, and requested/applied commands.

The ordering is a design decision. Health gates speed before the model step;
faults sit at the sensor and actuator boundaries; logging records both truth and
measurement so estimation error is distinguishable from control error.

## 6. Component decomposition and design choices

### `Track`

What it does: returns a 3.5 m lane width and a deterministic sum of two sine
waves as road curvature.

Why: it creates repeatable straight and curved conditions with almost no map
infrastructure. A fixed random seed and deterministic road make regression
comparisons reproducible.

Simplest alternative: constant zero curvature. That proves basic centring but
does not exercise heading compensation on bends.

Limitation: it is not a geometric road/map model, so it cannot represent
junctions, changing widths, topology, or realistic curvature continuity.

### `VehicleModel`

What it does: uses a kinematic bicycle-style error model. In simplified form:

```text
lateral_rate = speed * sin(heading_error)
heading_rate = speed/wheelbase * tan(steering) - speed * road_curvature
```

Speed approaches its target with acceleration clamped to +/-2 m/s², steering
is clamped to +/-0.45 rad, and explicit Euler integration advances the state.

Why: a kinematic model is cheap, deterministic, and adequate for explaining
low-level lateral-control feedback at moderate speed.

Simplest alternative: directly change lateral error in proportion to steering.
That is easier but removes heading dynamics and wheelbase, making controller
behaviour much less realistic.

Limitation: there are no tyre forces, slip, actuator dynamics, steering-rate
limit, latency except the injected delay, grade, localisation error, or model
uncertainty. It should not be used to claim high-speed vehicle stability.

### `SceneRenderer`

What it does: projects the simulated lateral and heading errors into two lane
lines on a synthetic road image, with optional noise or occlusion.

Why: it exercises the actual image-processing path instead of feeding perfect
state directly into the controller.

Simplest alternative: create a perfect `LaneObservation` from `VehicleState`.
That is useful for controller unit tests and Q-learning training, but it cannot
test perception/control integration.

Limitation: the renderer and detector use different effective camera
calibrations. The image is a trapezoid, not a physical camera model, and the
declared `heading_pixels_per_rad` setting is unused.

### `LaneDetector`

What it does:

1. Converts BGR to HLS.
2. Combines a lightness gradient, saturation threshold, and white threshold.
3. Applies a fixed inverse-perspective mapping to get a bird's-eye view.
4. Finds starting points from a bottom-half histogram.
5. Uses nine sliding windows to collect left/right pixels.
6. Fits `x(y) = ay² + by + c` using SVD least squares.
7. Rejects implausible lane width, crossing boundaries, and off-image fits.
8. Averages accepted polynomials over ten frames.
9. Converts geometry into lateral/heading error and a heuristic confidence.

Why classical vision: it is small, debuggable, requires no labelled training
set, and exposes every failure mode. This is a defensible prototype choice when
the goal is learning system integration rather than maximising perception
generality.

Simplest alternative: detect white pixels and fit two straight lines with a
Hough transform. That is shorter but handles curves and temporal continuity
poorly. A learned segmentation model would handle greater appearance variety,
but adds training-data, runtime, uncertainty, and deployment complexity.

Important detail: confidence is computed from supporting-pixel count and is a
heuristic score, not a calibrated probability. Say “confidence score,” never
“70% probability the lane is correct.”

### `LaneTracker`

What it does: rejects implausible temporal jumps and lane-width changes, holds
the last accepted observation for up to five missed frames with decaying
confidence, and recalculates control errors from bird's-eye polynomials.

Why: frame-by-frame detections flicker. Short temporal memory improves
continuity without requiring a full probabilistic state estimator.

Simplest alternative: use each detector result directly. `run_stack` currently
does exactly that. It has lower complexity but is more sensitive to one-frame
outliers and makes the simulation and video paths inconsistent.

Production alternative: timestamped tracking with a Kalman/filtering model and
explicit covariance. Holding the last value is not prediction; its uncertainty
should grow with time and vehicle motion.

### `StanleyController`

The nominal command is:

```text
lateral_correction = atan2(k * lateral_error, speed + softening)
steering = -(heading_error + lateral_correction)
steering = clamp(steering, -0.45, +0.45)
```

Why: Stanley combines heading alignment and cross-track correction, is
computationally cheap, and is understandable enough to calculate by hand. The
speed term naturally reduces aggressive lateral correction at higher speed.
Softening avoids division-like instability near zero speed.

Simplest alternative: proportional steering from lateral error only. It is
easy to tune but ignores whether the vehicle is pointing toward or away from
the lane and tends to oscillate. Pure pursuit is another reasonable baseline
when a geometric path and lookahead point are available.

Why clamp: it enforces the model's steering authority and prevents extreme
commands. A real controller also needs steering-rate and acceleration limits.

### `HealthMonitor`

What it does:

```text
NORMAL --4 unhealthy frames--> DEGRADED --12 low/failed frames--> SAFE_STOP
   ^                                ^                               |
   +--------8 healthy frames--------+--------8 healthy frames-------+
```

- Healthy means valid and confidence >= 0.70.
- Failed means invalid or confidence < 0.35.
- The intermediate band is neither healthy nor failed, but still increments
  the low-confidence counter.
- Recovery happens one state at a time.

Why: persistence thresholds add hysteresis, so a single bad frame does not
cause abrupt mode switching. Degrading speed before stopping is a simple form
of graceful degradation.

Simplest alternative: stop immediately on every invalid frame. That is
conservative but can cause unstable stop/start behaviour. A production safety
supervisor would monitor many independent signals, deadlines, hardware status,
localisation, planning validity, stopping feasibility, and a defined minimal
risk condition.

Subtle implementation point: in `DEGRADED`, transition to `SAFE_STOP` requires
the current frame to meet `failed`, although the 12-frame counter may include
intermediate-confidence frames. This deserves an explicit requirement and a
unit test.

### `FaultInjector`

What it does: schedules camera noise, probabilistic frame drops, lane
occlusion, and 200 ms steering delay. Random behaviour uses seed 42.

Why: repeatable faults let the health path and requested-versus-applied command
behaviour be examined without waiting for rare real failures.

Simplest alternative: manually invalidate a frame in a unit test. That is
excellent for one transition but does not test a timed end-to-end sequence.

Limitation: four fixed windows are not scenario coverage. A better framework
would parameterise duration, severity, combinations, expected response, and
pass/fail metrics, then sweep seeds and boundary cases.

### Telemetry, replay, and analysis

`CsvLogger` records one row per step and flushes every 20 frames. The replay
program reconstructs Stanley inputs from CSV and requires command equality
within `1e-9` rad. The Python analysis reports lateral-error and latency metrics
and creates plots.

Why: observability makes failures diagnosable; deterministic replay checks that
the controller calculation has not silently changed.

Simplest alternative: print to the console. That helps during development but
is difficult to compare, query, plot, or use as regression evidence.

Limitation: CSV has no schema version or run manifest, replay reconstructs
validity from confidence instead of logging/reusing the exact validity flag,
and the Python environment is not pinned.

### Residual tabular Q-learning

The Q controller discretises:

- lateral error into 11 bins;
- heading error into 11 bins;
- previous steering into 5 bins.

That gives `11 * 11 * 5 = 605` states. Each state has three residual actions:
`-0.06`, `0`, or `+0.06` rad. Training uses epsilon-greedy exploration and the
standard update:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
```

The residual is added to Stanley and total steering is clamped. At runtime it
is permitted only in normal mode with a valid observation and confidence at
least 0.70.

Why residual control: the learned component has narrow authority and learns a
correction around an explainable baseline. If it is gated out, Stanley remains
available. This is easier to inspect than asking tabular RL to learn the entire
steering policy.

Why include previous steering: it gives the otherwise memoryless table some
information related to command smoothness.

Simplest alternative: tune Stanley's gain. That should be the first baseline;
RL is justified only if it improves held-out metrics across varied scenarios.

Limitation: training uses perfect simulator state rather than detected state,
only three actions, and a small deterministic dynamics family. A trained table
can exploit simulator/reward assumptions and is not evidence of safe real-world
control. The real-data code currently builds behaviour-cloning samples, but the
documented neural BC and TD3+BC training/inference stages are not implemented.

## 7. Why this is relevant to a company like Oxa

As of September 2026, Oxa publicly describes an industrial-autonomy platform
with on-vehicle perception/reasoning/control, configurable operation across
vehicles and domains, simulation and assurance tooling, modular vehicle
integration, and fleet monitoring. Its public safety material emphasises
virtual, closed-course, on-road, and in-use validation.

The useful parallels are:

| This project | Oxa-relevant engineering theme | Honest boundary |
|---|---|---|
| Detector -> observation -> controller | Modular on-vehicle autonomy interfaces | This project handles lane centring only, not full perception, localisation, prediction, or planning |
| Bicycle model and renderer | Closed-loop virtual testing | This renderer is not a realistic digital twin and has little scenario diversity |
| Fault injector and health state machine | Resilience, degraded behaviour, and safety monitoring | A three-state confidence monitor is not a safety case or production minimal-risk system |
| Deterministic logging and replay | Explainability, traceability, regression, and in-use monitoring | CSV replay covers one controller equation, not system-level assurance |
| Track/vehicle interfaces | Configurability across routes and vehicle platforms | Parameters are mostly hard-coded and there is one vehicle model |
| Residual learning around Stanley | Bounded learned authority with an explainable baseline | The table is simulator-trained and has no real-world safety evidence |
| Real-video/CAN alignment scripts | Data quality, calibration, and offline evaluation | The pipeline needs route-level validation, training, export, and shadow-mode completion |

Useful official reading:

- [Oxa technology](https://oxa.tech/technology/) describes Oxa Driver, Foundry,
  modular autonomy hardware, and Oxa Hub.
- [Oxa safety](https://oxa.tech/safety/) describes simulation, closed-course,
  on-road, and in-use validation.
- [Oxa industrial solutions](https://oxa.tech/solutions/) shows the ports,
  yards, airports, manufacturing, and energy-site operational domains.

The strongest connection is not “I built what Oxa built.” It is:

> I built a small system that forced me to reason about the same categories of
> engineering problem—module contracts, closed-loop behaviour, uncertainty,
> fallback, vehicle interfaces, simulation, telemetry, and reproducibility—at
> prototype scale. I also understand exactly what evidence and architecture are
> still missing before deployment.

## 8. Design-choice answers to memorise

### Why C++17?

Predictable performance, direct OpenCV support, strong compile-time interfaces,
and relevance to on-vehicle robotics. Python is retained for offline analysis
where iteration speed matters more than deterministic runtime.

### Why OpenCV instead of deep learning?

For a constrained lane-marking prototype, classical vision is transparent,
small, and train-data-free. It exposes calibration and geometry directly. I
would reconsider that choice for varied weather, road types, worn markings,
shadows, and complex scenes.

### Why a static library plus small apps?

The library contains reusable mechanisms; each app owns orchestration and I/O.
This avoids duplicating controller or perception code and makes components unit
testable without launching a whole application.

### Why dependency injection through constructors/config structures?

It makes gains and geometric thresholds explicit and testable. The project is
only part-way there: many constants remain embedded in implementation files and
should move into validated configuration.

### Why deterministic seeds?

A failure can be replayed and a before/after change can use identical random
conditions. Robustness still requires many seeds; one deterministic run is a
debugging aid, not statistical evidence.

### Why distinguish truth, observation, requested command, and applied command?

They locate the source of error. Truth versus observation exposes perception
error; requested versus applied exposes actuator/interface faults. Collapsing
them makes root-cause analysis much harder.

### Why not let RL control everything?

Stanley is a known, inspectable baseline. A bounded residual reduces the action
space and limits how far learning can deviate. The real safety question remains
whether the gate, bound, fallback, and training distribution have adequate
evidence.

## 9. Current evidence: say this honestly

Checks performed on 5 September 2026:

- A clean `BUILD_TESTING=OFF` CMake configure and build succeeds with GCC 13.3
  and OpenCV 4.6.
- The repository has seven small unit tests covering parts of the vehicle,
  Stanley controller, detector, and deterministic replay.
- Test-enabled CMake names `tests/vehicle_model_test.cpp`, but the repository
  contains `tests/vehicle_model_tests.cpp`. Fix this before claiming tests pass.
- The latest `logs/latest.csv` contains 700 frames: 3 `NORMAL`, 8 `DEGRADED`,
  and 689 `SAFE_STOP`. Only two rows have confidence at least 0.35. This is
  strong evidence that the current synthetic renderer/detector calibration is
  not functioning as a useful end-to-end baseline in that recorded run.
- The analysis script could not run in the current system Python because
  `matplotlib` is missing. Dependency setup is not yet reproducible.
- Existing codebase documentation still says Q-learning is unimplemented even
  though tabular training/runtime code and a Q-table now exist. Documentation
  has drifted.

These facts improve an interview answer if framed as an engineering review:

> My first goal would be to restore a trustworthy baseline: fix the test target,
> calibrate or unify the renderer/detector geometry, add health-state tests, run
> a scenario matrix, and record versioned metrics. I would not tune RL against a
> broken perception loop because it could optimise around the wrong failure.

## 10. High-value improvements, in order

1. Fix the test filename and add tests for every health-state transition.
2. Create a headless mode so simulation and CI do not require GUI windows.
3. Unify camera calibration between renderer, detector, and tracker; remove
   unused parameters.
4. Make `run_stack` use the tracker or explain and test why the paths differ.
5. Establish baseline acceptance metrics: valid perception rate, mean/P95
   lateral error, departures, steering jerk/rate, stop distance, and recovery.
6. Replace fixed fault windows with a scenario specification and parameter
   sweeps over timing, severity, combinations, and random seeds.
7. Add timestamps, deadlines, stale-data handling, coordinate frames, units,
   configuration snapshots, Git revision, and schema version to telemetry.
8. Calibrate the real camera and steering mapping; split real data by complete
   route to prevent adjacent-frame leakage.
9. Compare Stanley-only and tuned-Stanley before residual Q-learning.
10. Keep learned policies in offline/shadow mode until simulation, closed-course,
    and safety-process evidence justify greater authority.

## 11. Likely interview questions

### “What was the hardest part?”

Good answer: maintaining consistent geometry and sign conventions across image
space, bird's-eye space, vehicle error state, and steering. Explain how a sign
mistake can create positive feedback, and how you would test a positive and a
negative initial offset end to end.

### “How do you know it works?”

Do not answer only with “the video looks right.” Separate evidence:

- unit tests for equations and state transitions;
- deterministic replay for controller regression;
- closed-loop metrics for centring and recovery;
- scenario coverage for faults;
- held-out real-video metrics for perception;
- closed-course evidence before physical authority.

Then acknowledge that the current end-to-end log fails its perception baseline,
so the honest answer is that individual mechanisms work, while the integrated
stack needs recalibration and stronger validation.

### “Why Stanley rather than PID?”

Stanley directly combines cross-track and heading error and scales the lateral
term with speed. A PID can control lateral error but needs careful formulation
to account for heading/path geometry and integral wind-up. PID is still a valid
baseline worth comparing.

### “What happens when perception disappears?”

In video, the tracker can briefly hold the last observation with decaying
confidence. The health monitor then degrades speed after persistent unhealthy
input and eventually commands a safe-stop target. Invalid observations cause
neutral requested steering. Discuss the limitation: neutral steering is not
necessarily the safest path while braking on a curve.

### “What would break first in the real world?”

Fixed camera perspective and colour thresholds: camera pose, lens distortion,
lighting, shadows, worn lines, adjacent markings, and road topology violate the
assumptions. After that, timing/actuator calibration and the simplistic safety
logic become major risks.

### “How would you test it?”

Start with requirements per module, unit-test boundaries and sign conventions,
then closed-loop scenarios covering road curvature, offsets, speeds, perception
loss, delay, and combinations. Sweep parameters and seeds, compare against
acceptance thresholds, replay regressions in CI, validate on held-out routes,
then move to controlled-course testing with independent safety supervision.

### “What did RL add?”

It explored hybrid control: keep an explainable geometric baseline and let a
small table choose one of three bounded corrections. The more important lesson
was experimental discipline—state/action design, reward shaping, exploration,
baseline comparison, distribution shift, and why simulator success is not a
safety argument.

### “What would you do differently?”

Define coordinate frames and acceptance tests first, create headless simulation
from day one, keep configuration outside source, run the tracker in both paths,
and establish a reliable Stanley-only baseline before adding learning.

## 12. A compact STAR story

**Situation:** I wanted to understand how perception, control, simulation, and
failure handling interact in an autonomy stack rather than build an isolated
lane detector.

**Task:** Build a modular lane-centring prototype that could run both closed
loop in simulation and over real recorded video, and make failures observable.

**Action:** I defined shared data contracts, implemented an OpenCV polynomial
lane detector, used Stanley control and a bicycle model, added a three-mode
health supervisor, deterministic sensor/actuator faults, telemetry and replay,
then experimented with a bounded residual Q policy and real-log alignment.

**Result:** I produced a reusable prototype and, more importantly, identified
where integration evidence is weak: camera calibration, confidence calibration,
path inconsistency, test coverage, dependency reproducibility, and sim-to-real
validity. My next milestone is a passing, metric-driven Stanley baseline across
a versioned scenario suite.

Do not invent performance numbers. Replace “Result” with measured results only
after the baseline is repaired and rerun.

## 13. Whiteboard walkthrough

If asked to draw the system, draw five boxes:

```text
Sensors -> Perception -> Health/State -> Control -> Vehicle
             |              |             |          |
             +-------------- Telemetry --------------+
```

Then add:

- coordinate frames and timestamps on arrows;
- faults at sensor and actuator interfaces;
- requested/applied commands around the actuator;
- a feedback arrow from vehicle to sensors;
- simulation/replay feeding the same module interfaces;
- acceptance metrics alongside the loop.

This demonstrates systems thinking beyond the individual algorithms.

## 14. Code-reading order

1. `autonomy/types.hpp`
2. `apps/run_stack.cpp`
3. `src/stanley_controller.cpp`
4. `src/vehicle_model.cpp`
5. `src/health_monitor.cpp`
6. `src/fault_injector.cpp`
7. `src/scene_renderer.cpp`
8. `src/lane_detector.cpp`
9. `src/lane_tracker.cpp`
10. `apps/run_video.cpp`
11. `src/q_learning_controller.cpp` and `apps/train_q_learning.cpp`
12. logger, replay, analysis, real-data scripts, and tests

For each file, answer: What is its contract? What assumptions does it make?
What happens on invalid/stale input? What unit and frame is every value in? How
would I prove it satisfies its requirement?

## 15. Final interview rule

Separate three levels of claim:

1. **Implemented:** the code path exists.
2. **Verified:** a repeatable test or metric supports it.
3. **Deployment-ready:** a safety and operational process supports it in the
   intended domain.

This repository contains meaningful implemented mechanisms and some component
verification. It is not deployment-ready. Being precise about that distinction
is one of the most Oxa-relevant things you can demonstrate.
