# Offline RL implementation guide (real data only)

## Decision

You can proceed with reinforcement learning, but the correct form for this
project is **offline reinforcement learning from real, time-synchronised driving
logs**. Do not train from `run_stack`, `SceneRenderer`, `Track`,
`VehicleModel`, injected faults, generated images, or generated trajectories.

The video currently in `data/videos/` is useful for perception regression only.
It is about 30 seconds long and has no synchronised measured speed, steering,
episode boundaries, or outcome signal. A steering command calculated while
replaying it cannot change the next frame. It is therefore not an RL training
dataset.

The recommended first policy is not end-to-end image-to-steering and is not the
placeholder tabular Q-learning program. Use a continuous-action offline method
to learn a small **residual correction around `StanleyController`**:

```text
real camera + CAN log
        |
        +--> LaneDetector --> LaneTracker --> numeric state
        |                                      |
        +--> measured speed/steering ----------+--> offline dataset
                                                       |
                                            BC baseline, then TD3+BC

runtime:

LaneObservation --> StanleyController --> nominal steering
        |                                      |
        +--> offline-RL policy --> bounded residual
                                               |
                          safety gate + clamp + rate limit --> request
```

This is a research path, not evidence that a learned controller is safe for a
public-road vehicle. Finish with shadow-mode and controlled-course validation;
do not jump from an offline score to actuation.

## What "real data only" means here

Allowed:

- recorded camera frames from a physical vehicle;
- recorded steering, speed, IMU, pose, and CAN signals from that same drive;
- lane/error measurements calculated from those real frames;
- rewards calculated from measured consecutive states;
- ordinary scaling, filtering, timestamp alignment, and train/validation
  splitting;
- replaying held-out real logs without changing their frames.

Excluded:

- `run_stack` output or any `SceneRenderer` frame;
- simulated vehicle transitions or generated road geometry;
- generated, flipped, relit, noised, or otherwise fabricated images;
- synthetic fault injection;
- mixing a real observation with a counterfactual next state produced by
  `VehicleModel`.

Derived lane errors and rewards are measurements/labels computed from real
records; they do not create new driving samples. Keep the raw record and the
derivation version so every training row is auditable.

## Data choice

### Recommended starting dataset: comma2k19

[comma2k19](https://github.com/commaai/comma2k19) is the closest public match
to this lane-control stack. It contains more than 33 hours of real highway
driving, road-facing video, timestamped vehicle speed, steering angle, IMU,
CAN, and camera pose. The download is about 100 GB, split into chunks. Start
with the repository's one-minute example to validate the importer, then use
complete routes from the full dataset.

Important limitations:

- it is primarily one California highway corridor and two vehicle platforms,
  so it is not evidence of broad generalisation;
- its `steering_angle` must not be assumed to equal this project's front-wheel
  angle in radians; establish sign, units, steering ratio, offset, and delay for
  each vehicle from the logs/DBC;
- this detector's perspective transform is calibrated for a different camera.
  Calibrate it using real comma2k19 frames before extracting states;
- check the dataset and source-code licence terms yourself for the intended use
  and record the exact dataset version/hash in the experiment manifest.

For a later planning project, [nuPlan](https://github.com/motional/nuplan-devkit)
provides more than 1,300 hours of real driving logs and a planning benchmark.
It is much larger and its task/interfaces are less aligned with this small
lane-steering controller, so it should not be the first integration.

The best domain match is eventually your own physical platform's passive log.
Record before taking control: monotonic timestamp, unmodified camera frame,
measured speed, measured steering/actuator feedback, yaw rate, drive/route ID,
manual/autonomous mode, and disengagement/e-stop flags. Collection must comply
with local law, privacy requirements, consent, and the vehicle's safety
process.

## Define the task before writing the trainer

Keep version 1 deliberately narrow:

- task: lane centring on roads represented by the current detector;
- longitudinal control: outside RL; retain the health monitor's speed policy;
- control rate: 20 Hz after timestamp alignment;
- policy output: one continuous residual steering value;
- nominal controller: the existing `StanleyController`;
- maximum RL residual: start with `0.05 rad` at the front wheel;
- final steering: clamp to the existing `[-0.45, 0.45] rad` range and apply a
  separately configured rate limit;
- invalid/low-confidence perception: do not query RL; use the existing
  degraded/safe-stop logic and the approved fallback controller.

Do not make speed a learned action in version 1. Steering-only data and reward
design are already substantial; combined lateral/longitudinal control makes
coverage and safety analysis much harder.

## MDP contract

Use a vector state derived from the real log rather than raw pixels initially.
At timestamp `t`:

```text
s_t = [
  lateral_error_m,
  heading_error_rad,
  speed_mps,
  confidence,
  previous_applied_steering_rad,
  delta_lateral_error_m,
  delta_heading_error_rad
]

a_t          = measured_front_wheel_angle_t - stanley(s_t)
policy(a_t)  is clipped to [-0.05, +0.05] rad
transition   = (s_t, a_t, r_t, s_(t+1), terminal, timeout)
```

Using residual actions gives the learned policy a smaller, easier-to-audit
action space. It also makes Stanley the explicit fallback. If reliable
front-wheel angle cannot be recovered, stop: steering-wheel degrees and
front-wheel radians are not interchangeable training targets.

Use the next measured state for the reward so action and outcome are aligned:

```text
r_t = clip(
    1.0
    - (lateral_error_(t+1) / 1.75)^2
    - 0.25 * (heading_error_(t+1) / 0.35)^2
    - 0.10 * ((steering_t - steering_(t-1)) / 0.45)^2,
    -5.0,
    +1.0
)
```

These constants are an initial, versioned specification, not ground truth.
Report each reward component separately so a policy cannot appear better only
because one weight dominates. Do not call a detector miss a lane departure:
close the usable sequence and mark the log boundary as a timeout. A terminal
must represent a real recorded terminal event with a defensible definition,
such as a recorded safety intervention; end-of-file is a timeout, not a
terminal.

## Dataset builder

Add a separate real-data pipeline; do not overload `CsvLogger`, whose current
schema describes the synthetic loop.

Proposed files:

```text
python/rl/
  import_comma2k19.py       align video/CAN and write an index
  build_transitions.py      construct states, residual actions and rewards
  validate_dataset.py       leakage, units, timestamp and coverage checks
  train_bc.py               mandatory behaviour-cloning baseline
  train_td3_bc.py           first offline-RL experiment
  evaluate_offline.py       held-out metrics and policy support checks
  export_policy.py          ONNX export plus manifest

apps/
  extract_real_observations.cpp  run detector/tracker on indexed real frames
  replay_rl_policy.cpp           shadow replay; never changes the video

autonomy/
  rl_policy.hpp             inference-only interface
  safety_steering_filter.hpp
```

Write one immutable, columnar transition file per route (Parquet is preferable)
and a manifest containing:

- dataset source/version and route IDs;
- raw-file checksums;
- camera calibration version;
- signal units, sign convention, steering ratio, and time offset;
- detector/tracker Git commit;
- state/action/reward schema version;
- extraction command and rejected-row counts.

Minimum row fields:

```text
route_id, timestamp_ns, next_timestamp_ns,
lateral_error_m, heading_error_rad, speed_mps, confidence,
previous_steering_rad, delta_lateral_error_m, delta_heading_error_rad,
stanley_steering_rad, measured_steering_rad, residual_action_rad,
next_lateral_error_m, next_heading_error_rad,
reward_lane, reward_heading, reward_smoothness, reward_total,
terminal, timeout, frame_path_or_id
```

### Synchronisation and quality rules

1. Use monotonic/source timestamps, never frame number alone.
2. Choose one 20 Hz decision grid and document interpolation for each signal.
3. Estimate camera-to-steering delay on training routes only. Apply the frozen
   value to validation/test routes.
4. Reject transitions with non-finite values, non-monotonic time, an excessive
   camera/CAN skew, missing steering calibration, or invalid lane geometry.
5. Break an episode at a route boundary, dropped interval, calibration change,
   manual/autonomous mode change, or long perception gap.
6. Never interpolate across an episode boundary.
7. Plot steering versus yaw-rate/heading change on several left and right turns
   to verify sign and delay manually.

Pick the permitted skew from the source timestamp resolution and measured log
jitter; do not silently invent a universal threshold.

## Split without leakage

Split by complete route before fitting scalers, delays, reward normalization,
or models. Never randomly split adjacent frames. Consecutive frames are nearly
duplicates and would make validation misleading.

A reasonable first split is 70% train, 15% validation, and 15% test by route,
stratified as far as the data permits by vehicle and conditions. Freeze the
test route list in the manifest. Report results per route and per vehicle, not
only a global average.

Before training, the validator must show:

- transition and episode counts for every split;
- valid-perception percentage and rejection reasons;
- state/action min, max, quantiles, and missing-value counts;
- action coverage conditional on lateral/heading-error bins;
- no route or raw-frame checksum shared between splits;
- no fitted preprocessing statistic derived from validation/test data.

If the logs contain almost only centred, smooth driving, they do not contain
evidence for recovery behaviour. An offline algorithm cannot safely infer
unseen recovery actions. Narrow the operating domain or collect additional
real, safely supervised recovery examples; do not fill the gap synthetically.

## Train in two gates

Use Python/PyTorch for training and keep C++ for perception and runtime
inference. [d3rlpy](https://d3rlpy.readthedocs.io/en/stable/) provides offline
RL algorithms and an MDP dataset abstraction. Pin the exact Python and package
versions in a lock file after the first working environment.

### Gate 1: behaviour cloning

First train the network to reproduce the logged residual action. This verifies
the full data path, sign convention, scaling, split, and export. If BC cannot
predict held-out actions better than constant-zero residual and Stanley-only
baselines, do not start RL.

Required comparisons on held-out routes:

- zero-residual (Stanley only);
- mean/constant residual;
- behaviour cloning;
- later, TD3+BC.

Measure residual MAE/RMSE, full-steering MAE, sign disagreement, steering-rate
violations, saturation rate, and metrics by error/speed bin.

### Gate 2: TD3+BC

Use TD3+BC first because steering is continuous and its behaviour-cloning term
discourages unsupported actions. CQL is a useful second algorithm, not a
substitute for the baseline or data checks.

An indicative d3rlpy skeleton is:

```python
import d3rlpy
from d3rlpy.algos import TD3PlusBCConfig
from d3rlpy.preprocessing import (
    MinMaxActionScaler,
    StandardObservationScaler,
)

dataset = d3rlpy.dataset.MDPDataset(
    observations=train_states.astype("float32"),
    actions=train_residuals.astype("float32").reshape(-1, 1),
    rewards=train_rewards.astype("float32"),
    terminals=train_terminals.astype("float32"),
    timeouts=train_timeouts.astype("float32"),
)

policy = TD3PlusBCConfig(
    observation_scaler=StandardObservationScaler(),
    action_scaler=MinMaxActionScaler(),
).create(device="cuda:0")

policy.fit(dataset, n_steps=500_000)
policy.save_model("artifacts/td3_bc.d3")
policy.save_policy("artifacts/td3_bc.onnx")
```

Treat the step count as a starting experiment, not a promised optimum. Run at
least three fixed seeds and preserve configuration, learning curves, artifact
hash, and dataset-manifest hash. The current d3rlpy API can export a greedy
policy as ONNX, which is suitable for a later C++ ONNX Runtime adapter.

## Offline evaluation

There is no trustworthy single "accuracy" number for an offline steering
policy. Use all of the following:

1. **Action agreement:** compare with recorded steering, while recognising that
   disagreement is not automatically wrong.
2. **Support check:** flag every predicted residual outside the local training
   action distribution for similar states. Report nearest-neighbour distance
   and residual saturation.
3. **Reward-component replay:** report rewards on recorded transitions only;
   do not pretend the log contains the next state caused by the new action.
4. **Off-policy evaluation:** use fitted-Q evaluation only as a comparative,
   uncertainty-bearing estimate. It is not a safety certificate.
5. **Ablations:** compare state without deltas, residual limits, reward weights,
   and BC versus TD3+BC on exactly the frozen routes.
6. **Robustness on real subsets:** separately report curves, high/low speed,
   shadows, worn markings, and detector-confidence bands when those conditions
   actually occur in the logs.

Reject a model that produces non-finite output, breaches the residual bound,
depends on test-fitted preprocessing, performs worse than BC on primary held-out
metrics, or frequently selects actions outside logged support.

## Integrate without giving it authority

Export the observation scaler and action postprocessing inside the model where
the library supports it, and record independent test vectors:

```text
input state -> expected ONNX residual -> expected final clamped steering
```

The C++ wrapper must:

- validate the exact seven-element input order and model/schema version;
- reject non-finite state or output;
- clamp the residual independently of the network;
- combine it with Stanley output using the documented sign convention;
- apply absolute steering and steering-rate limits;
- return immediately to Stanley/degraded/safe-stop behavior on low confidence,
  stale data, inference timeout, model error, or health-state transition;
- log nominal, residual, final request, applied feedback, gate reason, model
  hash, and inference latency.

First add the policy to real-video replay in **shadow mode**: calculate and log
its output, but keep it disconnected from all actuation. `run_video` remains an
open-loop perception/policy check and must not be described as closed-loop
validation.

## Real-world validation ladder

Proceed only when the previous stage has a written pass/fail report:

1. Held-out real-log replay with no actuation.
2. Passive shadow logging on the target physical platform with the approved
   human/baseline controller still in control.
3. Review disagreements, support violations, timing, sign, saturation, stale
   inputs, and fallback transitions.
4. Low-speed testing on a closed, access-controlled course with an independent
   e-stop, safety driver/operator, geofence, steering/speed limits, and a tested
   fallback path.
5. Gradually expand only within conditions represented by validated real data.

Define the operational design domain, hazard analysis, operator procedure,
abort conditions, and responsibility with the vehicle project's safety owner
before any actuation. Public-road deployment is outside what an offline model
and this repository can validate.

## End-to-end checklist

- [ ] Freeze steering-only residual task and real-data-only policy.
- [ ] Run the comma2k19 one-minute example through an importer.
- [ ] Calibrate the camera transform on real frames.
- [ ] Resolve steering units, ratio, sign, offset, and time delay.
- [ ] Extract detector/tracker observations from real frames.
- [ ] Build route-separated transitions and an immutable manifest.
- [ ] Pass dataset leakage, quality, and action-coverage checks.
- [ ] Train and beat zero-residual/Stanley and BC baselines appropriately.
- [ ] Train TD3+BC with multiple seeds; keep CQL as a comparison.
- [ ] Complete held-out offline evaluation and support analysis.
- [ ] Export ONNX with golden C++/Python parity vectors.
- [ ] Add independent C++ clamps, rate limits, health gates, and logging.
- [ ] Pass real-log replay and target-platform shadow mode.
- [ ] Obtain the project's safety approval for controlled-course testing.
- [ ] Complete closed-course tests before considering broader authority.

## Immediate next milestone

Do not implement the RL controller first. The next milestone is a **data
readiness report** from the comma2k19 example route containing:

1. 100 synchronised real transitions with source timestamps;
2. lane detector/tracker validity and calibration plots;
3. confirmed steering sign/units/delay;
4. the proposed seven state values, Stanley output, measured steering, and
   residual action;
5. a manifest and a validator that proves no split leakage.

Once this passes, implement BC. Only after BC and replay pass should
`train_q_learning` be replaced or retired in favour of the offline trainer.

## References

- [comma2k19 repository and data format](https://github.com/commaai/comma2k19)
- [nuPlan devkit](https://github.com/motional/nuplan-devkit)
- [d3rlpy offline RL documentation](https://d3rlpy.readthedocs.io/en/stable/)
- [d3rlpy MDP dataset design](https://d3rlpy.readthedocs.io/en/stable/software_design.html)
- [d3rlpy algorithm reference](https://d3rlpy.readthedocs.io/en/stable/references/algos.html)

