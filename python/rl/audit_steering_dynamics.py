#!/usr/bin/env python3
"""Audit comma2k19 steering sign and response delay against real IMU yaw."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure correlation between recorded steering angle and IMU "
            "yaw rate over a configurable lag range."
        )
    )
    parser.add_argument("--segment-dir", required=True, type=Path)
    parser.add_argument("--minimum-lag-ms", type=int, default=-250)
    parser.add_argument("--maximum-lag-ms", type=int, default=250)
    parser.add_argument("--lag-step-ms", type=int, default=10)
    parser.add_argument(
        "--gyro-axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Gyro value column; comma2k19 uses [forward, right, down]",
    )
    return parser.parse_args()


def load_vector(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")

    values = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid {name}: {path}")

    return values


def load_gyro_values(path: Path, axis: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing gyro values: {path}")

    values = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.float64,
    )

    if values.ndim != 2 or values.shape[1] <= axis:
        raise ValueError(
            f"Expected two-dimensional gyro values with axis {axis}: {path}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Gyro values contain non-finite entries: {path}")

    return values[:, axis]


def validate_time_series(
    timestamps: np.ndarray,
    values: np.ndarray,
    name: str,
) -> None:
    if timestamps.shape != values.shape:
        raise ValueError(
            f"{name} timestamp/value shape mismatch: "
            f"{timestamps.shape} versus {values.shape}"
        )
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError(f"{name} timestamps are not strictly increasing")


def main() -> None:
    args = parse_args()

    if args.lag_step_ms <= 0:
        raise ValueError("--lag-step-ms must be positive")
    if args.minimum_lag_ms > args.maximum_lag_ms:
        raise ValueError("minimum lag must not exceed maximum lag")

    steering_dir = (
        args.segment_dir
        / "processed_log"
        / "CAN"
        / "steering_angle"
    )
    gyro_dir = args.segment_dir / "processed_log" / "IMU" / "gyro"

    steering_times = load_vector(steering_dir / "t", "steering timestamps")
    steering_values = load_vector(steering_dir / "value", "steering values")
    validate_time_series(steering_times, steering_values, "steering")

    gyro_times = load_vector(gyro_dir / "t", "gyro timestamps")
    gyro_values = load_gyro_values(gyro_dir / "value", args.gyro_axis)
    validate_time_series(gyro_times, gyro_values, "gyro")

    results: list[tuple[float, int, float, int]] = []

    for lag_ms in range(
        args.minimum_lag_ms,
        args.maximum_lag_ms + 1,
        args.lag_step_ms,
    ):
        query_times = steering_times + lag_ms / 1000.0
        usable = (
            (query_times >= gyro_times[0])
            & (query_times <= gyro_times[-1])
        )

        if np.sum(usable) < 2:
            continue

        aligned_gyro = np.interp(
            query_times[usable],
            gyro_times,
            gyro_values,
        )
        correlation = float(
            np.corrcoef(steering_values[usable], aligned_gyro)[0, 1]
        )

        if np.isfinite(correlation):
            results.append(
                (
                    abs(correlation),
                    lag_ms,
                    correlation,
                    int(np.sum(usable)),
                )
            )

    if not results:
        raise ValueError("No valid lag correlations could be calculated")

    results.sort(reverse=True)
    _, best_lag_ms, best_correlation, sample_count = results[0]

    relation = "same" if best_correlation > 0.0 else "opposite"

if __name__ == "__main__":
    main()
