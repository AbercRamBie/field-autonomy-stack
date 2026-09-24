#!/usr/bin/env python3
"""Align extracted lane observations with real comma2k19 CAN signals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align observation rows to comma2k19 camera, steering, and "
            "speed timestamps. No timestamp extrapolation is performed."
        )
    )
    parser.add_argument(
        "--observations",
        required=True,
        type=Path,
        help="CSV produced by extract_observations",
    )
    parser.add_argument(
        "--segment-dir",
        required=True,
        type=Path,
        help="comma2k19 segment directory containing global_pose and processed_log",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination aligned CSV",
    )
    return parser.parse_args()


def load_vector(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")

    values = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0:
        raise ValueError(f"{name} is empty: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values: {path}")

    return values


def validate_signal(
    timestamps: np.ndarray,
    values: np.ndarray,
    name: str,
) -> None:
    if timestamps.shape != values.shape:
        raise ValueError(
            f"{name} timestamp/value shape mismatch: "
            f"{timestamps.shape} versus {values.shape}"
        )

    intervals = np.diff(timestamps)
    if not np.all(intervals > 0.0):
        raise ValueError(f"{name} timestamps are not strictly increasing")


def locate_speed_directory(segment_dir: Path) -> Path:
    can_dir = segment_dir / "processed_log" / "CAN"

    for signal_name in ("speed", "car_speed"):
        candidate = can_dir / signal_name
        if (candidate / "t").is_file() and (candidate / "value").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find CAN speed signal under "
        f"{can_dir}; expected speed/ or car_speed/"
    )


def read_observations(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Observation CSV does not exist: {path}")

    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"Observation CSV has no header: {path}")

        required = {
            "frame_index",
            "tracked_valid",
            "tracked_fresh",
            "lateral_error_m",
            "heading_error_rad",
            "confidence",
        }
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(
                "Observation CSV is missing required columns: "
                + ", ".join(missing)
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Observation CSV contains no rows: {path}")

    frame_indices = np.asarray(
        [int(row["frame_index"]) for row in rows],
        dtype=np.int64,
    )
    expected = np.arange(len(rows), dtype=np.int64)

    if not np.array_equal(frame_indices, expected):
        raise ValueError(
            "Observation frame_index must be contiguous and start at zero"
        )

    return rows, list(fieldnames)


def nearest_skew_seconds(
    query_times: np.ndarray,
    signal_times: np.ndarray,
) -> np.ndarray:
    right = np.searchsorted(signal_times, query_times, side="left")
    right = np.clip(right, 0, signal_times.size - 1)
    left = np.clip(right - 1, 0, signal_times.size - 1)

    left_skew = np.abs(query_times - signal_times[left])
    right_skew = np.abs(signal_times[right] - query_times)
    return np.minimum(left_skew, right_skew)


def interpolate_without_extrapolation(
    query_times: np.ndarray,
    signal_times: np.ndarray,
    signal_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bracketed = (
        (query_times >= signal_times[0])
        & (query_times <= signal_times[-1])
    )

    interpolated = np.full(query_times.shape, np.nan, dtype=np.float64)
    interpolated[bracketed] = np.interp(
        query_times[bracketed],
        signal_times,
        signal_values,
    )

    return interpolated, bracketed


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return format(value, ".17g")


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile))


def align(
    observations_path: Path,
    segment_dir: Path,
    output_path: Path,
) -> None:
    rows, observation_fields = read_observations(observations_path)

    frame_times = load_vector(
        segment_dir / "global_pose" / "frame_times",
        "camera frame timestamps",
    )

    if frame_times.size != len(rows):
        raise ValueError(
            "Camera timestamp count does not match observation count: "
            f"{frame_times.size} versus {len(rows)}"
        )
    if not np.all(np.diff(frame_times) > 0.0):
        raise ValueError("Camera frame timestamps are not strictly increasing")

    steering_dir = (
        segment_dir
        / "processed_log"
        / "CAN"
        / "steering_angle"
    )
    steering_times = load_vector(steering_dir / "t", "steering timestamps")
    steering_values = load_vector(
        steering_dir / "value",
        "steering values",
    )
    validate_signal(steering_times, steering_values, "steering")

    speed_dir = locate_speed_directory(segment_dir)
    speed_times = load_vector(speed_dir / "t", "speed timestamps")
    speed_values = load_vector(speed_dir / "value", "speed values")
    validate_signal(speed_times, speed_values, "speed")

    steering, steering_bracketed = interpolate_without_extrapolation(
        frame_times,
        steering_times,
        steering_values,
    )
    speed, speed_bracketed = interpolate_without_extrapolation(
        frame_times,
        speed_times,
        speed_values,
    )

    steering_skew_ms = 1000.0 * nearest_skew_seconds(
        frame_times,
        steering_times,
    )
    speed_skew_ms = 1000.0 * nearest_skew_seconds(
        frame_times,
        speed_times,
    )

    can_bracketed = steering_bracketed & speed_bracketed

    added_fields: Sequence[str] = (
        "frame_time_s",
        "steering_angle_deg",
        "speed_mps",
        "steering_nearest_skew_ms",
        "speed_nearest_skew_ms",
        "can_bracketed",
    )
    output_fields = ["frame_index", *added_fields]
    output_fields.extend(
        field for field in observation_fields if field != "frame_index"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fields)
        writer.writeheader()

        for index, source_row in enumerate(rows):
            output_row = dict(source_row)
            output_row.update(
                {
                    "frame_time_s": format_float(frame_times[index]),
                    "steering_angle_deg": format_float(steering[index]),
                    "speed_mps": format_float(speed[index]),
                    "steering_nearest_skew_ms": format_float(
                        steering_skew_ms[index]
                    ),
                    "speed_nearest_skew_ms": format_float(
                        speed_skew_ms[index]
                    ),
                    "can_bracketed": str(int(can_bracketed[index])),
                }
            )
            writer.writerow(output_row)

    valid_skew = can_bracketed

def main() -> None:
    args = parse_args()
    align(args.observations, args.segment_dir, args.output)


if __name__ == "__main__":
    main()
