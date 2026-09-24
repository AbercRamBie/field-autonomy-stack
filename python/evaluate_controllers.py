#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"No telemetry rows in {path}")

    return rows


def calculate_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    lateral_errors = [
        abs(float(row["true_lateral_error_m"]))
        for row in rows
    ]

    steering = [
        float(row["applied_steering_rad"])
        for row in rows
    ]

    steering_changes = [
        abs(current - previous)
        for previous, current in zip(
            steering,
            steering[1:],
        )
    ]

    safe_stop_frames = sum(
        row["mode"] == "SAFE_STOP"
        for row in rows
    )

    lane_departure_frames = sum(
        error > 1.75
        for error in lateral_errors
    )

    times = [
        float(row["time_s"])
        for row in rows
    ]

    timestep = (
        times[1] - times[0]
        if len(times) > 1
        else 0.0
    )

    return {
        "mae_m": sum(lateral_errors) / len(lateral_errors),
        "rmse_m": math.sqrt(
            sum(error**2 for error in lateral_errors)
            / len(lateral_errors)
        ),
        "maximum_error_m": max(lateral_errors),
        "mean_steering_change_rad": (
            sum(steering_changes) / len(steering_changes)
            if steering_changes
            else 0.0
        ),
        "safe_stop_seconds": safe_stop_frames * timestep,
        "lane_departure_frames": float(lane_departure_frames),
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Stanley and residual-Q telemetry."
    )

    parser.add_argument(
        "--stanley",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--residual-q",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    stanley_metrics = calculate_metrics(
        load_rows(args.stanley)
    )

    residual_q_metrics = calculate_metrics(
        load_rows(args.residual_q)
    )

if __name__ == "__main__":
    main()