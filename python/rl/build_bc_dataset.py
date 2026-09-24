#!/usr/bin/env python3
"""Build a behavior-cloning dataset from aligned real driving data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


OUTPUT_FIELDS = [
    "route_id",
    "frame_index",
    "frame_time_s",
    "lateral_error_m",
    "heading_error_rad",
    "speed_mps",
    "confidence",
    "previous_front_wheel_rad",
    "delta_lateral_error_m",
    "delta_heading_error_rad",
    "stanley_steering_rad",
    "target_front_wheel_rad",
    "target_residual_rad",
    "steering_nearest_skew_ms",
    "speed_nearest_skew_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build behavior-cloning samples from a real, "
            "timestamp-aligned route."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Aligned observation CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination BC CSV",
    )

    parser.add_argument(
        "--route-id",
        required=True,
        help="Stable route/segment identifier",
    )

    parser.add_argument(
        "--steering-ratio",
        required=True,
        type=float,
        help="Steering-wheel/front-wheel ratio",
    )

    parser.add_argument(
        "--steering-sign",
        required=True,
        type=float,
        choices=(-1.0, 1.0),
        help="Sign mapping from CAN angle to project steering",
    )

    parser.add_argument(
        "--maximum-can-skew-ms",
        required=True,
        type=float,
        help="Maximum permitted nearest CAN timestamp skew",
    )

    parser.add_argument(
        "--maximum-frame-gap-ms",
        required=True,
        type=float,
        help="Maximum permitted gap between adjacent camera frames",
    )

    parser.add_argument(
        "--stanley-lateral-gain",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--stanley-speed-softening",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--maximum-steering-rad",
        type=float,
        default=0.45,
    )

    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parse_flag(row: dict[str, str], name: str) -> bool:
    return bool(int(float(row[name])))


def parse_float(row: dict[str, str], name: str) -> float:
    text = row[name].strip()

    if not text:
        raise ValueError(f"Empty numeric field: {name}")

    value = float(text)

    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric field: {name}")

    return value


def format_float(value: float) -> str:
    return format(value, ".17g")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Aligned observation CSV does not exist: {path}"
        )

    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        required = {
            "frame_index",
            "frame_time_s",
            "can_bracketed",
            "tracked_valid",
            "tracked_fresh",
            "lateral_error_m",
            "heading_error_rad",
            "confidence",
            "steering_angle_deg",
            "speed_mps",
            "steering_nearest_skew_ms",
            "speed_nearest_skew_ms",
        }

        missing = sorted(
            required.difference(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Missing input columns: "
                + ", ".join(missing)
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV contains no rows")

    frame_indices = [
        int(row["frame_index"])
        for row in rows
    ]

    if frame_indices != list(range(len(rows))):
        raise ValueError(
            "frame_index must be contiguous and start at zero"
        )

    return rows


def quality_failure(
    row: dict[str, str],
    maximum_can_skew_ms: float,
) -> str | None:
    if not parse_flag(row, "can_bracketed"):
        return "can_unbracketed"

    if not parse_flag(row, "tracked_valid"):
        return "tracker_invalid"

    if not parse_flag(row, "tracked_fresh"):
        return "tracker_held"

    try:
        steering_skew_ms = parse_float(
            row,
            "steering_nearest_skew_ms",
        )

        speed_skew_ms = parse_float(
            row,
            "speed_nearest_skew_ms",
        )

        parse_float(row, "lateral_error_m")
        parse_float(row, "heading_error_rad")
        parse_float(row, "confidence")
        parse_float(row, "steering_angle_deg")
        parse_float(row, "speed_mps")

    except ValueError:
        return "invalid_numeric_value"

    if (
        steering_skew_ms > maximum_can_skew_ms
        or speed_skew_ms > maximum_can_skew_ms
    ):
        return "can_skew_exceeded"

    return None


def front_wheel_angle_rad(
    steering_angle_deg: float,
    steering_ratio: float,
    steering_sign: float,
) -> float:
    return (
        steering_sign
        * math.radians(steering_angle_deg)
        / steering_ratio
    )


def stanley_steering(
    lateral_error_m: float,
    heading_error_rad: float,
    speed_mps: float,
    lateral_gain: float,
    speed_softening: float,
    maximum_steering_rad: float,
) -> float:
    lateral_correction = math.atan2(
        lateral_gain * lateral_error_m,
        speed_mps + speed_softening,
    )

    steering = -(
        heading_error_rad
        + lateral_correction
    )

    return float(
        np.clip(
            steering,
            -maximum_steering_rad,
            maximum_steering_rad,
        )
    )


def build_dataset(args: argparse.Namespace) -> None:
    if args.steering_ratio <= 0.0:
        raise ValueError(
            "--steering-ratio must be positive"
        )

    if args.maximum_can_skew_ms <= 0.0:
        raise ValueError(
            "--maximum-can-skew-ms must be positive"
        )

    if args.maximum_frame_gap_ms <= 0.0:
        raise ValueError(
            "--maximum-frame-gap-ms must be positive"
        )

    if args.stanley_lateral_gain <= 0.0:
        raise ValueError(
            "--stanley-lateral-gain must be positive"
        )

    if args.stanley_speed_softening <= 0.0:
        raise ValueError(
            "--stanley-speed-softening must be positive"
        )

    if args.maximum_steering_rad <= 0.0:
        raise ValueError(
            "--maximum-steering-rad must be positive"
        )

    rows = read_rows(args.input)

    samples: list[dict[str, str]] = []
    rejected: Counter[str] = Counter()

    # A BC state uses the immediately preceding real frame,
    # so the first row cannot produce a sample.
    rejected["first_frame"] = 1

    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]

        previous_failure = quality_failure(
            previous,
            args.maximum_can_skew_ms,
        )

        if previous_failure is not None:
            rejected[
                f"previous_{previous_failure}"
            ] += 1
            continue

        current_failure = quality_failure(
            current,
            args.maximum_can_skew_ms,
        )

        if current_failure is not None:
            rejected[current_failure] += 1
            continue

        previous_time_s = parse_float(
            previous,
            "frame_time_s",
        )

        current_time_s = parse_float(
            current,
            "frame_time_s",
        )

        frame_gap_ms = (
            current_time_s - previous_time_s
        ) * 1000.0

        if (
            frame_gap_ms <= 0.0
            or frame_gap_ms >
                args.maximum_frame_gap_ms
        ):
            rejected["frame_gap_exceeded"] += 1
            continue

        previous_lateral = parse_float(
            previous,
            "lateral_error_m",
        )

        current_lateral = parse_float(
            current,
            "lateral_error_m",
        )

        previous_heading = parse_float(
            previous,
            "heading_error_rad",
        )

        current_heading = parse_float(
            current,
            "heading_error_rad",
        )

        speed_mps = parse_float(
            current,
            "speed_mps",
        )

        confidence = parse_float(
            current,
            "confidence",
        )

        previous_steering_deg = parse_float(
            previous,
            "steering_angle_deg",
        )

        current_steering_deg = parse_float(
            current,
            "steering_angle_deg",
        )

        previous_front_wheel = (
            front_wheel_angle_rad(
                previous_steering_deg,
                args.steering_ratio,
                args.steering_sign,
            )
        )

        current_front_wheel = (
            front_wheel_angle_rad(
                current_steering_deg,
                args.steering_ratio,
                args.steering_sign,
            )
        )

        nominal_steering = stanley_steering(
            lateral_error_m=current_lateral,
            heading_error_rad=current_heading,
            speed_mps=speed_mps,
            lateral_gain=args.stanley_lateral_gain,
            speed_softening=(
                args.stanley_speed_softening
            ),
            maximum_steering_rad=(
                args.maximum_steering_rad
            ),
        )

        residual_action = (
            current_front_wheel
            - nominal_steering
        )

        sample = {
            "route_id": args.route_id,
            "frame_index": current["frame_index"],
            "frame_time_s": format_float(
                current_time_s
            ),
            "lateral_error_m": format_float(
                current_lateral
            ),
            "heading_error_rad": format_float(
                current_heading
            ),
            "speed_mps": format_float(speed_mps),
            "confidence": format_float(confidence),
            "previous_front_wheel_rad": (
                format_float(previous_front_wheel)
            ),
            "delta_lateral_error_m": format_float(
                current_lateral
                - previous_lateral
            ),
            "delta_heading_error_rad": format_float(
                current_heading
                - previous_heading
            ),
            "stanley_steering_rad": format_float(
                nominal_steering
            ),
            "target_front_wheel_rad": format_float(
                current_front_wheel
            ),
            "target_residual_rad": format_float(
                residual_action
            ),
            "steering_nearest_skew_ms": (
                current[
                    "steering_nearest_skew_ms"
                ]
            ),
            "speed_nearest_skew_ms": (
                current[
                    "speed_nearest_skew_ms"
                ]
            ),
        }

        samples.append(sample)

    if not samples:
        raise ValueError(
            "No BC samples passed the quality rules"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(samples)

    front_actions = np.asarray(
        [
            float(sample["target_front_wheel_rad"])
            for sample in samples
        ],
        dtype=np.float64,
    )

    stanley_actions = np.asarray(
        [
            float(sample["stanley_steering_rad"])
            for sample in samples
        ],
        dtype=np.float64,
    )

    residual_actions = np.asarray(
        [
            float(sample["target_residual_rad"])
            for sample in samples
        ],
        dtype=np.float64,
    )

    manifest = {
        "schema_version": 1,
        "task": "behavior_cloning",
        "real_data_only": True,
        "route_id": args.route_id,
        "source_csv": str(
            args.input.resolve()
        ),
        "source_sha256": file_sha256(args.input),
        "output_csv": str(
            args.output.resolve()
        ),
        "row_count": len(samples),
        "steering_ratio": args.steering_ratio,
        "steering_sign": args.steering_sign,
        "maximum_can_skew_ms": (
            args.maximum_can_skew_ms
        ),
        "maximum_frame_gap_ms": (
            args.maximum_frame_gap_ms
        ),
        "stanley": {
            "lateral_gain": (
                args.stanley_lateral_gain
            ),
            "speed_softening": (
                args.stanley_speed_softening
            ),
            "maximum_steering_rad": (
                args.maximum_steering_rad
            ),
        },
        "rejected_rows": dict(rejected),
    }

    manifest_path = args.output.with_suffix(
        ".manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Input rows: {len(rows)}")
    print(f"BC samples written: {len(samples)}")
    print("Rejected rows:")

    for reason, count in sorted(rejected.items()):
        print(f"  {reason}: {count}")

    for name, values in (
        ("front-wheel action", front_actions),
        ("Stanley action", stanley_actions),
        ("residual action", residual_actions),
    ):
        print(f"{name} rad:")
        print(
            "  "
            f"min={float(np.min(values)):.6f} "
            f"p01={float(np.percentile(values, 1)):.6f} "
            f"p50={float(np.percentile(values, 50)):.6f} "
            f"p99={float(np.percentile(values, 99)):.6f} "
            f"max={float(np.max(values)):.6f}"
        )

    residual_over_limit = np.mean(
        np.abs(residual_actions) > 0.05
    )

    print(
        "Residual actions outside ±0.05 rad: "
        f"{100.0 * residual_over_limit:.2f}%"
    )

    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()