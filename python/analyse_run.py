#!/usr/bin/env python3

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(args.csv_path)

    absolute_error = (
        data["true_lateral_error_m"].abs()
    )

    departure_threshold_m = 1.75

    departure_frames = (
        absolute_error > departure_threshold_m
    ).sum()

    safe_stop_frames = (
        data["mode"] == "SAFE_STOP"
    ).sum()

    plt.figure(figsize=(10, 5))

    plt.plot(
        data["time_s"],
        data["true_lateral_error_m"],
        label="True lateral error",
    )

    plt.plot(
        data["time_s"],
        data["measured_lateral_error_m"],
        label="Measured lateral error",
        alpha=0.7,
    )

    plt.axhline(
        departure_threshold_m,
        linestyle="--",
        label="Lane boundary",
    )

    plt.axhline(
        -departure_threshold_m,
        linestyle="--",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Lateral error (m)")
    plt.title("Lane-keeping performance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        args.output_dir / "lateral_error.png",
        dpi=160,
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        data["time_s"],
        data["requested_steering_rad"],
        label="Requested steering",
    )
    plt.plot(
        data["time_s"],
        data["applied_steering_rad"],
        label="Applied steering",
        alpha=0.7,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle (rad)")
    plt.title("Controller commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "steering.png",
        dpi=160,
    )
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(
        data["time_s"],
        data["confidence"],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Perception confidence")
    plt.title("Lane-detection confidence")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "confidence.png",
        dpi=160,
    )
    plt.close()

if __name__ == "__main__":
    main()