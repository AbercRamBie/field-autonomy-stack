#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process one real comma2k19 segment into aligned observations "
            "and behavior-cloning samples."
        )
    )

    parser.add_argument(
        "--segment-dir",
        type=Path,
        required=True,
        help="Segment containing video.hevc, global_pose, and processed_log.",
    )
    parser.add_argument(
        "--route-id",
        required=True,
        help="Unique route/segment identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated CSV and manifest files.",
    )
    parser.add_argument(
        "--steering-ratio",
        type=float,
        required=True,
        help="Vehicle steering ratio, for example 16.88 for the RAV4.",
    )
    parser.add_argument(
        "--steering-sign",
        type=int,
        choices=(-1, 1),
        required=True,
        help="Mapping sign from CAN steering angle to front-wheel angle.",
    )
    parser.add_argument(
        "--maximum-can-skew-ms",
        type=float,
        default=10.0,
        help="Maximum permitted timestamp skew for CAN alignment.",
    )
    parser.add_argument(
        "--maximum-frame-gap-ms",
        type=float,
        default=75.0,
        help="Maximum permitted time between consecutive fresh observations.",
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=PROJECT_ROOT / "build" / "extract_observations",
        help="Path to the compiled observation extractor.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow existing generated files to be overwritten.",
    )

    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {description}: {path}")


def run_stage(name: str, command: list[str]) -> None:
    print(f"\n[{name}]")
    print(shlex.join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"\nStage '{name}' failed with exit code {error.returncode}."
        ) from error


def main() -> None:
    args = parse_args()

    segment_dir = args.segment_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extractor = args.extractor.expanduser().resolve()

    video_path = segment_dir / "video.hevc"
    frame_times_path = segment_dir / "global_pose" / "frame_times"

    align_script = PROJECT_ROOT / "python" / "rl" / "align_comma2k19.py"
    dataset_script = PROJECT_ROOT / "python" / "rl" / "build_bc_dataset.py"

    require_file(video_path, "segment video")
    require_file(frame_times_path, "frame timestamp file")
    require_file(extractor, "extract_observations executable")
    require_file(align_script, "alignment script")
    require_file(dataset_script, "BC dataset builder")

    observations_path = output_dir / "observations.csv"
    aligned_path = output_dir / "aligned_observations.csv"
    samples_path = output_dir / "bc_samples.csv"
    manifest_path = samples_path.with_suffix(".manifest.json")

    generated_files = [
        observations_path,
        aligned_path,
        samples_path,
        manifest_path,
    ]

    existing_files = [path for path in generated_files if path.exists()]
    if existing_files and not args.force:
        formatted = "\n".join(f"  {path}" for path in existing_files)
        raise SystemExit(
            "The following output files already exist:\n"
            f"{formatted}\n"
            "Use a different --output-dir or pass --force."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    run_stage(
        "1/3 Extract lane observations",
        [
            str(extractor),
            str(video_path),
            str(observations_path),
        ],
    )

    run_stage(
        "2/3 Align observations with real CAN data",
        [
            sys.executable,
            str(align_script),
            "--observations",
            str(observations_path),
            "--segment-dir",
            str(segment_dir),
            "--output",
            str(aligned_path),
        ],
    )

    run_stage(
        "3/3 Build behavior-cloning dataset",
        [
            sys.executable,
            str(dataset_script),
            "--input",
            str(aligned_path),
            "--output",
            str(samples_path),
            "--route-id",
            args.route_id,
            "--steering-ratio",
            str(args.steering_ratio),
            "--steering-sign",
            str(args.steering_sign),
            "--maximum-can-skew-ms",
            str(args.maximum_can_skew_ms),
            "--maximum-frame-gap-ms",
            str(args.maximum_frame_gap_ms),
        ],
    )

    missing_outputs = [
        path
        for path in generated_files
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_outputs:
        formatted = "\n".join(f"  {path}" for path in missing_outputs)
        raise SystemExit(
            "Pipeline finished, but these outputs are missing or empty:\n"
            f"{formatted}"
        )

    print("\nSegment processing completed successfully.")
    print(f"Observations: {observations_path}")
    print(f"Aligned data: {aligned_path}")
    print(f"BC samples:   {samples_path}")
    print(f"Manifest:     {manifest_path}")


if __name__ == "__main__":
    main()