"""Command-line entry point for Cartesian end-effector motion sequences."""

from __future__ import annotations

import argparse
import logging
import sys

from arm_mover.poses import POSE_UNITS_CHOICES, SEQUENCES, load_poses_file
from arm_mover.runner import (
    DEFAULT_MONITOR_INTERVAL_SECONDS,
    DEFAULT_STALE_POSE_THRESHOLD_SECONDS,
    DEFAULT_STREAM_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_STREAM_POLL_INTERVAL_SECONDS,
    build_robot,
    run_ee_monitor,
    run_pose_stream,
    run_sequence,
)


def _parse_xyz_tuple(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 3 comma-separated values, got {value!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse bound tuple {value!r} as floats"
        ) from exc


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    sequence_names = sorted(SEQUENCES.keys())
    parser = argparse.ArgumentParser(
        description="Send Cartesian EE poses to an SO-100/SO-101 follower arm via IK.",
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port for the robot (for example /dev/ttyACM0).",
    )
    parser.add_argument(
        "--robot-id",
        "--id",
        dest="robot_id",
        required=True,
        help="Stable id used for the lerobot calibration file name.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="How many times to repeat the full pose list.",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=1.5,
        help="Seconds to wait after each pose command.",
    )
    parser.add_argument(
        "--max-rel",
        "--max-relative-target",
        dest="max_rel",
        type=float,
        default=5.0,
        help="Maximum per-command joint step in degrees (safety clip).",
    )
    parser.add_argument(
        "--sequence",
        choices=sequence_names,
        default="home_left_right",
        help="Which named pose list to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log poses and timing without connecting to hardware.",
    )
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Skip IK conversion; valid only with --dry-run.",
    )
    parser.add_argument(
        "--urdf-path",
        help="Path to so101_new_calib.urdf used by the IK solver.",
    )
    parser.add_argument(
        "--bounds-min",
        type=_parse_xyz_tuple,
        default=(-0.35, -0.35, 0.0),
        help=(
            "Workspace minimum x,y,z in meters (comma-separated). "
            "If the first value is negative, use --bounds-min=-0.1,-0.2,0.05 "
            "so the shell does not treat the leading '-' as a new flag."
        ),
    )
    parser.add_argument(
        "--bounds-max",
        type=_parse_xyz_tuple,
        default=(0.35, 0.35, 0.4),
        help="Workspace maximum x,y,z in meters (comma-separated).",
    )
    parser.add_argument(
        "--max-ee-step",
        type=float,
        default=0.02,
        help="Maximum Cartesian step in meters per tick.",
    )
    parser.add_argument(
        "--poses-file",
        help="Optional JSON/YAML file with Cartesian end-effector poses.",
    )
    parser.add_argument(
        "--pose-units",
        choices=POSE_UNITS_CHOICES,
        default="euler-deg",
        help="Interpretation for orientation values in --poses-file.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream poses from hand_localizer and drive the arm continuously.",
    )
    parser.add_argument(
        "--pose-host",
        default="127.0.0.1",
        help="Pose server host used with --stream (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--pose-port",
        type=int,
        default=9876,
        metavar="PORT",
        help="Pose server port used with --stream (default: 9876).",
    )
    parser.add_argument(
        "--stream-connect-timeout",
        type=float,
        default=DEFAULT_STREAM_CONNECT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds to wait for initial pose server connection when using --stream "
            f"(default: {DEFAULT_STREAM_CONNECT_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--stream-poll-interval",
        type=float,
        default=DEFAULT_STREAM_POLL_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=(
            "Main-loop sleep interval for --stream mode "
            f"(default: {DEFAULT_STREAM_POLL_INTERVAL_SECONDS})."
        ),
    )
    parser.add_argument(
        "--stale-pose-threshold",
        type=float,
        default=DEFAULT_STALE_POSE_THRESHOLD_SECONDS,
        metavar="SECONDS",
        help=(
            "Treat latest streamed pose as stale after this age in seconds "
            f"(default: {DEFAULT_STALE_POSE_THRESHOLD_SECONDS})."
        ),
    )
    parser.add_argument(
        "--monitor-ee",
        action="store_true",
        help=(
            "Connect, disable motor torque, and log EE pose from encoder FK at "
            "--monitor-interval until Ctrl+C or --monitor-samples. Requires --urdf-path."
        ),
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds between EE read log lines when using --monitor-ee "
            f"(default: {DEFAULT_MONITOR_INTERVAL_SECONDS})."
        ),
    )
    parser.add_argument(
        "--monitor-samples",
        type=int,
        default=None,
        metavar="N",
        help="With --monitor-ee, stop after N reads (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level name (for example DEBUG or WARNING).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Configure logging and run the selected motion sequence."""
    args = _parse_arguments(argv)
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        print(
            f"Invalid --log-level {args.log_level!r}; use a logging level name.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.skip_ik and not args.dry_run:
        print("--skip-ik is only allowed with --dry-run.", file=sys.stderr)
        raise SystemExit(2)
    if args.stream:
        if args.dry_run:
            print("--stream cannot be used with --dry-run.", file=sys.stderr)
            raise SystemExit(2)
        if args.skip_ik:
            print("--stream cannot be used with --skip-ik.", file=sys.stderr)
            raise SystemExit(2)
        if args.monitor_ee:
            print("--stream cannot be used with --monitor-ee.", file=sys.stderr)
            raise SystemExit(2)
        if args.pose_port < 1 or args.pose_port > 65535:
            print("--pose-port must be in the range 1..65535.", file=sys.stderr)
            raise SystemExit(2)
        if args.stream_connect_timeout <= 0:
            print("--stream-connect-timeout must be positive.", file=sys.stderr)
            raise SystemExit(2)
        if args.stream_poll_interval <= 0:
            print("--stream-poll-interval must be positive.", file=sys.stderr)
            raise SystemExit(2)
        if args.stale_pose_threshold <= 0:
            print("--stale-pose-threshold must be positive.", file=sys.stderr)
            raise SystemExit(2)
        if args.urdf_path is None:
            print("--urdf-path is required for --stream.", file=sys.stderr)
            raise SystemExit(2)
    if args.monitor_ee:
        if args.dry_run:
            print("--monitor-ee cannot be used with --dry-run.", file=sys.stderr)
            raise SystemExit(2)
        if args.skip_ik:
            print("--monitor-ee cannot be used with --skip-ik.", file=sys.stderr)
            raise SystemExit(2)
        if args.urdf_path is None:
            print("--urdf-path is required for --monitor-ee.", file=sys.stderr)
            raise SystemExit(2)
        if args.monitor_interval <= 0:
            print("--monitor-interval must be positive.", file=sys.stderr)
            raise SystemExit(2)
        if args.monitor_samples is not None and args.monitor_samples < 1:
            print("--monitor-samples must be at least 1 when set.", file=sys.stderr)
            raise SystemExit(2)
    elif args.urdf_path is None and not (args.dry_run and args.skip_ik):
        print(
            "--urdf-path is required unless both --dry-run and --skip-ik are set.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        robot = build_robot(
            port=args.port,
            robot_id=args.robot_id,
            max_relative_target=args.max_rel,
        )
        if args.monitor_ee:
            run_ee_monitor(
                robot,
                urdf_path=args.urdf_path,
                interval_seconds=args.monitor_interval,
                max_samples=args.monitor_samples,
            )
        elif args.stream:
            run_pose_stream(
                robot=robot,
                host=args.pose_host,
                port=args.pose_port,
                connect_timeout_s=args.stream_connect_timeout,
                poll_interval_s=args.stream_poll_interval,
                stale_pose_threshold_s=args.stale_pose_threshold,
                urdf_path=args.urdf_path,
                bounds_min=args.bounds_min,
                bounds_max=args.bounds_max,
                max_ee_step_m=args.max_ee_step,
            )
        else:
            pose_list = (
                load_poses_file(args.poses_file, args.pose_units)
                if args.poses_file
                else SEQUENCES[args.sequence]
            )
            run_sequence(
                robot,
                pose_list,
                dwell_s=args.dwell,
                cycles=args.cycles,
                urdf_path=args.urdf_path,
                bounds_min=args.bounds_min,
                bounds_max=args.bounds_max,
                max_ee_step_m=args.max_ee_step,
                dry_run=args.dry_run,
                skip_ik=args.skip_ik,
            )
    except Exception:
        logging.exception("arm-mover failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
