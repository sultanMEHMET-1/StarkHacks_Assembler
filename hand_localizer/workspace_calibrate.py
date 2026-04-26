#!/usr/bin/env python3
"""Workspace (camera->robot) calibration using the board tag IDs in lib/calibration_board."""

import argparse
import gc
import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2

from lib.apriltag_detector import AprilTagDetector
from lib.calibration_board import (
    BOARD_TAG_IDS,
    BOARD_TAG_SIZE,
    estimate_board_pose,
    generate_board_image,
    is_board_tag,
)
from lib.camera_params import (
    configure_capture_mjpeg,
    get_intrinsics,
    load_calibration,
    open_video_capture,
    parse_camera_arg,
)
from lib.visualization import (
    COLOR_TAG_OUTLINE_ALLOWED,
    draw_hud,
    draw_instructions,
    draw_pose_axes,
    draw_tag_id,
    draw_tag_outline,
)
from lib.workspace_calibration import (
    board_to_robot_from_translation,
    camera_to_board_from_pose,
    compute_camera_to_robot,
    save_workspace_transform,
)

DEFAULT_CAMERA = "c920"
DEFAULT_CALIBRATION = "calibration_data.yaml"
DEFAULT_OUTPUT = "workspace_transform.npz"
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_BOARD_IMAGE = "workspace_board.png"


def _parse_float_meters(prompt: str) -> float:
    """Parse user input as meters; accepts comma as decimal separator."""
    raw = input(prompt).strip().replace(",", ".")
    if not raw:
        raise ValueError("empty input")
    return float(raw)


def parse_board_origin() -> tuple[float, float, float]:
    """Prompt user for board origin translation in robot frame."""
    a, b = int(BOARD_TAG_IDS[0]), int(BOARD_TAG_IDS[1])
    print("\nEnter board origin translation in ROBOT frame (meters).")
    print("Board placement constraint: board is flat on table (Z=0 plane) and")
    print(f"TAG {a} -> TAG {b} edge is aligned with robot +X (rotation is identity).")
    x = _parse_float_meters("board_origin_x_m: ")
    y = _parse_float_meters("board_origin_y_m: ")
    z = _parse_float_meters("board_origin_z_m: ")
    return (x, y, z)


def run_workspace_calibration(
    camera: int | str,
    calibration_path: str,
    output_path: str,
) -> None:
    """Interactive workspace calibration loop."""
    try:
        calib = load_calibration(calibration_path)
    except FileNotFoundError:
        print(f"ERROR: Calibration file not found: {calibration_path}")
        print("       Run 'python calibrate.py' first to create it.")
        sys.exit(1)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    camera_params = get_intrinsics(camera_matrix)

    cap = open_video_capture(camera)
    detector: AprilTagDetector | None = None
    try:
        actual_w, actual_h, actual_fps, actual_fourcc = configure_capture_mjpeg(
            cap, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1]
        )
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {camera!r}")
            sys.exit(1)
        print(f"Camera opened ({camera!r}): {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        print(f"Camera pixel format (FOURCC): {actual_fourcc}")
        print(f"Using board tag IDs: {BOARD_TAG_IDS}")
        print("Press SPACE to capture board pose and compute camera->robot.")
        print("Press ESC or Q to quit without saving.\n")

        detector = AprilTagDetector(family="tag16h5", nthreads=4, quad_decimate=1.0, refine_edges=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Corners-only: fused board pose uses solvePnP on corners. Skipping
            # per-tag pose avoids pupil-apriltags stderr ("more than one new minima")
            # and reduces native teardown issues with OpenCV GUI.
            detections = detector.detect(
                gray,
                estimate_tag_pose=False,
                camera_params=camera_params,
                tag_size=BOARD_TAG_SIZE,
            )
            board_detections = [d for d in detections if is_board_tag(int(d.tag_id))]
            board_pose = estimate_board_pose(board_detections, camera_matrix, dist_coeffs)

            for detection in detections:
                if not is_board_tag(int(detection.tag_id)):
                    continue
                draw_tag_outline(frame, detection.corners, color=COLOR_TAG_OUTLINE_ALLOWED)
                draw_tag_id(frame, int(detection.tag_id), detection.center, rejected=False)

            if board_pose is not None:
                draw_pose_axes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    board_pose.rotation_matrix,
                    board_pose.translation,
                    axis_length=BOARD_TAG_SIZE * 1.2,
                )
                status = (
                    f"Board pose OK | tags: {board_pose.num_tags_used} | "
                    f"err: {board_pose.reprojection_error:.2f}px"
                )
            else:
                status = "Board pose unavailable (show one or more board tags)"

            draw_hud(frame, fps=0.0, detection_ms=0.0, allowed_count=len(board_detections), rejected_count=0)
            draw_instructions(frame, f"{status} | SPACE capture | ESC/Q quit")
            cv2.imshow("Workspace Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("Quit without saving.")
                break
            if key == ord(" "):
                if board_pose is None:
                    print("No valid board pose yet; keep board visible and try again.")
                    continue
                try:
                    board_origin = parse_board_origin()
                except ValueError:
                    print("Invalid numeric input; try capture again.")
                    continue
                camera_to_board = camera_to_board_from_pose(board_pose)
                board_to_robot = board_to_robot_from_translation(board_origin)
                camera_to_robot = compute_camera_to_robot(camera_to_board, board_to_robot)
                save_workspace_transform(output_path, camera_to_robot)
                print("\nWorkspace calibration saved.")
                print(f"File: {output_path}")
                print("camera_to_robot:")
                print(camera_to_robot)
                break
    finally:
        # Tear down native backends in a stable order to avoid heap corruption
        # seen on some Linux builds (malloc next->prev_size) after exit.
        if detector is not None:
            del detector
        gc.collect()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        for _ in range(24):
            cv2.waitKey(1)


def _camera_cli_type(value: str) -> int | str:
    try:
        return parse_camera_arg(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Workspace calibration (camera->robot transform)")
    parser.add_argument(
        "--camera",
        type=_camera_cli_type,
        default=DEFAULT_CAMERA,
        help="Camera: c920 (default), integer index, or /dev/video path",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=DEFAULT_CALIBRATION,
        help=f"Camera calibration file from calibrate.py (default: {DEFAULT_CALIBRATION})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output .npz file containing camera_to_robot transform (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--generate-board",
        nargs="?",
        const=DEFAULT_BOARD_IMAGE,
        metavar="PATH",
        help=f"Generate printable board image and exit (default path: {DEFAULT_BOARD_IMAGE})",
    )
    args = parser.parse_args()

    if args.generate_board:
        image = generate_board_image()
        cv2.imwrite(args.generate_board, image)
        print(f"Workspace board image saved to: {args.generate_board}")
        print(f"Contains AprilTag IDs: {BOARD_TAG_IDS}")
        return

    run_workspace_calibration(
        camera=args.camera,
        calibration_path=args.calibration,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
