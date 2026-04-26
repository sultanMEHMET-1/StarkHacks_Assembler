#!/usr/bin/env python3
"""Camera calibration using ChArUco board.

Usage:
    python calibrate.py --generate-board board.png   # Generate printable board
    python calibrate.py                              # Run interactive calibration
    
Controls during calibration:
    SPACE - Capture frame (if enough corners detected)
    ESC   - Finish capture and compute calibration
    Q     - Quit without saving
"""

import os
import sys

# OpenCV highgui may use Qt; on Wayland the venv Qt build often lacks wayland plugins.
if sys.platform == "linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse

import cv2
import numpy as np

from lib.charuco import (
    calibrate_camera_charuco,
    create_board,
    detect_corners,
    draw_detected_corners,
    generate_board_image,
)
from lib.camera_params import (
    configure_capture_mjpeg,
    open_video_capture,
    parse_camera_arg,
    save_calibration,
)
from lib.visualization import draw_calibration_status, draw_instructions

# Defaults: prefer Logitech C920 (046d:082d) on Linux via /dev/v4l/by-id/
DEFAULT_CAMERA = "c920"
DEFAULT_OUTPUT = "calibration_data.yaml"
DEFAULT_NUM_IMAGES = 20
DEFAULT_RESOLUTION = (1920, 1080)
MIN_CORNERS_FOR_CAPTURE = 6


def generate_board(output_path: str) -> None:
    """Generate a printable ChArUco board image."""
    board = create_board()
    board_image = generate_board_image(board)
    cv2.imwrite(output_path, board_image)
    print(f"ChArUco board saved to: {output_path}")
    print("Print this on letter paper at 100% scale (no fit-to-page)")


def run_calibration(camera: int | str, output_path: str, num_images: int) -> None:
    """Run interactive calibration capture and compute calibration."""
    cap = open_video_capture(camera)
    actual_w, actual_h, actual_fps, actual_fourcc = configure_capture_mjpeg(
        cap, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1]
    )
    print(
        f"Camera opened ({camera!r}): {actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
    )
    print(f"Camera pixel format (FOURCC): {actual_fourcc}")
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera!r}")
        sys.exit(1)
    
    board = create_board()
    
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = (actual_w, actual_h)
    
    print("\nCalibration Controls:")
    print("  SPACE - Capture frame (when corners are detected)")
    print("  ESC   - Finish and compute calibration")
    print("  Q     - Quit without saving")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
        result = detect_corners(gray, board, min_corners=MIN_CORNERS_FOR_CAPTURE)
        corners_detected = result is not None
        
        if corners_detected:
            charuco_corners, charuco_ids = result
            draw_detected_corners(display_frame, charuco_corners, charuco_ids)
            corner_count = len(charuco_ids)
            instruction = f"SPACE to capture ({corner_count} corners) | ESC to finish | Q to quit"
        else:
            instruction = "Point camera at ChArUco board | ESC to finish | Q to quit"
        
        draw_calibration_status(display_frame, len(all_charuco_corners), num_images)
        draw_instructions(display_frame, instruction)
        
        cv2.imshow("Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and corners_detected:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"Captured frame {len(all_charuco_corners)}/{num_images}")
            
            if len(all_charuco_corners) >= num_images:
                print("Target reached! Press ESC to compute calibration or continue capturing.")
        
        elif key == 27:  # ESC
            break
        
        elif key == ord('q') or key == ord('Q'):
            print("Quit without saving.")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(all_charuco_corners) < 3:
        print(f"ERROR: Need at least 3 captured frames, got {len(all_charuco_corners)}")
        sys.exit(1)
    
    print(f"\nComputing calibration from {len(all_charuco_corners)} frames...")
    
    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_charuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    save_calibration(output_path, camera_matrix, dist_coeffs, image_size, rms)
    
    print("\n" + "=" * 50)
    print("CALIBRATION COMPLETE")
    print("=" * 50)
    print(f"Images used:    {len(all_charuco_corners)}")
    print(f"Image size:     {image_size[0]}x{image_size[1]}")
    print(f"RMS error:      {rms:.4f} pixels")
    print(f"Saved to:       {output_path}")
    print()
    
    if rms > 0.5:
        print("WARNING: RMS error > 0.5 pixels. Consider recalibrating with:")
        print("  - More varied board angles")
        print("  - Board positions covering the full frame")
        print("  - Better lighting (avoid glare)")
    else:
        print("Calibration quality: GOOD")


def _camera_cli_type(value: str) -> int | str:
    try:
        return parse_camera_arg(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration using ChArUco board",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--camera', type=_camera_cli_type, default=DEFAULT_CAMERA,
        help=(
            "Camera: c920 (default, Logitech C920 on Linux), integer index, "
            "or path e.g. /dev/video2"
        )
    )
    parser.add_argument(
        '--output', type=str, default=DEFAULT_OUTPUT,
        help=f"Output calibration file (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        '--num-images', type=int, default=DEFAULT_NUM_IMAGES,
        help=f"Target number of calibration images (default: {DEFAULT_NUM_IMAGES})"
    )
    parser.add_argument(
        '--generate-board', type=str, metavar='PATH',
        help="Generate a printable ChArUco board PNG and exit"
    )
    
    args = parser.parse_args()
    
    if args.generate_board:
        generate_board(args.generate_board)
    else:
        run_calibration(args.camera, args.output, args.num_images)


if __name__ == "__main__":
    main()
