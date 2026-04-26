#!/usr/bin/env python3
"""AprilTag detection and fused cube pose. Run with --help. ESC or Q quits."""
import argparse, csv, errno, json, os, sys, time
from datetime import datetime
from pathlib import Path

if sys.platform == "linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
from serial import SerialException

from lib.apriltag_detector import AprilTagDetector, TagDetection
from lib.camera_params import apply_manual_exposure, configure_capture_mjpeg, get_intrinsics, load_calibration, open_video_capture, parse_camera_arg
from lib.cube_pose import (
    CubePose,
    compute_per_tag_reproj_errors,
    detections_for_cube_fusion,
    estimate_cube_pose,
)
from lib.glove_reader import GloveReader
from lib.palm_model import compute_palm_pose
from lib.pose_math import rotation_matrix_to_euler
from scipy.spatial.transform import Rotation
from lib.pose_server import DEFAULT_PORT as DEFAULT_SERVER_PORT, PoseServer
from lib.visualization import COLOR_TAG_OUTLINE_ALLOWED, COLOR_TAG_OUTLINE_REJECTED, draw_cube_axes, draw_glove_status, draw_hud, draw_pose_axes, draw_run_status, draw_server_status, draw_tag_id, draw_tag_outline
from lib.workspace_calibration import load_workspace_transform, transform_pose

DEFAULT_CAMERA = "c920"
DEFAULT_CALIBRATION = "calibration_data.yaml"
DEFAULT_TAG_SIZE = 0.03
DEFAULT_PRINT_INTERVAL = 10
DEFAULT_RESOLUTION = (1920, 1080)
# Cube tag IDs 0-5 (default allowlist). Tag id 5 is modeled below the nominal bottom face; offset in lib/cube_model.py.
DEFAULT_ALLOWED_TAG_IDS = frozenset({0, 1, 2, 3, 4, 5})
DEFAULT_WORKSPACE_PATH = "workspace_transform.npz"
SERVE_WITHOUT_WORKSPACE_ERROR = "Cannot serve poses without workspace calibration. Run workspace_calibrate.py first, or use --skip-workspace to run without serving."

# Pose quality filters - reject garbage poses
MIN_CUBE_Z_DISTANCE = 0.10  # meters - cube can't be closer than 10cm to camera
MAX_REPROJECTION_ERROR = 20.0  # pixels - reject high-error poses

# Teleop gripper command streamed to arm_mover (LeRobot 0..100 scale)
GRIPPER_REQUEST_OPEN = 0
GRIPPER_REQUEST_CLOSED = 100

# Exponential weighted average smoothing: 70% previous, 30% current
EWA_ALPHA = 0.5


class PoseSmoother:
    """Exponential weighted average smoother for pose parameters (x, y, z, roll, pitch, yaw)."""

    def __init__(self, alpha: float = EWA_ALPHA):
        self.alpha = alpha
        self._prev = None  # (x, y, z, roll, pitch, yaw) or None if no previous

    def reset(self) -> None:
        """Reset smoother state. Call when starting a new run."""
        self._prev = None

    def update(
        self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float
    ) -> tuple[float, float, float, float, float, float]:
        """Apply EWA smoothing. Returns smoothed (x, y, z, roll, pitch, yaw)."""
        current = (x, y, z, roll, pitch, yaw)
        if self._prev is None:
            self._prev = current
            return current
        smoothed = tuple(
            self.alpha * prev + (1 - self.alpha) * curr
            for prev, curr in zip(self._prev, current)
        )
        self._prev = smoothed
        return smoothed

def format_pose_output(detection: TagDetection, euler: tuple[float, float, float], detection_ms: float) -> str:
    t = detection.translation
    roll, pitch, yaw = euler
    return f"[Tag {detection.tag_id:02d}] Pos: (x={t[0]:+.3f}, y={t[1]:+.3f}, z={t[2]:+.3f}) m | Rot: (R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f}) deg | dt: {detection_ms:.1f}ms"

def format_cube_output(cube_pose: CubePose) -> str:
    roll, pitch, yaw = rotation_matrix_to_euler(cube_pose.rotation_matrix)
    x, y, z = cube_pose.translation
    return f"[CUBE]   Pos: (x={x:+.3f}, y={y:+.3f}, z={z:+.3f}) m | Rot: (R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f}) deg | tags: {cube_pose.num_tags_used} | err: {cube_pose.reprojection_error:.2f}px"

def _format_grip(buttons: list[int] | None) -> str:
    if not buttons:
        return "-----"
    return "".join("X" if int(value) else "-" for value in buttons[:5])


def _gripper_request_from_buttons(buttons: list[int]) -> int:
    """Closed if thumb (index 0) and any finger 1–4 pressed; else open."""
    pressed = [int(v) != 0 for v in buttons[:5]]
    while len(pressed) < 5:
        pressed.append(False)
    return (
        GRIPPER_REQUEST_CLOSED
        if pressed[0] and any(pressed[1:5])
        else GRIPPER_REQUEST_OPEN
    )


def format_robot_output(
    rotation_matrix,
    translation,
    using_glove: bool,
    num_tags: int,
    buttons: list[int] | None,
    imu_age_ms: float | None,
) -> str:
    roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)
    x, y, z = translation
    label = "PALM/ROBOT" if using_glove else "CUBE/ROBOT"
    grip = _format_grip(buttons) if using_glove else "-----"
    gripper_part = ""
    if using_glove and buttons is not None:
        gripper_part = f" | gripper_req: {_gripper_request_from_buttons(buttons)}"
    imu_age_text = f"{imu_age_ms:.0f}ms" if imu_age_ms is not None else "N/A"
    return (
        f"[{label}] Pos: (x={x:+.3f}, y={y:+.3f}, z={z:+.3f}) m | "
        f"Rot: (R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f}) deg | "
        f"tags: {num_tags} | grip: {grip}{gripper_part} | imu_age: {imu_age_text}"
    )

# Track previous robot position for delta display
_prev_robot_pos = [None]

def format_robot_delta_output(translation) -> str:
    """Show direction of movement in robot frame for debugging axis mapping."""
    x, y, z = translation
    if _prev_robot_pos[0] is None:
        _prev_robot_pos[0] = (x, y, z)
        return ""
    
    px, py, pz = _prev_robot_pos[0]
    dx, dy, dz = x - px, y - py, z - pz
    _prev_robot_pos[0] = (x, y, z)
    
    # Only show if significant movement
    threshold = 0.01  # 1cm
    if abs(dx) < threshold and abs(dy) < threshold and abs(dz) < threshold:
        return ""
    
    dirs = []
    if abs(dx) >= threshold:
        dirs.append(f"X{'+' if dx > 0 else '-'}{abs(dx):.2f}")
    if abs(dy) >= threshold:
        dirs.append(f"Y{'+' if dy > 0 else '-'}{abs(dy):.2f}")
    if abs(dz) >= threshold:
        dirs.append(f"Z{'+' if dz > 0 else '-'}{abs(dz):.2f}")
    
    if dirs:
        return f"[DELTA]  Robot moved: {', '.join(dirs)}"
    return ""

def _parse_allowed_tag_ids(value: str) -> frozenset[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("allowed-tags list is empty")
    try:
        return frozenset(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid tag id in allowed-tags: {exc}") from exc

def _camera_cli_type(value: str) -> int | str:
    try:
        return parse_camera_arg(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

def _load_workspace_transform_for_detection(workspace_path: str, skip_workspace: bool, serve_enabled: bool):
    if skip_workspace:
        if serve_enabled:
            print(SERVE_WITHOUT_WORKSPACE_ERROR)
            sys.exit(1)
        return None
    path = Path(workspace_path)
    if not path.exists():
        if serve_enabled:
            print(SERVE_WITHOUT_WORKSPACE_ERROR)
            sys.exit(1)
        return None
    try:
        transform = load_workspace_transform(str(path))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Loaded workspace transform: {path}")
    return transform

def _start_pose_server(port: int) -> PoseServer:
    pose_server = PoseServer(port=port)
    try:
        pose_server.start()
    except OSError as exc:
        print(f"Port {port} already in use. Is another instance running?" if exc.errno == errno.EADDRINUSE else f"ERROR: Could not start pose server on port {port}: {exc}")
        sys.exit(1)
    print(f"Waiting for clients... (arm mover should connect to localhost:{port})")
    return pose_server

def _broadcast_output_pose(
    pose_server: PoseServer,
    cube_pose: CubePose,
    output_pose,
    run_id: int,
    smoother: PoseSmoother | None = None,
    buttons: list[int] | None = None,
) -> None:
    robot_rotation, robot_translation = output_pose
    roll, pitch, yaw = rotation_matrix_to_euler(robot_rotation)
    x, y, z = float(robot_translation[0]), float(robot_translation[1]), float(robot_translation[2])
    roll, pitch, yaw = float(roll), float(pitch), float(yaw)
    if smoother is not None:
        x, y, z, roll, pitch, yaw = smoother.update(x, y, z, roll, pitch, yaw)
    if buttons is not None:
        gripper_request = float(_gripper_request_from_buttons(buttons))
    else:
        gripper_request = float(GRIPPER_REQUEST_OPEN)
    payload = {
        "status": "running",
        "run_id": run_id,
        "x": x, "y": y, "z": z,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "num_tags": int(cube_pose.num_tags_used), "reproj_err": float(cube_pose.reprojection_error),
        "gripper": gripper_request,
    }
    pose_server.broadcast(payload)


def _broadcast_stopped(pose_server: PoseServer, run_id: int) -> None:
    pose_server.broadcast({"status": "stopped", "run_id": run_id})

def run_detection(
    camera: int | str,
    calibration_path: str,
    tag_size: float,
    print_interval: int,
    log_timing: bool,
    allowed_tag_ids: frozenset[int],
    use_cube_fusion: bool,
    workspace_path: str,
    skip_workspace: bool,
    serve_enabled: bool,
    serve_port: int,
    exposure: float | None,
    glove_port: str | None,
    glove_buttons_active_low: bool,
) -> None:
    try:
        calib = load_calibration(calibration_path)
    except FileNotFoundError:
        print(f"ERROR: Calibration file not found: {calibration_path}")
        print("       Run 'python calibrate.py' first to create it.")
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    camera_params = get_intrinsics(camera_matrix)
    print(f"Loaded calibration from: {calibration_path}")
    print(f"  RMS error: {calib['rms_error']:.4f} pixels")
    print(f"  fx={camera_params[0]:.1f}, fy={camera_params[1]:.1f}")
    print(f"  cx={camera_params[2]:.1f}, cy={camera_params[3]:.1f}\n")
    cap = open_video_capture(camera)
    actual_w, actual_h, actual_fps, actual_fourcc = configure_capture_mjpeg(cap, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1])
    print(f"Camera opened ({camera!r}): {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    print(f"Camera pixel format (FOURCC): {actual_fourcc}")
    if exposure is not None:
        actual_exposure = apply_manual_exposure(cap, exposure)
        print(f"Manual exposure request: {exposure:.3f} | camera reports: {actual_exposure:.3f}")
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera!r}")
        sys.exit(1)
    detector = AprilTagDetector(family="tag16h5", nthreads=4, quad_decimate=1.0, refine_edges=True)
    print(f"Detector backend: {'pupil-apriltags' if detector.using_pupil else 'OpenCV ArUco'}")
    print(f"Tag size: {tag_size * 100:.1f} cm ({tag_size * 1000:.0f} mm)")
    ids = ", ".join(str(tag_id) for tag_id in sorted(allowed_tag_ids))
    print(f"Allowlist (pose + print): {{{ids}}} - other IDs: magenta outline, no pose")
    print(f"Cube fusion: {'enabled' if use_cube_fusion else 'disabled (--no-cube)'}\nPress ESC or Q to quit\n")
    workspace_transform = _load_workspace_transform_for_detection(workspace_path, skip_workspace, serve_enabled)
    pose_server = _start_pose_server(serve_port) if serve_enabled else None
    glove_reader = None
    first_glove_reading_seen = False
    if glove_port:
        glove_reader = GloveReader(glove_port, buttons_active_low=glove_buttons_active_low)
        try:
            glove_reader.start()
        except SerialException as exc:
            print(f"ERROR: Could not open glove port {glove_port}: {exc}")
            print("Check that the ESP32 is paired and the port is correct.")
            sys.exit(1)
        wiring = "active-low (0=pressed, idle HIGH)" if glove_buttons_active_low else "active-high (1=pressed)"
        print(f"Glove connected on {glove_port} — buttons: {wiring}")
        print("Waiting for first IMU reading...")
    run_active = False
    run_id = 0
    pose_smoother = PoseSmoother()
    timing_file = None
    timing_writer = None
    timing_filename = ""
    if log_timing:
        timing_filename = f"timing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        timing_file = open(timing_filename, "w", newline="")
        timing_writer = csv.writer(timing_file)
        timing_writer.writerow(["frame_number", "capture_ms", "detection_ms", "total_ms", "num_tags_detected"])
        print(f"Logging timing to: {timing_filename}\n")
    frame_count = 0
    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    current_fps = 0.0
    try:
        while True:
            capture_start = time.perf_counter()
            ret, frame = cap.read()
            capture_end = time.perf_counter()
            if not ret:
                continue
            frame_count += 1
            capture_ms = (capture_end - capture_start) * 1000
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_start = time.perf_counter()
            detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
            detect_end = time.perf_counter()
            detection_ms = (detect_end - detect_start) * 1000
            total_ms = (detect_end - capture_start) * 1000
            cube_pose = (
                estimate_cube_pose(detections_for_cube_fusion(detections), camera_matrix, dist_coeffs)
                if use_cube_fusion
                else None
            )
            # region agent log
            if cube_pose is not None and frame_count % 15 == 0:
                used = [
                    d
                    for d in detections_for_cube_fusion(detections)
                    if d.tag_id in cube_pose.tag_ids_used
                ]
                rvec_dbg, _ = cv2.Rodrigues(cube_pose.rotation_matrix)
                tvec_dbg = cube_pose.translation.reshape(3, 1)
                per_err = compute_per_tag_reproj_errors(
                    used, rvec_dbg, tvec_dbg, camera_matrix, dist_coeffs
                )
                rel_eulers = {}
                for _tid in (0, 5):
                    _d = next((x for x in detections if x.tag_id == _tid), None)
                    if (
                        _d is not None
                        and _d.rotation_matrix is not None
                        and _d.translation is not None
                    ):
                        rrel = cube_pose.rotation_matrix.T @ _d.rotation_matrix
                        yaw, pitch, roll = Rotation.from_matrix(rrel).as_euler("ZYX", degrees=True)
                        rel_eulers[str(_tid)] = {
                            "yaw": float(yaw),
                            "pitch": float(pitch),
                            "roll": float(roll),
                        }
                with open(
                    "/home/mercanmeh/code/Hackathons/StarkHacks/hand_localizer/.cursor/debug-8468bc.log",
                    "a",
                    encoding="utf-8",
                ) as _df:
                    _df.write(
                        json.dumps(
                            {
                                "sessionId": "8468bc",
                                "runId": "post-fix",
                                "timestamp": int(time.time() * 1000),
                                "hypothesisId": "H1",
                                "location": "detect.py:loop",
                                "message": "per-tag reproj and tag-vs-cube relative euler",
                                "data": {
                                    "tag_ids_used": list(cube_pose.tag_ids_used),
                                    "per_tag_reproj_px": {str(k): v for k, v in per_err.items()},
                                    "cube_reproj_mean_px": cube_pose.reprojection_error,
                                    "rel_euler_ZYX_deg_tag_in_cube": rel_eulers,
                                },
                            }
                        )
                        + "\n"
                    )
            # endregion
            cube_robot_pose = None
            output_robot_pose = None
            stream_buttons = None
            glove_buttons_display = None
            glove_age_ms = None
            glove_state = None
            if glove_reader is not None:
                glove_state = glove_reader.get_latest()
                glove_age = glove_reader.get_data_age()
                glove_age_ms = None if glove_age is None else glove_age * 1000.0
                if glove_state is not None:
                    glove_buttons_display = glove_state.buttons
            pose_rejected = False
            rejection_reason = ""
            if cube_pose is not None and workspace_transform is not None:
                # Quality filters - reject garbage poses
                if cube_pose.translation[2] < MIN_CUBE_Z_DISTANCE:
                    pose_rejected = True
                    rejection_reason = f"z={cube_pose.translation[2]:.3f}m < {MIN_CUBE_Z_DISTANCE}m"
                elif cube_pose.reprojection_error > MAX_REPROJECTION_ERROR:
                    pose_rejected = True
                    rejection_reason = f"reproj_err={cube_pose.reprojection_error:.1f}px > {MAX_REPROJECTION_ERROR}px"
                
                if not pose_rejected:
                    cube_robot_pose = transform_pose(workspace_transform, cube_pose.rotation_matrix, cube_pose.translation)
                    output_robot_pose = cube_robot_pose
                    if glove_state is not None:
                        if not first_glove_reading_seen:
                            print("Glove data received. IMU active.")
                            first_glove_reading_seen = True
                        palm_pos, palm_rot = compute_palm_pose(
                            cube_translation=cube_robot_pose[1],
                            cube_rotation_matrix=cube_robot_pose[0],
                            wrist_pitch_degrees=glove_state.pitch,
                        )
                        output_robot_pose = (palm_rot, palm_pos)
                        stream_buttons = glove_state.buttons
                    if pose_server is not None and run_active and output_robot_pose is not None:
                        _broadcast_output_pose(
                            pose_server,
                            cube_pose,
                            output_robot_pose,
                            run_id,
                            pose_smoother,
                            stream_buttons,
                        )
            allowed_count = 0
            rejected_count = 0
            for det in detections:
                if det.tag_id in allowed_tag_ids:
                    allowed_count += 1
                    draw_tag_outline(frame, det.corners, color=COLOR_TAG_OUTLINE_ALLOWED)
                    draw_tag_id(frame, det.tag_id, det.center, rejected=False)
                    if det.translation is not None and det.rotation_matrix is not None:
                        draw_pose_axes(frame, camera_matrix, dist_coeffs, det.rotation_matrix, det.translation, axis_length=tag_size * 0.7)
                else:
                    rejected_count += 1
                    draw_tag_outline(frame, det.corners, color=COLOR_TAG_OUTLINE_REJECTED, thickness=3)
                    draw_tag_id(frame, det.tag_id, det.center, rejected=True)
            if cube_pose is not None and not pose_rejected:
                draw_cube_axes(frame, camera_matrix, dist_coeffs, cube_pose.rotation_matrix, cube_pose.translation, axis_length=tag_size * 1.2)
            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_start_time
            if elapsed >= 0.5:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.perf_counter()
            draw_hud(frame, current_fps, detection_ms, allowed_count, rejected_count)
            draw_server_status(frame, pose_server.client_count() if pose_server is not None else 0, pose_server is not None)
            draw_run_status(frame, run_active, run_id)
            draw_glove_status(
                frame,
                connected=glove_reader is not None,
                data_age_ms=glove_age_ms,
                buttons=glove_buttons_display,
            )
            if frame_count % print_interval == 0:
                for det in detections:
                    if det.tag_id not in allowed_tag_ids:
                        print(f"[Tag {det.tag_id:02d}] Rejected (not in allowlist) | dt: {detection_ms:.1f}ms")
                    elif det.translation is None or det.rotation_matrix is None:
                        print(f"[Tag {det.tag_id:02d}] Pose unavailable | dt: {detection_ms:.1f}ms")
                    else:
                        try:
                            print(format_pose_output(det, rotation_matrix_to_euler(det.rotation_matrix), detection_ms))
                        except ValueError:
                            continue
                if cube_pose is not None:
                    if pose_rejected:
                        print(f"[CUBE]   REJECTED: {rejection_reason} | tags: {cube_pose.num_tags_used}")
                    else:
                        try:
                            print(format_cube_output(cube_pose))
                        except ValueError:
                            pass
                if output_robot_pose is not None and cube_pose is not None:
                    try:
                        print(
                            format_robot_output(
                                output_robot_pose[0],
                                output_robot_pose[1],
                                using_glove=glove_reader is not None,
                                num_tags=cube_pose.num_tags_used,
                                buttons=stream_buttons,
                                imu_age_ms=glove_age_ms,
                            )
                        )
                        delta_msg = format_robot_delta_output(output_robot_pose[1])
                        if delta_msg:
                            print(delta_msg)
                    except ValueError:
                        pass
                if not detections:
                    print(f"[No tags] dt: {detection_ms:.1f}ms")
            if timing_writer:
                timing_writer.writerow([frame_count, f"{capture_ms:.2f}", f"{detection_ms:.2f}", f"{total_ms:.2f}", len(detections)])
            cv2.imshow("AprilTag Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not run_active:
                    run_active = True
                    run_id += 1
                    pose_smoother.reset()
                    print(f"Run {run_id} STARTED")
                else:
                    run_active = False
                    if pose_server is not None:
                        _broadcast_stopped(pose_server, run_id)
                    print(f"Run {run_id} STOPPED")
            elif key == 27:
                if run_active:
                    run_active = False
                    if pose_server is not None:
                        _broadcast_stopped(pose_server, run_id)
                    print(f"Run {run_id} STOPPED (ESC emergency exit)")
                break
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        if glove_reader is not None:
            glove_reader.stop()
        if pose_server is not None:
            pose_server.stop()
        cap.release()
        cv2.destroyAllWindows()
        if timing_file:
            timing_file.close()
            print(f"\nTiming log saved to: {timing_filename}")

def main() -> None:
    parser = argparse.ArgumentParser(description="AprilTag detection and pose estimation", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--camera", type=_camera_cli_type, default=DEFAULT_CAMERA, help="Camera: c920 (default), integer index, or path e.g. /dev/video2")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION, help=f"Path to calibration file (default: {DEFAULT_CALIBRATION})")
    parser.add_argument("--tag-size", type=float, default=DEFAULT_TAG_SIZE, help=f"Physical tag size in meters (default: {DEFAULT_TAG_SIZE})")
    parser.add_argument("--print-interval", type=int, default=DEFAULT_PRINT_INTERVAL, help=f"Print pose to terminal every N frames (default: {DEFAULT_PRINT_INTERVAL})")
    parser.add_argument("--log-timing", action="store_true", help="Log per-frame timing to CSV file")
    parser.add_argument("--exposure", type=float, default=None, help="Manual exposure value (backend-specific scale)")
    parser.add_argument("--no-cube", action="store_true", help="Disable fused cube pose estimation overlay/print")
    parser.add_argument("--serve", action="store_true", help="Enable TCP pose streaming server")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help=f"TCP port for pose streaming when --serve is used (default: {DEFAULT_SERVER_PORT})")
    parser.add_argument("--workspace", "--workspace-transform", dest="workspace_path", type=str, default=DEFAULT_WORKSPACE_PATH, help=f"Path to camera_to_robot transform (default: {DEFAULT_WORKSPACE_PATH})")
    parser.add_argument("--skip-workspace", action="store_true", help="Skip workspace calibration loading even if a transform file exists")
    parser.add_argument("--glove-port", type=str, default=None, help="Bluetooth serial port for glove (e.g., /dev/rfcomm0, COM5)")
    parser.add_argument(
        "--glove-buttons-active-high",
        action="store_true",
        help="Firmware sends 1=pressed (omit for typical ESP32 pull-up: idle 1, pressed 0)",
    )
    parser.add_argument("--allowed-tags", type=str, default=",".join(str(i) for i in sorted(DEFAULT_ALLOWED_TAG_IDS)), help=f"Comma-separated allowlist (default: {','.join(str(i) for i in sorted(DEFAULT_ALLOWED_TAG_IDS))})")
    args = parser.parse_args()
    run_detection(
        args.camera,
        args.calibration,
        args.tag_size,
        args.print_interval,
        args.log_timing,
        _parse_allowed_tag_ids(args.allowed_tags),
        not args.no_cube,
        args.workspace_path,
        args.skip_workspace,
        args.serve,
        args.port,
        args.exposure,
        args.glove_port,
        glove_buttons_active_low=not args.glove_buttons_active_high,
    )

if __name__ == "__main__":
    main()
