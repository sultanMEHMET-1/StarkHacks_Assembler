"""Hardware connection helpers and Cartesian motion sequence execution."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from arm_mover.kinematics import (
    build_ee_to_joints_pipeline,
    build_robot_kinematics,
    ee_pose_from_encoder_fk,
)
from arm_mover.pose_client import PoseClient
from arm_mover.poses import EEPose, euler_xyz_to_rotvec

if TYPE_CHECKING:
    from lerobot.robots.so_follower import SOFollower

logger = logging.getLogger(__name__)

# After long calibration traffic, the Feetech bus sometimes drops a packet when lerobot
# re-enables torque (see configure() -> torque_disabled). Retries recover many transient cases.
CONNECT_MAX_ATTEMPTS: int = 5
CONNECT_RETRY_DELAY_SECONDS: float = 0.75

_LEROBOT_INSTALL_HINT = (
    "The lerobot package is required. Install it with: pip install 'lerobot[feetech,kinematics]'"
)

_POSITION_EPSILON_M: float = 1e-6
# Stay strictly below ``max_ee_step_m``; LeRobot uses ``>`` and float noise can exceed 0.02.
_EE_STEP_MARGIN_M: float = 1e-5


def _format_ee_pose_for_log(ee: dict[str, float]) -> str:
    """Single-line human-readable ``ee.*`` pose for logs."""
    return (
        f"x={ee['ee.x']:.4f} y={ee['ee.y']:.4f} z={ee['ee.z']:.4f} "
        f"wx={ee['ee.wx']:.4f} wy={ee['ee.wy']:.4f} wz={ee['ee.wz']:.4f} "
        f"gripper={ee['ee.gripper_pos']:.2f}"
    )


def _format_ee_request_and_fk_line(
    ee_request: dict[str, float], ee_read: dict[str, float]
) -> str:
    """One log line comparing commanded EE pose to encoder FK (for tuning / tracking checks)."""
    return (
        f"EE request: {_format_ee_pose_for_log(ee_request)} | "
        f"EE read (FK): {_format_ee_pose_for_log(ee_read)}"
    )


def _ee_actions_along_segment(
    start: dict[str, float] | None,
    end: dict[str, float],
    max_step_m: float,
) -> list[dict[str, float]]:
    """Expand one logical EE goal into a chain that satisfies ``EEBoundsAndSafety`` per tick.

    LeRobot's :class:`EEBoundsAndSafety` rejects (raises) any single command whose EE
    position changes by more than ``max_step_m`` from the previous command. Waypoint lists
    therefore require multiple pipeline calls along straight-line segments in Cartesian
    space (orientation and gripper interpolated linearly in tandem).
    """
    if start is None:
        return [dict(end)]
    position_start = np.array(
        [start["ee.x"], start["ee.y"], start["ee.z"]], dtype=float
    )
    position_end = np.array([end["ee.x"], end["ee.y"], end["ee.z"]], dtype=float)
    rotation_start = np.array(
        [start["ee.wx"], start["ee.wy"], start["ee.wz"]], dtype=float
    )
    rotation_end = np.array([end["ee.wx"], end["ee.wy"], end["ee.wz"]], dtype=float)
    gripper_start = float(start["ee.gripper_pos"])
    gripper_end = float(end["ee.gripper_pos"])
    segment_length_m = float(np.linalg.norm(position_end - position_start))
    if segment_length_m < _POSITION_EPSILON_M:
        return [dict(end)]
    unit_direction = (position_end - position_start) / segment_length_m
    traveled_m = 0.0
    actions: list[dict[str, float]] = []
    effective_max_step_m = max(0.0, float(max_step_m) - _EE_STEP_MARGIN_M)
    while segment_length_m - traveled_m > _POSITION_EPSILON_M:
        remaining_m = segment_length_m - traveled_m
        step_m = min(effective_max_step_m, remaining_m)
        traveled_m += step_m
        interpolation = traveled_m / segment_length_m
        position = position_start + unit_direction * traveled_m
        rotation = (1.0 - interpolation) * rotation_start + interpolation * rotation_end
        gripper = (1.0 - interpolation) * gripper_start + interpolation * gripper_end
        actions.append(
            {
                "ee.x": float(position[0]),
                "ee.y": float(position[1]),
                "ee.z": float(position[2]),
                "ee.wx": float(rotation[0]),
                "ee.wy": float(rotation[1]),
                "ee.wz": float(rotation[2]),
                "ee.gripper_pos": float(gripper),
            }
        )
    return actions


def _observation_from_joint_action(joint_action: dict[str, float]) -> dict[str, float]:
    """Build a fake observation dict compatible with the next IK solve."""
    return {key: float(value) for key, value in joint_action.items() if ".pos" in key}


def _import_so_follower() -> tuple[type[Any], type[Any]]:
    """Load lerobot SO follower classes or raise with an actionable message."""
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    except ImportError as exc:
        raise ImportError(_LEROBOT_INSTALL_HINT) from exc
    return SOFollower, SOFollowerRobotConfig


def build_robot(port: str, robot_id: str, max_relative_target: float = 5.0) -> SOFollower:
    """Build an :class:`SOFollower` instance without connecting to the bus.

    Args:
        port: Serial device path (for example ``/dev/ttyACM0``).
        robot_id: Identifier used for the calibration file under the Hugging Face cache.
        max_relative_target: Maximum joint step magnitude per command (degrees when
            ``use_degrees`` is true).

    Returns:
        A configured robot instance. Call :meth:`SOFollower.connect` before use.
    """
    so_follower_cls, config_cls = _import_so_follower()
    config = config_cls(
        port=port,
        id=robot_id,
        max_relative_target=max_relative_target,
        use_degrees=True,
    )
    return so_follower_cls(config)


def connect_robot_with_retries(
    robot: SOFollower,
    max_attempts: int = CONNECT_MAX_ATTEMPTS,
    retry_delay_seconds: float = CONNECT_RETRY_DELAY_SECONDS,
) -> None:
    """Call :meth:`SOFollower.connect`, retrying on serial bus errors.

    LeRobot's SO follower runs calibration and then configures servos; occasionally the
    Feetech SDK reports no status packet when re-enabling torque (often motor id 2,
    ``shoulder_lift``). That is a transport-layer failure, not a bug in pose definitions.

    Args:
        robot: Robot instance that has not yet successfully called :meth:`SOFollower.connect`.
        max_attempts: How many times to try :meth:`SOFollower.connect` before failing.
        retry_delay_seconds: Pause between attempts so the USB adapter can settle.
    """
    last_error: BaseException | None = None
    for attempt_index in range(1, max_attempts + 1):
        try:
            robot.connect()
            if attempt_index > 1:
                logger.info("Connected successfully on attempt %s.", attempt_index)
            return
        except (ConnectionError, OSError) as exc:
            last_error = exc
            logger.warning(
                "connect() failed (attempt %s/%s): %s",
                attempt_index,
                max_attempts,
                exc,
            )
            try:
                robot.disconnect()
            except Exception:
                logger.debug("disconnect() after failed connect raised", exc_info=True)
            if attempt_index >= max_attempts:
                break
            logger.info(
                "Retrying connect in %.2fs (USB/Feetech timing after calibration is often flaky).",
                retry_delay_seconds,
            )
            time.sleep(retry_delay_seconds)
    assert last_error is not None
    raise ConnectionError(
        "Could not connect to the arm after %s attempts. "
        "If the error mentioned motor id 2, that is shoulder_lift — check cabling, power, "
        "and USB; unplug/replug the cable and try again. Underlying error: %s"
        % (max_attempts, last_error)
    ) from last_error


DEFAULT_MONITOR_INTERVAL_SECONDS: float = 0.2
DEFAULT_STREAM_CONNECT_TIMEOUT_SECONDS: float = 10.0
DEFAULT_STREAM_POLL_INTERVAL_SECONDS: float = 0.01
DEFAULT_STALE_POSE_THRESHOLD_SECONDS: float = 0.2
_STALE_POSE_WARNING_INTERVAL_SECONDS: float = 1.0

# Streamed ``gripper`` from the pose server: maps to ``ee.gripper_pos`` on the arm.
STREAMED_GRIPPER_FULLY_OPEN: float = 0.0
STREAMED_GRIPPER_FULLY_CLOSED: float = 100.0


def _clamp_streamed_gripper_to_ee_pos(value: float) -> float:
    """Clamp teleop gripper to the range that means fully open .. fully closed."""
    return float(
        min(
            max(value, STREAMED_GRIPPER_FULLY_OPEN),
            STREAMED_GRIPPER_FULLY_CLOSED,
        )
    )


def _gripper_position_from_streamed_pose(
    pose: dict[str, object],
    observation: dict[str, Any],
) -> float:
    """Use streamed ``gripper`` if present; else gripper encoder FK."""
    raw_gripper = pose.get("gripper")
    if (
        raw_gripper is not None
        and isinstance(raw_gripper, (int, float))
        and not isinstance(raw_gripper, bool)
    ):
        return _clamp_streamed_gripper_to_ee_pos(float(raw_gripper))
    return float(observation.get("gripper.pos", 0.0))


def run_ee_monitor(
    robot: SOFollower,
    urdf_path: str,
    interval_seconds: float,
    max_samples: int | None,
) -> None:
    """Connect, disable torque, and log encoder FK ``ee.*`` pose for manual arm movement."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1 when set")

    kinematics, motor_names = build_robot_kinematics(robot, urdf_path)
    connect_robot_with_retries(robot)
    try:
        robot.bus.disable_torque()
        logger.info(
            "Torque disabled. Move the arm by hand; logging EE read (FK) every %.3fs. "
            "Press Ctrl+C to stop.",
            interval_seconds,
        )
        sample_count = 0
        while True:
            observation = robot.get_observation()
            ee_read = ee_pose_from_encoder_fk(observation, kinematics, motor_names)
            logger.info("EE read (FK): %s", _format_ee_pose_for_log(ee_read))
            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("EE monitor interrupted; exiting.")
    finally:
        robot.disconnect()


def _pose_field_as_float(pose: dict[str, object], field_name: str) -> float:
    """Validate a streamed pose field before handing it to IK."""
    value = pose.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"pose field {field_name!r} must be numeric, got {value!r}")
    return float(value)


def _ee_action_from_stream_pose(
    pose: dict[str, object],
    gripper_position: float,
) -> tuple[dict[str, float], float]:
    """Convert streamed roll/pitch/yaw JSON fields into the EE action dict expected by IK."""
    timestamp = _pose_field_as_float(pose, "timestamp")
    roll = _pose_field_as_float(pose, "roll")
    pitch = _pose_field_as_float(pose, "pitch")
    yaw = _pose_field_as_float(pose, "yaw")
    wx, wy, wz = euler_xyz_to_rotvec(roll, pitch, yaw, degrees=True)
    return (
        {
            "ee.x": _pose_field_as_float(pose, "x"),
            "ee.y": _pose_field_as_float(pose, "y"),
            "ee.z": _pose_field_as_float(pose, "z"),
            "ee.wx": wx,
            "ee.wy": wy,
            "ee.wz": wz,
            "ee.gripper_pos": float(gripper_position),
        },
        timestamp,
    )


def _compute_relative_position_action(
    hand_start_pos: np.ndarray,
    arm_start_pos: np.ndarray,
    current_hand_pos: np.ndarray,
    absolute_orientation: tuple[float, float, float],
    gripper_position: float,
) -> dict[str, float]:
    """Compute arm target from relative position delta and absolute orientation."""
    pos_delta = current_hand_pos - hand_start_pos
    arm_target_pos = arm_start_pos + pos_delta
    wx, wy, wz = absolute_orientation
    return {
        "ee.x": float(arm_target_pos[0]),
        "ee.y": float(arm_target_pos[1]),
        "ee.z": float(arm_target_pos[2]),
        "ee.wx": wx,
        "ee.wy": wy,
        "ee.wz": wz,
        "ee.gripper_pos": float(gripper_position),
    }


def run_pose_stream(
    robot: SOFollower,
    host: str,
    port: int,
    connect_timeout_s: float,
    poll_interval_s: float,
    stale_pose_threshold_s: float,
    urdf_path: str,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    max_ee_step_m: float,
) -> None:
    """Continuously map the freshest streamed hand pose into IK arm commands.

    Uses RELATIVE position (delta from hand start applied to arm start) but
    ABSOLUTE orientation (hand roll/pitch/yaw passed directly to IK).
    """
    if connect_timeout_s <= 0:
        raise ValueError("connect_timeout_s must be positive")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    if stale_pose_threshold_s <= 0:
        raise ValueError("stale_pose_threshold_s must be positive")

    pipeline, kinematics, motor_names = build_ee_to_joints_pipeline(
        robot=robot,
        urdf_path=urdf_path,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        max_ee_step_m=max_ee_step_m,
    )
    pose_client = PoseClient(host=host, port=port)
    connect_robot_with_retries(robot)

    hand_start_pos: np.ndarray | None = None
    arm_start_pos: np.ndarray | None = None
    pending_run_start: bool = False

    try:
        pose_client.connect(timeout=connect_timeout_s)
        logger.info("Waiting for run to start...")
        previous_action: dict[str, float] | None = None
        last_stale_warning_at = 0.0
        while True:
            if not pose_client.connected:
                logger.warning("Pose server disconnected; stopping pose stream.")
                break

            if pose_client.check_new_run():
                pending_run_start = True
                pipeline, kinematics, motor_names = build_ee_to_joints_pipeline(
                    robot=robot,
                    urdf_path=urdf_path,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    max_ee_step_m=max_ee_step_m,
                )
                logger.info("Run starting — pipeline rebuilt, will capture references on first pose.")

            if not pose_client.is_run_active():
                time.sleep(poll_interval_s)
                continue

            pose = pose_client.get_latest_pose()
            if pose is None:
                time.sleep(poll_interval_s)
                continue

            pose_age = pose_client.get_pose_age()
            if pose_age is not None and pose_age > stale_pose_threshold_s:
                now = time.monotonic()
                if now - last_stale_warning_at >= _STALE_POSE_WARNING_INTERVAL_SECONDS:
                    logger.warning("Latest pose is stale (%.3fs old); holding position.", pose_age)
                    last_stale_warning_at = now
                time.sleep(poll_interval_s)
                continue

            observation = robot.get_observation()
            gripper_position = _gripper_position_from_streamed_pose(pose, observation)

            try:
                current_hand_pos = np.array([
                    float(pose["x"]), float(pose["y"]), float(pose["z"])
                ])
                roll = float(pose["roll"])
                pitch = float(pose["pitch"])
                yaw = float(pose["yaw"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed streamed pose: %s", exc)
                time.sleep(poll_interval_s)
                continue

            if pending_run_start:
                hand_start_pos = current_hand_pos.copy()
                arm_ee = ee_pose_from_encoder_fk(observation, kinematics, motor_names)
                arm_start_pos = np.array([
                    arm_ee["ee.x"], arm_ee["ee.y"], arm_ee["ee.z"]
                ])
                previous_action = arm_ee
                pending_run_start = False
                logger.info(
                    "Run started — hand ref: [%.4f, %.4f, %.4f], arm ref: [%.4f, %.4f, %.4f]",
                    hand_start_pos[0], hand_start_pos[1], hand_start_pos[2],
                    arm_start_pos[0], arm_start_pos[1], arm_start_pos[2],
                )

            if hand_start_pos is None or arm_start_pos is None:
                time.sleep(poll_interval_s)
                continue

            wx, wy, wz = euler_xyz_to_rotvec(roll, pitch, yaw, degrees=True)
            goal_action = _compute_relative_position_action(
                hand_start_pos,
                arm_start_pos,
                current_hand_pos,
                (wx, wy, wz),
                gripper_position,
            )

            segment_actions = _ee_actions_along_segment(
                previous_action,
                goal_action,
                max_ee_step_m,
            )
            for ee_action in segment_actions:
                step_observation = robot.get_observation()
                try:
                    joint_action = pipeline((ee_action, step_observation))
                except ValueError as exc:
                    if "EE jump" in str(exc):
                        logger.warning(
                            "EE jump exceeded limit (robot drifted from expected position); "
                            "skipping this step: %s",
                            exc,
                        )
                        continue
                    raise
                robot.send_action(joint_action)
                after_observation = robot.get_observation()
                ee_read = ee_pose_from_encoder_fk(
                    after_observation, kinematics, motor_names
                )
                logger.info(
                    "%s",
                    _format_ee_request_and_fk_line(ee_action, ee_read),
                )
            previous_action = goal_action
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        logger.info("Pose stream interrupted; disconnecting.")
    finally:
        pose_client.disconnect()
        robot.disconnect()


def run_sequence(
    robot: SOFollower,
    poses: list[EEPose],
    dwell_s: float | None,
    cycles: int,
    urdf_path: str | None,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    max_ee_step_m: float,
    dry_run: bool = False,
    skip_ik: bool = False,
) -> None:
    """Run a repeating list of EE poses, converting each tick through IK."""
    if not poses:
        raise ValueError("At least one pose is required")
    if cycles <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}")

    if dry_run and skip_ik:
        try:
            for cycle_index in range(cycles):
                for pose_index, pose in enumerate(poses):
                    ee_action = pose.to_action()
                    logger.info(
                        "Dry-run cycle %s pose %s (%s), IK skipped: %s",
                        cycle_index + 1,
                        pose_index + 1,
                        pose.name,
                        ee_action,
                    )
                    time.sleep(dwell_s if dwell_s is not None else pose.hold_s)
        except KeyboardInterrupt:
            logger.info("Interrupted during dry-run; exiting.")
        return

    if urdf_path is None:
        raise ValueError("urdf_path is required when inverse kinematics is enabled")

    pipeline, kinematics, motor_names = build_ee_to_joints_pipeline(
        robot=robot,
        urdf_path=urdf_path,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        max_ee_step_m=max_ee_step_m,
    )

    def _build_fake_observation() -> dict[str, float]:
        return {f"{motor_name}.pos": 0.0 for motor_name in robot.bus.motors.keys()}

    if dry_run:
        fake_observation = _build_fake_observation()
        try:
            previous_action: dict[str, float] | None = None
            for cycle_index in range(cycles):
                for pose_index, pose in enumerate(poses):
                    goal_action = pose.to_action()
                    segment_actions = _ee_actions_along_segment(
                        previous_action,
                        goal_action,
                        max_ee_step_m,
                    )
                    logger.info(
                        "Dry-run cycle %s pose %s (%s): %s EE command(s) along segment",
                        cycle_index + 1,
                        pose_index + 1,
                        pose.name,
                        len(segment_actions),
                    )
                    for step_index, ee_action in enumerate(segment_actions):
                        joint_action = pipeline((ee_action, fake_observation))
                        fake_observation = _observation_from_joint_action(joint_action)
                        logger.debug(
                            "Dry-run EE substep %s/%s: %s",
                            step_index + 1,
                            len(segment_actions),
                            ee_action,
                        )
                    logger.info(
                        "Dry-run cycle %s pose %s (%s) final ee_action=%s",
                        cycle_index + 1,
                        pose_index + 1,
                        pose.name,
                        goal_action,
                    )
                    logger.info("Computed joint action (at goal): %s", joint_action)
                    previous_action = goal_action
                    time.sleep(dwell_s if dwell_s is not None else pose.hold_s)
        except KeyboardInterrupt:
            logger.info("Interrupted during dry-run; exiting.")
        return

    try:
        connect_robot_with_retries(robot)
        try:
            previous_action = None
            for cycle_index in range(cycles):
                for pose_index, pose in enumerate(poses):
                    goal_action = pose.to_action()
                    segment_actions = _ee_actions_along_segment(
                        previous_action,
                        goal_action,
                        max_ee_step_m,
                    )
                    logger.info(
                        "Cycle %s pose %s (%s): %s EE command(s) along segment",
                        cycle_index + 1,
                        pose_index + 1,
                        pose.name,
                        len(segment_actions),
                    )
                    for step_index, ee_action in enumerate(segment_actions):
                        observation = robot.get_observation()
                        try:
                            joint_action = pipeline((ee_action, observation))
                        except ValueError as exc:
                            if "EE jump" in str(exc):
                                logger.warning(
                                    "EE jump exceeded limit (robot drifted from expected position); "
                                    "skipping substep %s/%s: %s",
                                    step_index + 1,
                                    len(segment_actions),
                                    exc,
                                )
                                continue
                            raise
                        robot.send_action(joint_action)
                        after_observation = robot.get_observation()
                        ee_read = ee_pose_from_encoder_fk(
                            after_observation, kinematics, motor_names
                        )
                        logger.info(
                            "%s",
                            _format_ee_request_and_fk_line(ee_action, ee_read),
                        )
                        logger.debug(
                            "EE substep %s/%s: %s",
                            step_index + 1,
                            len(segment_actions),
                            ee_action,
                        )
                    logger.debug(
                        "Cycle %s pose %s (%s) segment complete; goal=%s",
                        cycle_index + 1,
                        pose_index + 1,
                        pose.name,
                        goal_action,
                    )
                    previous_action = goal_action
                    time.sleep(dwell_s if dwell_s is not None else pose.hold_s)
        except KeyboardInterrupt:
            logger.info("Interrupted; disconnecting.")
    finally:
        robot.disconnect()
