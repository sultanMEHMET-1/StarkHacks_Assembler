"""LeRobot end-effector to joint pipeline helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_robot_kinematics(robot: Any, urdf_path: str) -> tuple[Any, list[str]]:
    """Construct a :class:`~lerobot.model.kinematics.RobotKinematics` for the SO follower."""
    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError as exc:
        raise ImportError(
            "Inverse kinematics requires lerobot kinematics support (placo). "
            "Install with: pip install 'lerobot[feetech,kinematics]'"
        ) from exc

    motor_names = list(robot.bus.motors.keys())
    solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )
    return solver, motor_names


def ee_pose_from_encoder_fk(
    observation: Mapping[str, Any],
    kinematics: Any,
    motor_names: list[str],
) -> dict[str, float]:
    """Map current joint encoders to ``ee.*`` pose using the same FK as LeRobot datasets."""
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        compute_forward_kinematics_joints_to_ee,
    )

    joints = {f"{name}.pos": float(observation[f"{name}.pos"]) for name in motor_names}
    fk_out = compute_forward_kinematics_joints_to_ee(joints, kinematics, motor_names)
    return {
        key: float(fk_out[key])
        for key in (
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "ee.gripper_pos",
        )
    }


def build_ee_to_joints_pipeline(
    robot: Any,
    urdf_path: str,
    bounds_min: Sequence[float] = (-0.35, -0.35, 0.0),
    bounds_max: Sequence[float] = (0.35, 0.35, 0.40),
    max_ee_step_m: float = 0.02,
):
    """Build a reusable LeRobot EE->joint action processing pipeline.

    Returns:
        ``pipeline`` — EE action + observation → joint action.
        ``kinematics`` — same :class:`~lerobot.model.kinematics.RobotKinematics` as IK (for FK logging).
        ``motor_names`` — bus motor order, required for :func:`ee_pose_from_encoder_fk`.
    """
    try:
        from lerobot.processor import RobotProcessorPipeline
        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            transition_to_robot_action,
        )
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            EEBoundsAndSafety,
            InverseKinematicsEEToJoints,
        )
        from lerobot.types import RobotAction, RobotObservation
    except ImportError as exc:
        raise ImportError(
            "Inverse kinematics requires lerobot kinematics support (placo). "
            "Install with: pip install 'lerobot[feetech,kinematics]'"
        ) from exc

    solver, motor_names = build_robot_kinematics(robot, urdf_path)
    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds={"min": list(bounds_min), "max": list(bounds_max)},
                max_ee_step_m=max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return pipeline, solver, motor_names
