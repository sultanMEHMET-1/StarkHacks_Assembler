"""Integration-style tests for LeRobot IK pipeline wiring."""

from __future__ import annotations

import os

import pytest

from arm_mover.kinematics import (
    build_ee_to_joints_pipeline,
    ee_pose_from_encoder_fk,
)


class _FakeBus:
    def __init__(self) -> None:
        self.motors = {
            "shoulder_pan": object(),
            "shoulder_lift": object(),
            "elbow_flex": object(),
            "wrist_flex": object(),
            "wrist_roll": object(),
            "gripper": object(),
        }


class _FakeRobot:
    def __init__(self) -> None:
        self.bus = _FakeBus()


def test_pipeline_produces_joint_action() -> None:
    pytest.importorskip("placo")
    urdf_path = os.getenv("LEROBOT_SO101_URDF")
    if not urdf_path:
        pytest.skip("Set LEROBOT_SO101_URDF to run IK pipeline test.")

    robot = _FakeRobot()
    pipeline, kinematics, motor_names = build_ee_to_joints_pipeline(
        robot=robot, urdf_path=urdf_path
    )
    ee_action = {
        "ee.x": 0.15,
        "ee.y": 0.0,
        "ee.z": 0.15,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 37.0,
    }
    observation = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }
    output = pipeline((ee_action, observation))
    ee_fk = ee_pose_from_encoder_fk(observation, kinematics, motor_names)
    assert "ee.x" in ee_fk and isinstance(ee_fk["ee.x"], float)

    expected_joint_keys = {
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    }
    assert expected_joint_keys.issubset(set(output.keys()))
    assert output["gripper.pos"] == pytest.approx(ee_action["ee.gripper_pos"])
