"""Unit tests for Cartesian pose conversions and action formatting."""

from __future__ import annotations

from arm_mover.poses import EEPose, rotvec_to_euler_xyz


def test_ee_pose_euler_roundtrip() -> None:
    pose = EEPose.from_euler(
        name="roundtrip",
        x=0.15,
        y=0.0,
        z=0.15,
        roll=30.0,
        pitch=0.0,
        yaw=0.0,
        gripper_pos=50.0,
    )
    recovered_roll, recovered_pitch, recovered_yaw = rotvec_to_euler_xyz(
        pose.rotvec, degrees=True
    )
    assert abs(recovered_roll - 30.0) < 1e-6
    assert abs(recovered_pitch - 0.0) < 1e-6
    assert abs(recovered_yaw - 0.0) < 1e-6


def test_to_action_keys() -> None:
    pose = EEPose.from_euler(
        name="keys",
        x=0.15,
        y=0.01,
        z=0.16,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        gripper_pos=42.0,
    )
    action = pose.to_action()
    assert set(action.keys()) == {
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "ee.gripper_pos",
    }
