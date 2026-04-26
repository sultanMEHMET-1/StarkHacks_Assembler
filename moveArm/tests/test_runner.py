"""Tests for runner helpers (no lerobot import)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arm_mover.runner import (
    _format_ee_pose_for_log,
    _format_ee_request_and_fk_line,
    _ee_action_from_stream_pose,
    _gripper_position_from_streamed_pose,
    connect_robot_with_retries,
    run_ee_monitor,
)


def test_format_ee_pose_for_log_includes_position_and_gripper() -> None:
    line = _format_ee_pose_for_log(
        {
            "ee.x": 0.1,
            "ee.y": -0.2,
            "ee.z": 0.3,
            "ee.wx": 0.01,
            "ee.wy": -0.02,
            "ee.wz": 0.03,
            "ee.gripper_pos": 50.0,
        }
    )
    assert "x=0.1000" in line
    assert "gripper=50.00" in line


def test_format_ee_request_and_fk_line_joins_both_poses() -> None:
    pose = {
        "ee.x": 0.1,
        "ee.y": 0.0,
        "ee.z": 0.2,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 40.0,
    }
    line = _format_ee_request_and_fk_line(pose, pose)
    assert "EE request:" in line
    assert "EE read (FK):" in line
    assert "|" in line


def test_run_ee_monitor_rejects_non_positive_interval() -> None:
    robot = MagicMock()
    with pytest.raises(ValueError, match="interval_seconds"):
        run_ee_monitor(robot, "/fake/so101.urdf", 0.0, None)


def test_run_ee_monitor_rejects_invalid_sample_count() -> None:
    robot = MagicMock()
    with pytest.raises(ValueError, match="max_samples"):
        run_ee_monitor(robot, "/fake/so101.urdf", 0.1, 0)


def test_gripper_position_from_streamed_pose_open_and_closed() -> None:
    observation: dict = {"gripper.pos": 50.0}
    assert _gripper_position_from_streamed_pose({"gripper": 0}, observation) == pytest.approx(0.0)
    assert _gripper_position_from_streamed_pose({"gripper": 100}, observation) == pytest.approx(100.0)


def test_gripper_position_from_streamed_pose_clamps() -> None:
    observation: dict = {"gripper.pos": 50.0}
    assert _gripper_position_from_streamed_pose({"gripper": -10}, observation) == pytest.approx(0.0)
    assert _gripper_position_from_streamed_pose({"gripper": 150}, observation) == pytest.approx(100.0)


def test_gripper_position_from_streamed_pose_falls_back_to_encoder() -> None:
    observation: dict = {"gripper.pos": 33.0}
    assert _gripper_position_from_streamed_pose({"gripper": 40}, observation) == pytest.approx(40.0)
    assert _gripper_position_from_streamed_pose({}, observation) == pytest.approx(33.0)


def test_ee_action_from_stream_pose_converts_fields() -> None:
    action, timestamp = _ee_action_from_stream_pose(
        {
            "x": 0.1,
            "y": -0.2,
            "z": 0.3,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "timestamp": 123.0,
        },
        gripper_position=42.0,
    )
    assert timestamp == 123.0
    assert action["ee.x"] == pytest.approx(0.1)
    assert action["ee.y"] == pytest.approx(-0.2)
    assert action["ee.z"] == pytest.approx(0.3)
    assert action["ee.gripper_pos"] == pytest.approx(42.0)
    assert action["ee.wx"] == pytest.approx(0.0)
    assert action["ee.wy"] == pytest.approx(0.0)
    assert action["ee.wz"] == pytest.approx(0.0)


def test_connect_robot_with_retries_succeeds_on_second_attempt() -> None:
    robot = MagicMock()
    robot.connect.side_effect = [ConnectionError("no status packet"), None]

    connect_robot_with_retries(robot, max_attempts=5, retry_delay_seconds=0.0)

    assert robot.connect.call_count == 2
    robot.disconnect.assert_called_once()


def test_connect_robot_with_retries_raises_after_exhausting_attempts() -> None:
    robot = MagicMock()
    robot.connect.side_effect = ConnectionError("no status packet")

    with pytest.raises(ConnectionError, match="Could not connect to the arm"):
        connect_robot_with_retries(robot, max_attempts=3, retry_delay_seconds=0.0)

    assert robot.connect.call_count == 3
    assert robot.disconnect.call_count == 3
