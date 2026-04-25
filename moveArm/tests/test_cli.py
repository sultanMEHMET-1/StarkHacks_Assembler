"""Tests for CLI argument parsing."""

from arm_mover.cli import _parse_arguments


def test_bounds_min_negative_requires_equals_form() -> None:
    """Leading '-' on the value is ambiguous as a separate optional; use =."""
    namespace = _parse_arguments(
        [
            "--port",
            "/dev/ttyACM0",
            "--robot-id",
            "test_robot",
            "--urdf-path",
            "/fake/so101.urdf",
            "--dry-run",
            "--bounds-min=-0.25,-0.25,0.05",
            "--bounds-max=0.25,0.25,0.35",
        ]
    )
    assert namespace.bounds_min == (-0.25, -0.25, 0.05)
    assert namespace.bounds_max == (0.25, 0.25, 0.35)


def test_monitor_ee_parses_optional_flags() -> None:
    namespace = _parse_arguments(
        [
            "--port",
            "/dev/ttyACM0",
            "--robot-id",
            "rid",
            "--urdf-path",
            "/fake/so101.urdf",
            "--monitor-ee",
            "--monitor-interval",
            "0.5",
            "--monitor-samples",
            "10",
        ]
    )
    assert namespace.monitor_ee is True
    assert namespace.monitor_interval == 0.5
    assert namespace.monitor_samples == 10


def test_stream_parses_pose_transport_flags() -> None:
    namespace = _parse_arguments(
        [
            "--port",
            "/dev/ttyACM0",
            "--robot-id",
            "rid",
            "--urdf-path",
            "/fake/so101.urdf",
            "--stream",
            "--pose-host",
            "127.0.0.1",
            "--pose-port",
            "9876",
            "--stream-connect-timeout",
            "5.0",
            "--stream-poll-interval",
            "0.02",
            "--stale-pose-threshold",
            "0.5",
        ]
    )
    assert namespace.stream is True
    assert namespace.pose_host == "127.0.0.1"
    assert namespace.pose_port == 9876
    assert namespace.stream_connect_timeout == 5.0
    assert namespace.stream_poll_interval == 0.02
    assert namespace.stale_pose_threshold == 0.5
