"""Tests for streaming pose TCP client state handling."""

from __future__ import annotations

import json
import socket
import time

import pytest

from arm_mover.pose_client import PoseClient


def _wait_until(condition, timeout_seconds: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("Condition not reached before timeout")


def test_pose_client_keeps_only_latest_running_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    client_socket, server_socket = socket.socketpair()

    def fake_create_connection(address, timeout):
        return client_socket

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    client = PoseClient(host="127.0.0.1", port=9876)
    client.connect(timeout=1.0)
    try:
        message_one = json.dumps(
            {"status": "running", "run_id": 1, "x": 0.1, "y": 0.0, "z": 0.2}
        )
        message_two = json.dumps(
            {"status": "running", "run_id": 1, "x": 0.15, "y": 0.01, "z": 0.25}
        )
        server_socket.sendall(f"{message_one}\n{message_two}\n".encode("utf-8"))

        _wait_until(lambda: client.get_latest_pose() is not None)
        _wait_until(lambda: float(client.get_latest_pose()["x"]) == 0.15)

        latest_pose = client.get_latest_pose()
        assert latest_pose is not None
        assert latest_pose["x"] == pytest.approx(0.15)
        assert client.check_new_run() is True
        assert client.check_new_run() is False
        assert client.is_run_active() is True
        assert client.get_pose_age() is not None
    finally:
        client.disconnect()
        server_socket.close()


def test_pose_client_handles_stopped_message(monkeypatch: pytest.MonkeyPatch) -> None:
    client_socket, server_socket = socket.socketpair()

    def fake_create_connection(address, timeout):
        return client_socket

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    client = PoseClient(host="127.0.0.1", port=9876)
    client.connect(timeout=1.0)
    try:
        running_message = json.dumps(
            {"status": "running", "run_id": 4, "x": 0.2, "y": -0.1, "z": 0.3}
        )
        stopped_message = json.dumps({"status": "stopped", "run_id": 4})
        server_socket.sendall(f"{running_message}\n".encode("utf-8"))
        _wait_until(client.is_run_active)

        server_socket.sendall(f"{stopped_message}\n".encode("utf-8"))
        _wait_until(lambda: not client.is_run_active())

        assert client.get_latest_pose() is None
        assert client.get_pose_age() is None
    finally:
        client.disconnect()
        server_socket.close()
