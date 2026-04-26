"""TCP JSONL client that keeps only the freshest streamed hand pose."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_RECV_TIMEOUT_SECONDS = 0.2
_SOCKET_READ_SIZE_BYTES = 4096


class PoseClient:
    """Receive pose JSONL messages and expose the latest state safely."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self.connected = False
        self._latest_pose: dict[str, Any] | None = None
        self._latest_pose_received_at_monotonic: float | None = None
        self._run_active = False
        self._last_run_id: int | None = None
        self._pending_new_run = False

    def connect(self, timeout: float) -> None:
        """Connect to the pose server and start the background reader."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.connected:
            return
        self._stop_event.clear()
        self._socket = socket.create_connection((self.host, self.port), timeout=timeout)
        try:
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            # Some test sockets (for example socketpair) do not support TCP options.
            pass
        self._socket.settimeout(_RECV_TIMEOUT_SECONDS)
        self.connected = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def disconnect(self) -> None:
        """Stop the background reader and close the socket."""
        self._stop_event.set()
        socket_to_close = self._socket
        self._socket = None
        if socket_to_close is not None:
            try:
                socket_to_close.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                socket_to_close.close()
            except OSError:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        self.connected = False

    def check_new_run(self) -> bool:
        """Return whether a new run started since the previous check."""
        with self._state_lock:
            has_new_run = self._pending_new_run
            self._pending_new_run = False
            return has_new_run

    def is_run_active(self) -> bool:
        """Return the current run state reported by the server."""
        with self._state_lock:
            return self._run_active

    def get_latest_pose(self) -> dict[str, Any] | None:
        """Return the newest running pose, if available."""
        with self._state_lock:
            if self._latest_pose is None:
                return None
            return dict(self._latest_pose)

    def get_pose_age(self) -> float | None:
        """Return seconds since the latest running pose was received."""
        with self._state_lock:
            if self._latest_pose_received_at_monotonic is None:
                return None
            return time.monotonic() - self._latest_pose_received_at_monotonic

    def _reader_loop(self) -> None:
        """Read JSONL messages continuously and update state with the latest one."""
        buffer = ""
        while not self._stop_event.is_set():
            socket_instance = self._socket
            if socket_instance is None:
                break
            try:
                chunk = socket_instance.recv(_SOCKET_READ_SIZE_BYTES)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Ignoring invalid JSON line from pose server: %r", line)
                    continue
                if not isinstance(message, dict):
                    logger.warning("Ignoring non-object JSON message from pose server: %r", message)
                    continue
                self._process_message(message)
        self.connected = False

    def _process_message(self, message: dict[str, Any]) -> None:
        """Apply one server message to local run/pose state."""
        status = message.get("status")
        run_id_value = message.get("run_id")
        run_id = run_id_value if isinstance(run_id_value, int) else None
        with self._state_lock:
            if status == "running":
                if run_id is not None and run_id != self._last_run_id:
                    self._pending_new_run = True
                if run_id is not None:
                    self._last_run_id = run_id
                self._run_active = True
                self._latest_pose = dict(message)
                self._latest_pose_received_at_monotonic = time.monotonic()
                return
            if status == "stopped":
                if run_id is not None:
                    self._last_run_id = run_id
                self._run_active = False
                self._latest_pose = None
                self._latest_pose_received_at_monotonic = None
                return
            logger.debug("Ignoring pose-server message with unknown status: %r", message)
