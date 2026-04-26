"""Bluetooth serial reader for ESP32 glove IMU + button stream."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import serial


@dataclass
class GloveState:
    """Immutable snapshot of the most recent glove reading."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    buttons: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    timestamp_ms: int = 0
    received_at: float = 0.0


class GloveReader:
    """Read JSONL glove state from a serial (Bluetooth SPP) port."""

    def __init__(self, port: str, baudrate: int = 115200, *, buttons_active_low: bool = True):
        """
        buttons_active_low: True for typical ESP32 INPUT_PULLUP (idle=1, pressed=0).
        Set False if firmware sends 1=pressed, 0=released (active-high).
        """
        self.port = port
        self.baudrate = baudrate
        self.buttons_active_low = buttons_active_low
        self.serial_conn: serial.Serial | None = None
        self.running = False
        self._reader_thread: threading.Thread | None = None

        self._latest_state: GloveState | None = None
        self._state_lock = threading.Lock()
        self.connected = False

    def start(self) -> None:
        """Open port and launch the background read loop."""
        self.serial_conn = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1.0,
        )
        self.connected = True
        self.running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        while self.running:
            if self.serial_conn is None:
                break
            try:
                raw = self.serial_conn.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    state = GloveState(
                        roll=float(data.get("roll", 0.0)),
                        pitch=float(data.get("pitch", 0.0)),
                        yaw=float(data.get("yaw", 0.0)),
                        buttons=self._normalize_buttons(data.get("btn", [])),
                        timestamp_ms=int(data.get("t", 0)),
                        received_at=time.time(),
                    )
                except (TypeError, ValueError):
                    continue
                with self._state_lock:
                    self._latest_state = state
            except (serial.SerialException, OSError):
                self.connected = False
                print("WARNING: Glove serial connection lost. Using last known glove state.")
                break

    def _normalize_buttons(self, raw_buttons: object) -> list[int]:
        if not isinstance(raw_buttons, list):
            return [0, 0, 0, 0, 0]
        normalized = []
        for value in raw_buttons[:5]:
            try:
                v = int(value)
                if self.buttons_active_low:
                    is_pressed = v == 0
                else:
                    is_pressed = v != 0
                normalized.append(1 if is_pressed else 0)
            except (TypeError, ValueError):
                normalized.append(0)
        if len(normalized) < 5:
            normalized.extend([0] * (5 - len(normalized)))
        return normalized

    def get_latest(self) -> GloveState | None:
        """Return the latest known state, or None before first reading."""
        with self._state_lock:
            return self._latest_state

    def get_data_age(self) -> float | None:
        """Seconds since latest sample, or None if no sample yet."""
        with self._state_lock:
            if self._latest_state is None:
                return None
            return time.time() - self._latest_state.received_at

    def stop(self) -> None:
        """Stop background read loop and close serial connection."""
        self.running = False
        if self.serial_conn is not None and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except OSError:
                pass
        self.connected = False
