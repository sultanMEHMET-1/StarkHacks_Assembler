"""Palm offset model (cube -> wrist -> palm)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# === MEASURE AND UPDATE THESE ===
# Distance from cube center to wrist surface, meters.
CUBE_TO_WRIST_DISTANCE = 0.025
# Distance from wrist joint to palm center, meters.
WRIST_TO_PALM_DISTANCE = 0.080
# === END USER-EDITABLE SECTION ===

# Cube mounting axis assumptions in cube-local coordinates.
# Update these vectors if your physical cube mounting differs.
CUBE_DOWN_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float64)
HAND_FORWARD_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
PITCH_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _normalized(axis: np.ndarray) -> np.ndarray:
    vec = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        raise ValueError("Axis vector must be non-zero.")
    return vec / norm


def compute_palm_pose(
    cube_translation: np.ndarray,
    cube_rotation_matrix: np.ndarray,
    wrist_pitch_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute palm pose in the same frame as the cube pose."""
    cube_pos = np.asarray(cube_translation, dtype=np.float64).reshape(3)
    cube_rot = np.asarray(cube_rotation_matrix, dtype=np.float64).reshape(3, 3)

    cube_to_wrist_local = _normalized(CUBE_DOWN_AXIS) * float(CUBE_TO_WRIST_DISTANCE)
    wrist_pos = cube_pos + cube_rot @ cube_to_wrist_local

    pitch_local = Rotation.from_rotvec(
        _normalized(PITCH_AXIS) * np.radians(float(wrist_pitch_degrees))
    ).as_matrix()
    hand_forward_pitched_local = pitch_local @ _normalized(HAND_FORWARD_AXIS)
    wrist_to_palm_robot = cube_rot @ (hand_forward_pitched_local * float(WRIST_TO_PALM_DISTANCE))
    palm_pos = wrist_pos + wrist_to_palm_robot

    palm_rot = cube_rot @ pitch_local
    return palm_pos, palm_rot
