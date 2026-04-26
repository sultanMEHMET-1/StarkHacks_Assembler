"""Rotation matrix to Euler angle conversion utilities."""

import numpy as np
from scipy.spatial.transform import Rotation


def _nearest_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 matrix onto SO(3) for stable ``Rotation.from_matrix``."""
    r = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(r)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to (roll, pitch, yaw) in degrees.
    
    Uses ZYX Euler convention (standard in robotics).
    
    Returns:
        Tuple of (roll, pitch, yaw) in degrees.
        - Roll: rotation about camera's Z axis (tag spins in-plane)
        - Pitch: rotation about camera's X axis (tag tilts toward/away)
        - Yaw: rotation about camera's Y axis (tag turns left/right)
    """
    r = Rotation.from_matrix(_nearest_rotation_matrix(rotation_matrix))
    yaw, pitch, roll = r.as_euler('ZYX', degrees=True)
    return (roll, pitch, yaw)
