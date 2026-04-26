"""Workspace transform utilities (camera<->board<->robot)."""

from pathlib import Path

import numpy as np

from lib.calibration_board import BoardPose


def make_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from R and t."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 homogeneous transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def camera_to_board_from_pose(board_pose: BoardPose) -> np.ndarray:
    """Convert solvePnP board->camera output into camera->board transform."""
    board_to_camera = make_transform(board_pose.rotation_matrix, board_pose.translation)
    return invert_transform(board_to_camera)


def board_to_robot_from_translation(board_origin_in_robot: tuple[float, float, float]) -> np.ndarray:
    """Create board->robot transform with identity rotation."""
    return make_transform(np.eye(3, dtype=np.float64), np.array(board_origin_in_robot, dtype=np.float64))


def compute_camera_to_robot(
    camera_to_board: np.ndarray,
    board_to_robot: np.ndarray,
) -> np.ndarray:
    """Compose camera->robot using camera->board and board->robot."""
    return board_to_robot @ camera_to_board


def save_workspace_transform(filepath: str, camera_to_robot: np.ndarray) -> None:
    """Persist camera->robot transform as a .npz file."""
    np.savez(filepath, camera_to_robot=np.asarray(camera_to_robot, dtype=np.float64))


def load_workspace_transform(filepath: str) -> np.ndarray:
    """Load camera->robot transform from .npz file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Workspace transform file not found: {filepath}")
    with np.load(filepath) as data:
        if "camera_to_robot" not in data:
            raise ValueError(f"Malformed workspace transform file (missing camera_to_robot): {filepath}")
        transform = np.asarray(data["camera_to_robot"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Malformed workspace transform shape {transform.shape}; expected (4, 4)")
    return transform


def transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to a 3D point."""
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    result = transform @ point_h
    return result[:3]


def transform_pose(
    transform: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frame transform to a pose (R,t)."""
    source_pose = make_transform(rotation_matrix, translation)
    target_pose = transform @ source_pose
    return target_pose[:3, :3], target_pose[:3, 3]
