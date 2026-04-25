"""Fuse multi-tag detections into a single cube pose via solvePnP."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lib.cube_model import get_tag_corners_3d

if TYPE_CHECKING:
    from lib.apriltag_detector import TagDetection


@dataclass
class CubePose:
    """Fused cube pose in camera frame."""

    translation: np.ndarray
    rotation_matrix: np.ndarray
    num_tags_used: int
    reprojection_error: float
    tag_ids_used: list[int]


def detections_for_cube_fusion(detections: list["TagDetection"]) -> list["TagDetection"]:
    """Return only tag detections that belong to the cube model (known 3D corners)."""
    return [d for d in detections if get_tag_corners_3d(d.tag_id) is not None]


def estimate_cube_pose(
    detections: list["TagDetection"],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> CubePose | None:
    """Estimate cube pose from all known tag corners visible in a frame."""
    cube_detections = detections_for_cube_fusion(detections)
    if len(cube_detections) >= 3:
        if all(d.pose_error is not None for d in cube_detections):
            cube_detections = sorted(cube_detections, key=lambda d: d.pose_error)[:2]
        else:
            cube_detections = sorted(cube_detections, key=lambda d: d.tag_id, reverse=True)[
                :2
            ]

    object_points_chunks: list[np.ndarray] = []
    image_points_chunks: list[np.ndarray] = []
    tag_ids_used: list[int] = []

    for detection in cube_detections:
        corners_3d = get_tag_corners_3d(detection.tag_id)
        if corners_3d is None:
            continue
        corners_2d = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        object_points_chunks.append(corners_3d)
        image_points_chunks.append(corners_2d)
        tag_ids_used.append(int(detection.tag_id))

    if not object_points_chunks:
        return None

    object_points_3d = np.vstack(object_points_chunks).astype(np.float64)
    image_points_2d = np.vstack(image_points_chunks).astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points_3d,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projected_points, _ = cv2.projectPoints(
        object_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_points - image_points_2d, axis=1)
    reprojection_error = float(np.sqrt(np.mean(np.square(errors))))

    return CubePose(
        translation=tvec.reshape(3),
        rotation_matrix=rotation_matrix,
        num_tags_used=len(tag_ids_used),
        reprojection_error=reprojection_error,
        tag_ids_used=tag_ids_used,
    )


def compute_per_tag_reproj_errors(
    cube_detections: list["TagDetection"],
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[int, float]:
    """Mean pixel reprojection error per tag for a fused pose (same 3D/2D pairing as fusion)."""
    out: dict[int, float] = {}
    for detection in cube_detections:
        corners_3d = get_tag_corners_3d(detection.tag_id)
        if corners_3d is None:
            continue
        corners_2d = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        projected, _ = cv2.projectPoints(
            corners_3d,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(projected - corners_2d, axis=1)))
        out[int(detection.tag_id)] = err
    return out
