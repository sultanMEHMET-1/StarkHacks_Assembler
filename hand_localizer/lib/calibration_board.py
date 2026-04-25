"""Workspace calibration board geometry, image generation, and pose estimation.

This module defines:
- A board model in meters (used for solvePnP). Z=0 for all tag centers/corners.
- A printable layout in pixels (used to generate a PNG).

Printable layout uses **true page margins** measured from the page edges to the
center of the *top-left* tag (BOARD_TAG_IDS[0]).
"""

from dataclasses import dataclass

import cv2
import numpy as np

# === USER-EDITABLE BOARD PARAMETERS ===
# Calibration board IDs must not overlap with cube IDs (0-5).
BOARD_TAG_IDS = (0,1,2,3,4)
# Tag size on the printed board (meters).
BOARD_TAG_SIZE = 0.028
# Center-to-center spacing between neighboring tags (meters).
BOARD_TAG_SPACING_HORIZONTAL = 0.1255
BOARD_TAG_SPACING_VERTICAL = 0.087

# Printable page settings (physical units + DPI).
PAGE_DPI = 300
# US Letter (inches). If you use A4, update these.
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0

# True page margins (mm) to the center of the top-left tag (BOARD_TAG_IDS[0]).
MARGIN_LEFT_TO_TOPLEFT_TAG_CENTER_MM = 48.0
MARGIN_TOP_TO_TOPLEFT_TAG_CENTER_MM = 44.0
# === END USER-EDITABLE PARAMETERS ===


@dataclass
class BoardPose:
    """Board pose and solve quality in camera frame."""

    rotation_matrix: np.ndarray
    translation: np.ndarray
    num_tags_used: int
    reprojection_error: float
    tag_ids_used: list[int]


def board_tag_id_set() -> set[int]:
    """Return calibration board tag IDs as a set."""
    return set(BOARD_TAG_IDS)


def is_board_tag(tag_id: int) -> bool:
    """True if the tag belongs to the workspace calibration board."""
    return tag_id in board_tag_id_set()


def _board_tag_centers() -> dict[int, np.ndarray]:
    """Board-frame centers for each tag (meters), z=0 plane."""
    # Naming convention:
    # - +x is to the right on the board
    # - -y is down on the board (so mapping to image pixels is intuitive: pixels use +y down)
    dx = BOARD_TAG_SPACING_HORIZONTAL
    dy = BOARD_TAG_SPACING_VERTICAL
    return {
        BOARD_TAG_IDS[0]: np.array([0.0, 0.0, 0.0], dtype=np.float64),
        BOARD_TAG_IDS[1]: np.array([dx, 0.0, 0.0], dtype=np.float64),
        BOARD_TAG_IDS[2]: np.array([0.0, -dy, 0.0], dtype=np.float64),
        BOARD_TAG_IDS[3]: np.array([dx, -dy, 0.0], dtype=np.float64),
        BOARD_TAG_IDS[4]: np.array([0.0, -2.0 * dy, 0.0], dtype=np.float64),
    }


def get_board_tag_corners_3d(tag_id: int) -> np.ndarray | None:
    """Return board-frame tag corners in pupil-apriltags corner order."""
    center = _board_tag_centers().get(tag_id)
    if center is None:
        return None
    half = BOARD_TAG_SIZE * 0.5
    return np.array(
        [
            center + np.array([-half, -half, 0.0], dtype=np.float64),  # bottom-left
            center + np.array([half, -half, 0.0], dtype=np.float64),  # bottom-right
            center + np.array([half, half, 0.0], dtype=np.float64),  # top-right
            center + np.array([-half, half, 0.0], dtype=np.float64),  # top-left
        ],
        dtype=np.float64,
    )


def estimate_board_pose(
    detections: list,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> BoardPose | None:
    """Estimate board pose via one solvePnP over all visible board-tag corners."""
    object_points_chunks: list[np.ndarray] = []
    image_points_chunks: list[np.ndarray] = []
    tag_ids_used: list[int] = []
    for detection in detections:
        corners_3d = get_board_tag_corners_3d(int(detection.tag_id))
        if corners_3d is None:
            continue
        corners_2d = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        object_points_chunks.append(corners_3d)
        image_points_chunks.append(corners_2d)
        tag_ids_used.append(int(detection.tag_id))
    if not object_points_chunks:
        return None

    object_points = np.vstack(object_points_chunks).astype(np.float64)
    image_points = np.vstack(image_points_chunks).astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    reprojection_error = float(np.sqrt(np.mean(np.square(errors))))
    return BoardPose(
        rotation_matrix=rotation_matrix,
        translation=tvec.reshape(3),
        num_tags_used=len(tag_ids_used),
        reprojection_error=reprojection_error,
        tag_ids_used=tag_ids_used,
    )


def _page_size_px() -> tuple[int, int]:
    """Return page (width_px, height_px) from PAGE_* and PAGE_DPI."""
    width_px = int(round(float(PAGE_WIDTH_IN) * float(PAGE_DPI)))
    height_px = int(round(float(PAGE_HEIGHT_IN) * float(PAGE_DPI)))
    return (width_px, height_px)


def _mm_to_px(mm: float) -> int:
    """Convert millimeters to pixels using PAGE_DPI."""
    inches = float(mm) / 25.4
    return int(round(inches * float(PAGE_DPI)))


def _m_to_px(meters: float) -> int:
    """Convert meters to pixels using PAGE_DPI."""
    inches = float(meters) * 39.37007874015748
    return int(round(inches * float(PAGE_DPI)))


def generate_board_image() -> np.ndarray:
    """Generate a printable board image using true page margins and DPI.

    Margins are measured from the page edges to the center of BOARD_TAG_IDS[0].
    All tag center offsets come from `_board_tag_centers()` (meters), so the
    printable layout matches the solvePnP geometry.
    """
    width, height = _page_size_px()
    image = np.full((height, width), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

    if float(PAGE_DPI) <= 0:
        raise ValueError(f"PAGE_DPI must be positive, got {PAGE_DPI!r}")
    if float(PAGE_WIDTH_IN) <= 0 or float(PAGE_HEIGHT_IN) <= 0:
        raise ValueError(f"PAGE_WIDTH_IN/PAGE_HEIGHT_IN must be positive, got {PAGE_WIDTH_IN!r}/{PAGE_HEIGHT_IN!r}")
    if float(BOARD_TAG_SIZE) <= 0:
        raise ValueError(f"BOARD_TAG_SIZE must be positive, got {BOARD_TAG_SIZE!r}")

    tag_side_px = max(10, _m_to_px(float(BOARD_TAG_SIZE)))

    # Anchor: center of top-left tag in pixels.
    top_left_center_px = np.array(
        [
            _mm_to_px(float(MARGIN_LEFT_TO_TOPLEFT_TAG_CENTER_MM)),
            _mm_to_px(float(MARGIN_TOP_TO_TOPLEFT_TAG_CENTER_MM)),
        ],
        dtype=np.int32,
    )
    if top_left_center_px[0] < 0 or top_left_center_px[1] < 0:
        raise ValueError(
            "Margins must be non-negative (measured to tag center). "
            f"Got left/top px = {tuple(int(x) for x in top_left_center_px)}"
        )

    centers_m = _board_tag_centers()
    origin_id = int(BOARD_TAG_IDS[0])
    origin_m = centers_m.get(origin_id)
    if origin_m is None:
        raise ValueError("BOARD_TAG_IDS[0] missing from _board_tag_centers()")

    for tag_id, center_m in centers_m.items():
        # Offsets in the board model (meters) relative to top-left tag center.
        dx_m = float(center_m[0] - origin_m[0])
        dy_m = float(center_m[1] - origin_m[1])
        # Image pixels: x right, y down.
        # Board model uses negative y to go down, so flip sign when mapping.
        center_px = top_left_center_px + np.array([_m_to_px(dx_m), _m_to_px(-dy_m)], dtype=np.int32)

        marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), int(tag_side_px))
        # Physical print is rotated 180° relative to OpenCV’s default marker bitmap;
        # rotate so the PNG matches the real board (positions unchanged).
        marker = cv2.rotate(marker, cv2.ROTATE_180)
        x0 = int(center_px[0] - tag_side_px // 2)
        y0 = int(center_px[1] - tag_side_px // 2)
        x1 = x0 + int(tag_side_px)
        y1 = y0 + int(tag_side_px)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            print(
                f"WARNING: tag {tag_id} would clip page bounds "
                f"(x0={x0}, y0={y0}, x1={x1}, y1={y1}; page={width}x{height}). "
                "Adjust margins or spacing."
            )
            continue

        image[y0:y1, x0:x1] = marker
        cv2.putText(
            image,
            f"ID {tag_id}",
            (x0, min(height - 10, y1 + int(tag_side_px * 0.15) + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0,),
            2,
            cv2.LINE_AA,
        )
    return image
