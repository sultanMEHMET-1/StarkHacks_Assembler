"""Drawing overlays for AprilTag visualization (axes, bounding boxes, HUD)."""

import cv2
import numpy as np

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
# Rejected / not-on-allowlist detections (high-visibility magenta, BGR)
COLOR_TAG_OUTLINE_REJECTED = (255, 0, 255)
COLOR_TAG_OUTLINE_ALLOWED = COLOR_GREEN

# Axis colors: X=red, Y=green, Z=blue
AXIS_COLORS = {
    'x': COLOR_RED,
    'y': COLOR_GREEN,
    'z': COLOR_BLUE
}
MAX_DRAW_ABS_PX = 1_000_000.0
INT32_MIN = -2_147_483_648
INT32_MAX = 2_147_483_647


def draw_tag_outline(
    frame: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = COLOR_GREEN,
    thickness: int = 2
) -> None:
    """Draw a quadrilateral outline around detected tag corners.
    
    Args:
        frame: BGR image to draw on (modified in place).
        corners: 4x2 array of corner pixel coordinates.
        color: BGR color tuple.
        thickness: Line thickness in pixels.
    """
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_tag_id(
    frame: np.ndarray,
    tag_id: int,
    center: np.ndarray,
    color: tuple[int, int, int] | None = None,
    font_scale: float = 0.8,
    rejected: bool = False
) -> None:
    """Draw the tag ID near the tag center.
    
    Args:
        frame: BGR image to draw on (modified in place).
        tag_id: The AprilTag ID number.
        center: 2-element array with center pixel coordinates.
        color: BGR color tuple; defaults to yellow (valid) or magenta (rejected).
        font_scale: Font size scale factor.
        rejected: If True, label as not on allowlist (false-positive guard).
    """
    if rejected:
        text = f"ID:{tag_id} REJECT"
        use_color = color if color is not None else COLOR_TAG_OUTLINE_REJECTED
        pos = (int(center[0]) - 55, int(center[1]) - 15)
    else:
        text = f"ID:{tag_id}"
        use_color = color if color is not None else COLOR_YELLOW
        pos = (int(center[0]) - 20, int(center[1]) - 15)
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, use_color, 2, cv2.LINE_AA
    )


def _projected_row_to_pixel(projected: np.ndarray) -> tuple[int, int] | None:
    """One row from cv2.projectPoints -> integer pixel; None if not drawable.

    OpenCV 4.x Python bindings require plain Python int/float coordinates for drawing
    primitives; nested numpy scalars or 1-D slices can be rejected as ``pt2``.
    """
    arr = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 1:
        return None
    u = float(arr[0, 0])
    v = float(arr[0, 1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (int(round(u)), int(round(v)))


def _coerce_pixel_point(point: object) -> tuple[int, int] | None:
    """Convert a generic point-like value to a strict (int, int) tuple."""
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    u = float(arr[0])
    v = float(arr[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    if abs(u) > MAX_DRAW_ABS_PX or abs(v) > MAX_DRAW_ABS_PX:
        return None
    ui = int(round(u))
    vi = int(round(v))
    if ui < INT32_MIN or ui > INT32_MAX or vi < INT32_MIN or vi > INT32_MAX:
        return None
    return (ui, vi)


def _draw_line_safe(
    frame: np.ndarray,
    pt1: object,
    pt2: object,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a line only when points are valid OpenCV pixel tuples."""
    p1 = _coerce_pixel_point(pt1)
    p2 = _coerce_pixel_point(pt2)
    if p1 is None or p2 is None:
        return
    try:
        cv2.line(frame, p1, p2, color, thickness)
    except cv2.error as exc:
        # Keep detection loop alive even if OpenCV rejects an edge-case point payload.
        print(f"[visualization] skipped invalid line draw: p1={p1} p2={p2} err={exc}")


def draw_pose_axes(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
    axis_length: float = 0.02
) -> None:
    """Draw RGB coordinate axes on a detected tag.
    
    X=red, Y=green, Z=blue.
    
    Args:
        frame: BGR image to draw on (modified in place).
        camera_matrix: 3x3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        rotation_matrix: 3x3 rotation matrix from pose estimation.
        translation: [x, y, z] translation vector in meters.
        axis_length: Length of drawn axes in meters.
    """
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(translation)):
        return
    if translation[2] <= 1e-6:
        return
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = translation.reshape(3, 1)
    
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    x_axis = np.array([[axis_length, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, axis_length, 0]], dtype=np.float32)
    z_axis = np.array([[0, 0, axis_length]], dtype=np.float32)
    
    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
    x_2d, _ = cv2.projectPoints(x_axis, rvec, tvec, camera_matrix, dist_coeffs)
    y_2d, _ = cv2.projectPoints(y_axis, rvec, tvec, camera_matrix, dist_coeffs)
    z_2d, _ = cv2.projectPoints(z_axis, rvec, tvec, camera_matrix, dist_coeffs)
    
    origin_pt = _projected_row_to_pixel(origin_2d)
    x_pt = _projected_row_to_pixel(x_2d)
    y_pt = _projected_row_to_pixel(y_2d)
    z_pt = _projected_row_to_pixel(z_2d)
    origin_pt = _coerce_pixel_point(origin_pt)
    x_pt = _coerce_pixel_point(x_pt)
    y_pt = _coerce_pixel_point(y_pt)
    z_pt = _coerce_pixel_point(z_pt)
    if None in (origin_pt, x_pt, y_pt, z_pt):
        return
    
    _draw_line_safe(frame, origin_pt, x_pt, AXIS_COLORS['x'], 3)
    _draw_line_safe(frame, origin_pt, y_pt, AXIS_COLORS['y'], 3)
    _draw_line_safe(frame, origin_pt, z_pt, AXIS_COLORS['z'], 3)


def draw_cube_axes(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
    axis_length: float = 0.03
) -> None:
    """Draw fused cube axes using thicker, longer lines than per-tag axes."""
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(translation)):
        return
    if translation[2] <= 1e-6:
        return
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = translation.reshape(3, 1)
    axes_3d = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ], dtype=np.float32)
    points_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
    origin_pt = _projected_row_to_pixel(points_2d[0:1])
    x_pt = _projected_row_to_pixel(points_2d[1:2])
    y_pt = _projected_row_to_pixel(points_2d[2:3])
    z_pt = _projected_row_to_pixel(points_2d[3:4])
    origin_pt = _coerce_pixel_point(origin_pt)
    x_pt = _coerce_pixel_point(x_pt)
    y_pt = _coerce_pixel_point(y_pt)
    z_pt = _coerce_pixel_point(z_pt)
    if None in (origin_pt, x_pt, y_pt, z_pt):
        return
    _draw_line_safe(frame, origin_pt, x_pt, AXIS_COLORS['x'], 4)
    _draw_line_safe(frame, origin_pt, y_pt, AXIS_COLORS['y'], 4)
    _draw_line_safe(frame, origin_pt, z_pt, AXIS_COLORS['z'], 4)


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: tuple[int, int] = (10, 30)
) -> None:
    """Draw FPS counter on frame.
    
    Args:
        frame: BGR image to draw on (modified in place).
        fps: Frames per second value.
        position: (x, y) position for text.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2, cv2.LINE_AA
    )


def draw_detection_time(
    frame: np.ndarray,
    detection_ms: float,
    position: tuple[int, int] = (10, 60)
) -> None:
    """Draw detection time on frame.
    
    Args:
        frame: BGR image to draw on (modified in place).
        detection_ms: Detection time in milliseconds.
        position: (x, y) position for text.
    """
    text = f"Det: {detection_ms:.1f}ms"
    color = COLOR_GREEN if detection_ms < 15 else COLOR_YELLOW
    if detection_ms > 30:
        color = COLOR_RED
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def draw_tag_allowlist_status(
    frame: np.ndarray,
    allowed_count: int,
    rejected_count: int,
    position: tuple[int, int] = (10, 90)
) -> None:
    """Draw allowlist vs rejected detection counts."""
    text = f"Valid (allowlist): {allowed_count}"
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2, cv2.LINE_AA
    )
    rej = f"Rejected: {rejected_count}"
    rej_pos = (position[0], position[1] + 28)
    rej_color = COLOR_TAG_OUTLINE_REJECTED if rejected_count else COLOR_WHITE
    cv2.putText(
        frame, rej, rej_pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rej_color, 2, cv2.LINE_AA
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    detection_ms: float,
    allowed_count: int,
    rejected_count: int
) -> None:
    """Draw complete HUD with FPS, detection time, and allowlist status."""
    draw_fps(frame, fps)
    draw_detection_time(frame, detection_ms)
    draw_tag_allowlist_status(frame, allowed_count, rejected_count)


def draw_calibration_status(
    frame: np.ndarray,
    captured: int,
    target: int,
    position: tuple[int, int] = (10, 30)
) -> None:
    """Draw calibration capture status.
    
    Args:
        frame: BGR image to draw on (modified in place).
        captured: Number of captured frames.
        target: Target number of frames.
        position: (x, y) position for text.
    """
    text = f"Captured: {captured}/{target}"
    color = COLOR_GREEN if captured > 0 else COLOR_WHITE
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
    )


def draw_instructions(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] | None = None
) -> None:
    """Draw instruction text at bottom of frame.
    
    Args:
        frame: BGR image to draw on (modified in place).
        text: Instruction text to display.
        position: (x, y) position. Defaults to bottom-left.
    """
    if position is None:
        position = (10, frame.shape[0] - 20)
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA
    )


def draw_server_status(
    frame: np.ndarray,
    client_count: int,
    serve_enabled: bool,
) -> None:
    """Show TCP pose-server status (top-right). No-op if --serve not used."""
    if not serve_enabled:
        return
    _, w = frame.shape[:2]
    text = f"POSE TCP: {client_count} client{'s' if client_count != 1 else ''}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = max(10, w - tw - 12)
    y = 32
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GREEN, 2, cv2.LINE_AA,
    )


def draw_run_status(
    frame: np.ndarray,
    run_active: bool,
    run_id: int,
) -> None:
    """Prominent run indicator: READY vs RUN N ACTIVE."""
    h, w = frame.shape[:2]
    if run_active:
        text = f"RUN {run_id}  ACTIVE"
        color = COLOR_GREEN
    else:
        text = "READY  press SPACE to start"
        color = COLOR_YELLOW
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = min(h - 20, max(40, int(h * 0.12)))
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_glove_status(
    frame: np.ndarray,
    *,
    connected: bool,
    data_age_ms: float | None,
    buttons: list[int] | None,
) -> None:
    """Glove serial + IMU freshness and finger row (bottom-left). Optional device."""
    if not connected:
        return
    h, _w = frame.shape[:2]
    y0 = h - 72
    line_gap = 26

    if data_age_ms is None:
        line1 = "Glove: WAITING"
        c1 = COLOR_YELLOW
    else:
        stale_ms = 200.0
        if data_age_ms > stale_ms:
            line1 = f"Glove: STALE ({data_age_ms:.0f}ms)"
            c1 = COLOR_RED
        else:
            line1 = f"Glove: OK ({data_age_ms:.0f}ms)"
            c1 = COLOR_GREEN

    cv2.putText(
        frame, line1, (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c1, 2, cv2.LINE_AA,
    )

    if buttons is None:
        return
    parts = []
    labels = ("T", "I", "M", "R", "P")
    for i, lab in enumerate(labels):
        on = i < len(buttons) and int(buttons[i]) != 0
        parts.append(f"[{lab}:{'X' if on else ' '}]")
    line2 = "Grip: " + " ".join(parts)
    cv2.putText(
        frame, line2, (10, y0 + line_gap),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA,
    )
