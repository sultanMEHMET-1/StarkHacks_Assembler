"""Load and save camera calibration data using OpenCV FileStorage."""

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Logitech HD Pro Webcam C920 (USB ID 046d:082d) — prefer this over laptop camera.
_C920_BY_ID_GLOBS = (
    "/dev/v4l/by-id/usb-Logitech_HD_Pro_Webcam_C920*-video-index0",
    "/dev/v4l/by-id/*Logitech*C920*-video-index0",
    "/dev/v4l/by-id/*046d*_082d*-video-index0",
    "/dev/v4l/by-id/*046d*082d*-video-index0",
)


def _is_c920_by_id_name(name: str) -> bool:
    """Match udev symlink names for the C920 (patterns vary by distro / hub)."""
    low = name.lower()
    if "metadata" in low or "snd-" in low:
        return False
    is_usb_ids = "082d" in low and "046d" in low
    is_brand = ("logitech" in low and "c920" in low) or ("c920" in low and "046d" in low)
    if not (is_usb_ids or is_brand):
        return False
    # Prefer primary capture node (index0); avoid IR/aux streams when named distinctly
    if "video-index0" in low:
        return True
    if "index0" in low and "video" in low:
        return True
    return False


def _find_c920_scan_by_id_dir() -> str | None:
    """Enumerate /dev/v4l/by-id (more reliable than a few glob patterns)."""
    d = Path("/dev/v4l/by-id")
    if not d.is_dir():
        return None
    hits: list[str] = []
    for p in d.iterdir():
        if p.name.startswith("."):
            continue
        if not _is_c920_by_id_name(p.name):
            continue
        path = str(p)
        if os.path.exists(path):
            hits.append(path)
    return sorted(hits)[0] if hits else None


def _find_c920_from_sysfs() -> str | None:
    """Match kernel V4L2 card name (works when by-id symlinks are missing)."""
    base = Path("/sys/class/video4linux")
    if not base.is_dir():
        return None
    indices: list[int] = []
    for vdir in base.iterdir():
        if not vdir.is_dir() or not vdir.name.startswith("video"):
            continue
        suffix = vdir.name[5:]
        if not suffix.isdigit():
            continue
        name_file = vdir / "name"
        if not name_file.is_file():
            continue
        try:
            text = name_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
        except OSError:
            continue
        if "c920" in text or ("hd pro webcam" in text and "logitech" in text):
            indices.append(int(suffix))
    if not indices:
        return None
    n = min(indices)
    dev = f"/dev/video{n}"
    return dev if os.path.exists(dev) else None


def find_logitech_c920_video_device() -> str | None:
    """Return a Linux path to the C920 capture device, or None if not found."""
    if sys.platform != "linux":
        return None
    seen: set[str] = set()

    def take(path: str | None) -> str | None:
        if not path or path in seen:
            return None
        seen.add(path)
        return path if os.path.exists(path) else None

    for pattern in _C920_BY_ID_GLOBS:
        for path in sorted(glob.glob(pattern)):
            got = take(path)
            if got:
                return got

    got = take(_find_c920_scan_by_id_dir())
    if got:
        return got

    return take(_find_c920_from_sysfs())


def parse_camera_arg(value: str) -> int | str:
    """Parse --camera: c920/auto, numeric index, or /dev/videoN path."""
    s = value.strip()
    key = s.lower()
    if key in ("c920", "auto", "logitech", "logitech-c920"):
        found = find_logitech_c920_video_device()
        if found:
            return found
        print(
            "WARNING: Logitech C920 (046d:082d) not found (checked /dev/v4l/by-id "
            "and /sys/class/video4linux/*/name). Using camera index 0. "
            "Pass e.g. --camera /dev/video2 if needed."
        )
        return 0
    if s.startswith("/dev/"):
        return s
    if s.isdigit():
        return int(s)
    raise ValueError(
        f"Invalid --camera {value!r}: use c920, a non-negative integer, or a /dev/... path"
    )


def open_video_capture(camera: int | str) -> cv2.VideoCapture:
    """Open a camera by V4L2 device path or numeric index (CAP_V4L2 on Linux)."""
    use_v4l2 = sys.platform == "linux"
    if isinstance(camera, str) and use_v4l2:
        cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    if isinstance(camera, str):
        return cv2.VideoCapture(camera)
    if use_v4l2:
        cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera)


def fourcc_to_str(value: float) -> str:
    """Decode OpenCV FOURCC to a short string (e.g. MJPG)."""
    v = int(value)
    chars = [chr((v >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(chars).strip() or "unknown"


def configure_capture_mjpeg(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    preferred_fps: float = 60.0,
    fallback_fps: float = 30.0,
) -> tuple[int, int, float, str]:
    """Request MJPEG (Linux) and high FPS; returns negotiated width, height, fps, FOURCC."""
    if sys.platform == "linux":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, preferred_fps)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0 and actual_fps < 20:
        cap.set(cv2.CAP_PROP_FPS, fallback_fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    return w, h, fps, fourcc


def apply_manual_exposure(cap: cv2.VideoCapture, exposure_value: float) -> float:
    """Try to force manual exposure; returns camera-reported exposure value."""
    # V4L2 commonly uses 0.25=manual / 0.75=auto; other backends use 0/1.
    for mode in (0.25, 1.0, 0.0):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))
    return float(cap.get(cv2.CAP_PROP_EXPOSURE))


def save_calibration(
    filepath: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms_error: float
) -> None:
    """Save camera calibration data to a YAML file.
    
    Args:
        filepath: Path to save the calibration file.
        camera_matrix: 3x3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        image_size: (width, height) of calibration images.
        rms_error: RMS reprojection error from calibration.
    """
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("rms_error", rms_error)
    fs.release()


def load_calibration(filepath: str) -> dict:
    """Load camera calibration data from a YAML file.
    
    Args:
        filepath: Path to the calibration file.
        
    Returns:
        Dictionary containing:
        - camera_matrix: 3x3 numpy array
        - dist_coeffs: distortion coefficients numpy array
        - image_size: (width, height) tuple
        - rms_error: float
        
    Raises:
        FileNotFoundError: If the calibration file doesn't exist.
        ValueError: If the calibration file is malformed.
    """
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    
    if not fs.isOpened():
        raise FileNotFoundError(f"Calibration file not found: {filepath}")
    
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())
    rms_error = fs.getNode("rms_error").real()
    
    fs.release()
    
    if camera_matrix is None or dist_coeffs is None:
        raise ValueError(f"Malformed calibration file: {filepath}")
    
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": (image_width, image_height),
        "rms_error": rms_error
    }


def get_intrinsics(camera_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from camera matrix.
    
    This is the format required by pupil-apriltags for pose estimation.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix.
        
    Returns:
        Tuple of (fx, fy, cx, cy) - focal lengths and principal point.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return (fx, fy, cx, cy)
