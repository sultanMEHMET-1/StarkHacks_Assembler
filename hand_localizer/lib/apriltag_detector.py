"""AprilTag detection and pose estimation wrapper.

Tries pupil-apriltags first, falls back to OpenCV ArUco if unavailable.
"""

from dataclasses import dataclass
import numpy as np
import cv2

# Try to import pupil-apriltags, fall back to OpenCV ArUco
_USE_PUPIL_APRILTAGS = False
try:
    from pupil_apriltags import Detector as PupilDetector
    _USE_PUPIL_APRILTAGS = True
except ImportError:
    print("WARNING: pupil-apriltags not available, using OpenCV ArUco fallback")
    print("         Install with: pip install pupil-apriltags")

# pupil-apriltags raw output: drop weak reads before pose / TagDetection
PUPIL_RAW_MAX_HAMMING = 1  # allow hamming distance <= this (0 = exact decode)
PUPIL_RAW_MIN_DECISION_MARGIN = 30


@dataclass
class TagDetection:
    """Detection result for a single AprilTag."""
    tag_id: int
    corners: np.ndarray  # 4x2 array of corner pixel coords
    center: np.ndarray  # 2-element array, pixel coords
    translation: np.ndarray | None = None  # [x, y, z] meters
    rotation_matrix: np.ndarray | None = None  # 3x3
    pose_error: float | None = None  # reprojection error from detector


class AprilTagDetector:
    """AprilTag detector with pose estimation.
    
    Uses pupil-apriltags when available, falls back to OpenCV ArUco.
    """
    
    def __init__(
        self,
        family: str = "tag16h5",
        nthreads: int = 4,
        quad_decimate: float = 1.0,
        refine_edges: bool = True
    ):
        """Initialize the detector.
        
        Args:
            family: Tag family to detect. Must be "tag16h5".
            nthreads: Number of threads for detection (pupil-apriltags only).
            quad_decimate: Decimation factor. 1.0 = full resolution, 2.0 = half.
            refine_edges: Enable subpixel edge refinement.
        """
        self.family = family
        self.using_pupil = _USE_PUPIL_APRILTAGS
        
        if _USE_PUPIL_APRILTAGS:
            self._detector = PupilDetector(
                families=family,
                nthreads=nthreads,
                quad_decimate=quad_decimate,
                refine_edges=refine_edges
            )
        else:
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_16h5
            )
            self._aruco_params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(
                self._aruco_dict, self._aruco_params
            )
    
    def detect(
        self,
        gray_frame: np.ndarray,
        estimate_tag_pose: bool = True,
        camera_params: tuple[float, float, float, float] | None = None,
        tag_size: float | None = None
    ) -> list[TagDetection]:
        """Detect AprilTags in a grayscale frame.
        
        Args:
            gray_frame: Single-channel uint8 grayscale image.
            estimate_tag_pose: When True, request per-tag pose from backend.
            camera_params: (fx, fy, cx, cy) camera intrinsics for pose estimation.
            tag_size: Physical tag size in meters for pose estimation.
            
        Returns:
            List of TagDetection objects. Pose fields are populated only if
            both camera_params and tag_size are provided.
        """
        if self.using_pupil:
            return self._detect_pupil(gray_frame, estimate_tag_pose, camera_params, tag_size)
        else:
            return self._detect_opencv(gray_frame, estimate_tag_pose, camera_params, tag_size)
    
    def _detect_pupil(
        self,
        gray_frame: np.ndarray,
        estimate_tag_pose: bool,
        camera_params: tuple[float, float, float, float] | None,
        tag_size: float | None
    ) -> list[TagDetection]:
        """Detect using pupil-apriltags.

        Raw list filtered by hamming and decision_margin before pose / TagDetection.
        """
        estimate_pose = estimate_tag_pose and camera_params is not None and tag_size is not None
        
        if estimate_pose:
            detections = self._detector.detect(
                gray_frame,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=tag_size
            )
        else:
            detections = self._detector.detect(gray_frame)

        detections = [
            d for d in detections
            if d.hamming <= PUPIL_RAW_MAX_HAMMING
            and float(d.decision_margin) > PUPIL_RAW_MIN_DECISION_MARGIN
        ]

        results = []
        for det in detections:
            corners = np.array(det.corners, dtype=np.float32)
            center = np.array(det.center, dtype=np.float32)
            
            translation = None
            rotation_matrix = None
            pose_error = None
            
            if estimate_pose and det.pose_t is not None and det.pose_R is not None:
                candidate_translation = det.pose_t.flatten()
                candidate_rotation = det.pose_R
                if self._is_valid_pose(candidate_rotation, candidate_translation):
                    translation = candidate_translation
                    rotation_matrix = candidate_rotation
                    pose_error = det.pose_err
            
            results.append(TagDetection(
                tag_id=det.tag_id,
                corners=corners,
                center=center,
                translation=translation,
                rotation_matrix=rotation_matrix,
                pose_error=pose_error
            ))
        
        return results
    
    def _detect_opencv(
        self,
        gray_frame: np.ndarray,
        estimate_tag_pose: bool,
        camera_params: tuple[float, float, float, float] | None,
        tag_size: float | None
    ) -> list[TagDetection]:
        """Detect using OpenCV ArUco fallback."""
        corners_list, ids, _ = self._aruco_detector.detectMarkers(gray_frame)
        
        if ids is None:
            return []
        
        estimate_pose = estimate_tag_pose and camera_params is not None and tag_size is not None
        
        camera_matrix = None
        dist_coeffs = None
        if estimate_pose:
            fx, fy, cx, cy = camera_params
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros(5)
        
        results = []
        for i, marker_id in enumerate(ids.flatten()):
            corners = corners_list[i].reshape(4, 2).astype(np.float32)
            center = corners.mean(axis=0)
            
            translation = None
            rotation_matrix = None
            pose_error = None
            
            if estimate_pose:
                obj_points = np.array([
                    [-tag_size/2, tag_size/2, 0],
                    [tag_size/2, tag_size/2, 0],
                    [tag_size/2, -tag_size/2, 0],
                    [-tag_size/2, -tag_size/2, 0]
                ], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    candidate_translation = tvec.flatten()
                    candidate_rotation, _ = cv2.Rodrigues(rvec)
                    if self._is_valid_pose(candidate_rotation, candidate_translation):
                        translation = candidate_translation
                        rotation_matrix = candidate_rotation
            
            results.append(TagDetection(
                tag_id=int(marker_id),
                corners=corners,
                center=center,
                translation=translation,
                rotation_matrix=rotation_matrix,
                pose_error=pose_error
            ))
        
        return results

    @staticmethod
    def _is_valid_pose(rotation_matrix: np.ndarray, translation: np.ndarray) -> bool:
        """Validate pose from detector before downstream math.

        Rejects degenerate / left-handed rotations that can occur on false positives
        and pose solves with poor geometry.
        """
        if rotation_matrix.shape != (3, 3):
            return False
        if translation.shape[0] != 3:
            return False
        if not np.isfinite(rotation_matrix).all() or not np.isfinite(translation).all():
            return False

        determinant = float(np.linalg.det(rotation_matrix))
        if determinant <= 0.0:
            return False

        orthogonality_error = np.linalg.norm(rotation_matrix.T @ rotation_matrix - np.eye(3))
        if orthogonality_error > 0.1:
            return False

        # In OpenCV camera coordinates, visible tags must be in front of camera.
        if float(translation[2]) <= 0.0:
            return False

        return True
