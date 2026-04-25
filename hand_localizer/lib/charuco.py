"""ChArUco board creation and corner detection for camera calibration."""

import cv2
import numpy as np

# Default ChArUco board parameters
DEFAULT_SQUARES_X = 7
DEFAULT_SQUARES_Y = 5
# Full chessboard square (outer cell); inner ArUco marker is smaller (white border).
DEFAULT_SQUARE_LENGTH = 0.036  # 3.6 cm
# Printed black ArUco marker side length (the “tag-looking” part inside the square).
DEFAULT_MARKER_LENGTH = 0.026  # 2.6 cm; must be < square_length


def create_board(
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_length: float = DEFAULT_SQUARE_LENGTH,
    marker_length: float = DEFAULT_MARKER_LENGTH
) -> cv2.aruco.CharucoBoard:
    """Create a ChArUco board object.
    
    Args:
        squares_x: Number of squares in X direction.
        squares_y: Number of squares in Y direction.
        square_length: Size of each square in meters.
        marker_length: Size of ArUco markers in meters (must be smaller than square).
        
    Returns:
        ChArUco board object for detection and calibration.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )
    return board


def generate_board_image(
    board: cv2.aruco.CharucoBoard,
    pixel_size: tuple[int, int] = (1400, 1000),
    margin: int = 20
) -> np.ndarray:
    """Generate a printable image of the ChArUco board.
    
    Args:
        board: ChArUco board object.
        pixel_size: (width, height) of output image in pixels.
        margin: White margin around the board in pixels.
        
    Returns:
        Grayscale image array of the board.
    """
    board_image = board.generateImage(pixel_size, marginSize=margin)
    return board_image


def detect_corners(
    gray_frame: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    min_corners: int = 6
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect ChArUco corners in a grayscale frame.
    
    Args:
        gray_frame: Grayscale image (single channel uint8).
        board: ChArUco board object used for detection.
        min_corners: Minimum number of corners required for a valid detection.
        
    Returns:
        Tuple of (charuco_corners, charuco_ids) if enough corners found.
        Each is a numpy array. Returns None if not enough corners detected.
    """
    # OpenCV 4.7+ removed interpolateCornersCharuco; use CharucoDetector instead.
    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        aruco_dict = board.getDictionary()
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray_frame)
        if marker_ids is None or len(marker_ids) < 4:
            return None
        charuco_corners, charuco_ids, _, _ = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray_frame, board
        )
    else:
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray_frame)

    if charuco_ids is None or charuco_corners is None:
        return None
    if len(charuco_ids) < min_corners:
        return None

    return (charuco_corners, charuco_ids)


def calibrate_camera_charuco(
    all_charuco_corners: list[np.ndarray],
    all_charuco_ids: list[np.ndarray],
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list, list]:
    """Compute intrinsics from captured ChArUco corner lists per frame.

    ``opencv-contrib-python`` provides ``cv2.aruco.calibrateCameraCharuco``;
    plain ``opencv-python`` does not. In that case we pair 3D / 2D points with
    ``CharucoBoard.matchImagePoints`` and call ``cv2.calibrateCamera`` (same
    math as the official ChArUco calibration tutorial).
    """
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        return cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
            None,
            None,
        )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or img_pts is None or len(obj_pts) < 4:
            continue
        object_points.append(obj_pts)
        image_points.append(img_pts)

    if len(object_points) < 3:
        raise ValueError(
            "Need at least 3 frames with enough matched ChArUco corners after "
            f"matchImagePoints; got {len(object_points)} usable views."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


def draw_detected_corners(
    frame: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray
) -> np.ndarray:
    """Draw detected ChArUco corners on a frame.
    
    Args:
        frame: BGR image to draw on.
        charuco_corners: Detected corner positions.
        charuco_ids: IDs of detected corners.
        
    Returns:
        Frame with corners drawn (modifies in place and returns).
    """
    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    return frame
