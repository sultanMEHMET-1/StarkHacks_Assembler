"""Cube geometry definition and AprilTag corner lookup."""

import json
import time

import numpy as np

_agent_bottom_logged = False  # reset per run

# === MEASURE AND UPDATE THESE ===
# Outer cube edge length (meters).
CUBE_FACE_SIZE = 0.030
# AprilTag black-border size on each face (meters).
TAG_SIZE = 0.027
# Tag id 5: extra distance along bottom-face normal from cube center (meters).
# Physical tag sits below the nominal bottom face; model uses half_face + this offset.
TAG_ID_5_EXTRA_OFFSET = 0.07
# Map physical tag IDs to cube faces.
TAG_ID_TO_FACE = {
    0: "top",   
    1: "front", 
    2: "right", 
    3: "left",  
    4: "back",  
    5: "bottom",
}
# === END OF USER-EDITABLE SECTION ===


_FACE_BASIS = {
    # right x up = outward normal
    "top": (
        np.array([0.0, +1.0, 0.0]),
        np.array([+1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ),
    "front": (
        np.array([0.0, 0.0, +1.0]),
        np.array([+1.0, 0.0, 0.0]),
        np.array([0.0, +1.0, 0.0]),
    ),
    "right": (
        np.array([+1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, +1.0, 0.0]),
    ),
    "left": (
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, +1.0]),
        np.array([0.0, +1.0, 0.0]),
    ),
    "back": (
        np.array([0.0, 0.0, -1.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, +1.0, 0.0]),
    ),
    # In-plane basis rotated 90° CW from original (+X,+Z) so tag frame matches detector output.
    "bottom": (
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([+1.0, 0.0, 0.0]),
    ),
}


def get_tag_corners_3d(tag_id: int) -> np.ndarray | None:
    """Return tag corners in cube-local coordinates as (4, 3) float64.

    Corner order matches pupil-apriltags:
    [bottom-left, bottom-right, top-right, top-left] when viewing face-on.
    Returns None if tag_id is not part of the cube model.
    """
    face_name = TAG_ID_TO_FACE.get(tag_id)
    if face_name is None:
        return None
    normal, face_right, face_up = _FACE_BASIS[face_name]

    half_face = CUBE_FACE_SIZE * 0.5
    half_tag = TAG_SIZE * 0.5
    depth = half_face + (TAG_ID_5_EXTRA_OFFSET if tag_id == 5 else 0.0)
    face_center = normal * depth

    corners = np.array([
        face_center - half_tag * face_right - half_tag * face_up,  # bottom-left
        face_center + half_tag * face_right - half_tag * face_up,  # bottom-right
        face_center + half_tag * face_right + half_tag * face_up,  # top-right
        face_center - half_tag * face_right + half_tag * face_up,  # top-left
    ], dtype=np.float64)
    # region agent log
    global _agent_bottom_logged
    if tag_id == 5 and not _agent_bottom_logged:
        _agent_bottom_logged = True
        with open(
            "/home/mercanmeh/code/Hackathons/StarkHacks/hand_localizer/.cursor/debug-8468bc.log",
            "a",
            encoding="utf-8",
        ) as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "8468bc",
                        "timestamp": int(time.time() * 1000),
                        "hypothesisId": "H1",
                        "location": "cube_model.py:get_tag_corners_3d",
                        "message": "bottom face basis and edge vectors",
                        "data": {
                            "face_right": face_right.tolist(),
                            "face_up": face_up.tolist(),
                            "edge_bl_to_br": (corners[1] - corners[0]).tolist(),
                            "edge_tl_to_bl": (corners[0] - corners[3]).tolist(),
                        },
                    }
                )
                + "\n"
            )
    # endregion
    return corners
