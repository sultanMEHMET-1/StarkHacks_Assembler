# AprilTag Pose Estimation System - Implementation Plan

## Context

This is the first milestone of a teleoperation system. A glove with a 3D-printed cube on the wrist will have AprilTags on multiple faces. A webcam watches the glove and estimates the hand's 3D position and orientation. Later milestones will add IMU fusion, workspace calibration, inverse kinematics, and robot arm control. None of that is in scope here.

**This milestone's goal:** Detect AprilTags in a live 1080p 60fps webcam feed, compute the 6DOF pose (translation + rotation) of each detected tag relative to the camera, and display results as a real-time overlay. This is a sanity check that the core sensing pipeline works before building anything on top of it.

**Important constraints:**
- The camera runs at 1080p 60fps. Use the full resolution, do not downscale.
- Multiple tags will be visible simultaneously (cube has tags on multiple faces). The system must handle N tags per frame cleanly.
- Tag family is `tag16h5`. The cube faces are ~1.2 inches (~30mm), which is too small for `tag36h11` to detect reliably at working distance. `tag16h5` has fewer, larger cells per tag so it works better at this size. It only has 30 unique IDs, but the cube only needs 6, so that's fine. The higher false-positive rate is an accepted trade-off.
- This is a hackathon project. Favor working code over perfect architecture. No over-engineering.

---

## Project Structure

```
hand_localizer/
├── README.md
├── requirements.txt
├── .gitignore
├── calibrate.py                # Camera calibration utility (standalone script)
├── detect.py                   # Main detection loop (standalone script)
├── lib/
│   ├── __init__.py
│   ├── apriltag_detector.py    # AprilTag detection + pose estimation wrapper
│   ├── camera_params.py        # Load/save camera calibration data
│   ├── charuco.py              # ChArUco board creation and corner detection
│   ├── pose_math.py            # Rotation matrix -> Euler angles
│   └── visualization.py        # Drawing overlays (axes, bounding boxes, HUD)
```

No config module. No YAML config file. No dataclass hierarchies. Defaults are hardcoded constants at the top of each script. CLI arguments override the important ones (camera index, tag size, calibration file path). That's it.

---

## Dependencies (requirements.txt)

```
opencv-python>=4.8.0
pupil-apriltags>=1.0.4
numpy>=1.24.0
scipy>=1.11.0
```

**Fallback plan:** If `pupil-apriltags` fails to install (it has C dependencies that can be problematic), fall back to OpenCV's built-in ArUco module. OpenCV includes the `DICT_APRILTAG_16h5` dictionary and can do pose estimation via `cv2.aruco.estimatePoseSingleMarkers()`. It's slightly less accurate but has zero extra native dependencies. Try `pupil-apriltags` first. If the install fails or the import fails at runtime, print a clear warning and switch to the OpenCV backend.

---

## Coordinate System (document this in the README and in code comments)

OpenCV camera convention:
```
        ^ Y (down in image)
        |
        |
        +-------> X (right in image)
       /
      /
     v Z (forward, into scene)
```

- Translation vector: [x, y, z] in meters from camera lens to tag center.
  - x positive = tag is to the right of camera center
  - y positive = tag is below camera center
  - z positive = tag is in front of camera (always positive for visible tags)
- Euler angles: ZYX convention (roll, pitch, yaw) in degrees.
  - Roll: rotation about camera's Z axis (tag spins in-plane)
  - Pitch: rotation about camera's X axis (tag tilts toward/away)
  - Yaw: rotation about camera's Y axis (tag turns left/right)

This convention matters because later milestones will need to transform these into a workspace frame. Getting it wrong here means debugging phantom sign flips later.

---

## Design Decisions

### ChArUco over plain checkerboard for calibration
- Partial visibility is fine (30-40% of board visible still works, vs 100% required for checkerboard)
- No 180-degree ambiguity (ArUco markers on the board have unique IDs)
- Captures near edges are valid, which is where lens distortion is worst and matters most
- Fewer captures needed (15-20 vs 40-60)

### Do NOT undistort full frames
- `pupil-apriltags` accepts camera intrinsics (fx, fy, cx, cy) and handles the pose math internally
- Full-frame undistortion at 1920x1080 is expensive and unnecessary
- Pass the raw frame to the detector; pass intrinsics for pose calculation

### No YAML config system
- This is milestone 1 of a hackathon. Constants at the top of files, CLI args for the things that change. Refactor later if needed.

---

## Implementation Order

Build in this order. Each phase should produce something you can run and verify before moving on.

### Phase 1: Get the camera working (do this FIRST)

File: `detect.py` (initial skeleton)

Before touching AprilTags, confirm the camera actually opens at the resolution and framerate you expect. This step exists because camera setup is where things go wrong silently.

```python
import cv2
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # ALWAYS verify what you actually got
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
```

**Verify:** You see a live 1080p feed. The printed resolution matches what you expect. If the camera doesn't support 1080p60, you'll know now rather than after building everything else.

### Phase 2: AprilTag detection (no pose yet)

File: `lib/apriltag_detector.py`

Wrap `pupil-apriltags` (or the OpenCV fallback) in a class. For this phase, just detect tags and return their 2D corner positions and IDs. No pose estimation yet.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class TagDetection:
    tag_id: int
    corners: np.ndarray          # 4x2 array of corner pixel coords
    center: np.ndarray           # 2-element array, pixel coords
    # Pose fields, populated only when camera is calibrated:
    translation: np.ndarray | None = None   # [x, y, z] meters
    rotation_matrix: np.ndarray | None = None  # 3x3
    pose_error: float | None = None         # reprojection error from detector

class AprilTagDetector:
    def __init__(self, family="tag16h5", nthreads=4, quad_decimate=1.0, refine_edges=True):
        # Try pupil-apriltags first, fall back to OpenCV
        ...

    def detect(self, gray_frame, camera_params=None, tag_size=None):
        # Returns List[TagDetection]
        # If camera_params and tag_size are provided, populate pose fields
        # If not, return detections with pose fields as None
        ...
```

Key details:
- `quad_decimate=1.0` means no decimation. This preserves accuracy at the cost of speed. At 1080p this might be slow. If detection takes more than ~10ms per frame, try `quad_decimate=2.0` as a compromise. Log the detection time so you can make this decision with data.
- `refine_edges=True` for subpixel accuracy on the tag edges.
- `nthreads=4` to parallelize detection. Adjust based on your CPU.

**Verify:** Hold a printed `tag16h5` tag in front of the camera. The detector finds it and returns the correct tag ID and reasonable corner coordinates.

### Phase 3: Basic visualization

File: `lib/visualization.py`

Draw detected tags on the frame:
- Green quadrilateral around each tag using the corner points
- Tag ID text near the tag
- FPS counter in the top-left corner
- Per-frame detection time (milliseconds) in the top-left, below FPS

Wire this into `detect.py` so you see tags highlighted in the live feed.

**Verify:** Tags are outlined in green with their ID displayed. FPS and detection time are visible.

### Phase 4: Camera calibration

Files: `lib/charuco.py`, `lib/camera_params.py`, `calibrate.py`

`lib/charuco.py`:
- `create_board(squares_x=7, squares_y=5, square_length=0.03, marker_length=0.022)` -> ChArUco board object
- `generate_board_image(board, pixel_size=(1400, 1000))` -> image array (save as PNG for printing)
- `detect_corners(gray_frame, board)` -> (charuco_corners, charuco_ids) or None if not enough corners found

`lib/camera_params.py`:
- `save_calibration(filepath, camera_matrix, dist_coeffs, image_size, rms_error)` -> writes YAML
- `load_calibration(filepath)` -> dict with camera_matrix, dist_coeffs, image_size, rms_error. Raises FileNotFoundError or ValueError with clear messages if the file is missing or malformed.
- Expose a helper property or function to extract `(fx, fy, cx, cy)` from the camera matrix since that's what the detector needs.

`calibrate.py`:
- CLI args: `--camera` (int, default 0), `--output` (path, default "calibration_data.yaml"), `--num-images` (int, default 20), `--generate-board` (path, generates a printable PNG and exits)
- Live preview showing detected ChArUco corners highlighted on each frame
- SPACE captures a frame (only if enough corners are detected; reject and warn otherwise)
- Show a running count: "Captured 7/20"
- ESC computes calibration from captured frames, prints RMS reprojection error, saves to file
- Q quits without saving
- After calibration, print a clear summary: number of images used, RMS error, image resolution, and the path where the file was saved.

**Verify:** Print the generated ChArUco board. Run calibration, capture 20 images from various angles and distances. RMS error should be below 0.5 pixels. The saved YAML file contains camera_matrix (3x3) and dist_coeffs.

### Phase 5: Pose estimation

File: `lib/pose_math.py`

```python
from scipy.spatial.transform import Rotation

def rotation_matrix_to_euler(rotation_matrix):
    """Convert 3x3 rotation matrix to (roll, pitch, yaw) in degrees.
    Uses ZYX Euler convention (standard in robotics).
    """
    r = Rotation.from_matrix(rotation_matrix)
    # as_euler('ZYX') returns [yaw, pitch, roll]
    yaw, pitch, roll = r.as_euler('ZYX', degrees=True)
    return (roll, pitch, yaw)
```

Now update `detect.py`:
- On startup, load calibration data. If the file doesn't exist, print a clear error telling the user to run calibration first, and exit.
- Extract (fx, fy, cx, cy) from the camera matrix.
- Pass camera_params and tag_size to the detector's `detect()` call.
- For each detected tag with a valid pose, compute Euler angles.

Add to `lib/visualization.py`:
- `draw_pose_axes(frame, camera_matrix, dist_coeffs, rotation_matrix, translation, axis_length=0.02)`: Draw RGB axes on the tag. X=red, Y=green, Z=blue. Use `cv2.projectPoints` to project the 3D axis endpoints onto the image.

Update terminal output. Print pose data for ALL detected tags, formatted like:
```
[Tag 03] Pos: (x=+0.052, y=-0.031, z=+0.347) m | Rot: (R:+2.1, P:-5.3, Y:+12.7) deg | dt: 8.2ms
```

CLI args to add to `detect.py`:
- `--tag-size` (float, default 0.03, in meters — matches the ~1.2 inch cube faces)
- `--calibration` (path, default "calibration_data.yaml")
- `--print-interval` (int, default 10, print to terminal every N frames; set to 1 for continuous)

**Verify (this is your success criteria for the entire milestone):**
- Hold a tag16h5 tag in front of the camera
- Green outline and RGB axes are drawn on the tag
- Terminal shows position and orientation
- z value decreases as you move the tag closer to the camera
- Euler angles change as you tilt/rotate the tag
- Hold the tag still: readings should be stable (jitter < ~2mm position, < ~1 degree rotation)
- Move the tag smoothly: readings should track smoothly without jumps
- Show multiple tags at once: all are detected and reported independently
- Detection time per frame stays under ~15ms (you need headroom for later pipeline stages)

### Phase 6: Performance logging

Add a `--log-timing` flag to `detect.py`. When enabled, log per-frame timing to a CSV:

```
frame_number, capture_ms, detection_ms, total_ms, num_tags_detected
```

This isn't glamorous but you will need these numbers when you start building the control loop in later milestones. Knowing your real detection budget matters.

### Phase 7: README

Cover these and only these:
1. One-line description of what this does
2. Install instructions (venv, pip install)
3. How to generate and print the ChArUco board
4. How to run calibration (with what to expect: RMS < 0.5)
5. How to run the detector
6. Coordinate system diagram (copy from this plan)
7. Troubleshooting: camera not found, calibration file missing, pupil-apriltags install failure (and how to use OpenCV fallback)

---

## Critical Things to Get Right

1. **`pupil-apriltags` pose estimation requires three arguments.** If you call `detector.detect(gray)` without `estimate_tag_pose=True`, `camera_params=(fx, fy, cx, cy)`, and `tag_size=0.03`, you get 2D corners only. No pose. This is the most common mistake.

2. **Camera intrinsics must come from calibration, not defaults.** Using assumed focal lengths will produce inaccurate poses. The difference between calibrated and uncalibrated intrinsics can be centimeters of error at arm's length distances.

3. **Tag size must match the physical printed tag.** If your printed tag is 3.8cm but you tell the detector 4.0cm, every distance measurement is off by ~5%. Measure the printed tag with calipers if you can.

4. **The `tag16h5` family, not `tag36h11`.** The cube faces are ~30mm, which is too small for `tag36h11` cells to be reliably detected. `tag16h5` has larger cells that work at this physical size. It only provides 30 unique IDs and has a higher false-positive rate, but 30 IDs is plenty for a 6-face cube, and the false-positive trade-off is accepted.

5. **Multiple simultaneous detections are the normal case.** The cube has tags on multiple faces. When two or three are visible, the detector returns all of them. Every part of the pipeline (detection, visualization, terminal output, data structures) must handle a list of N detections, not assume there's exactly one.

6. **Log detection time per frame.** You need to know whether you're spending 5ms or 50ms on detection. This determines your real control loop budget for later milestones. Don't skip this, even if it feels like premature optimization. It's not optimization, it's measurement.

---

## What Is NOT In Scope

Do not build any of these yet:
- Fusing multiple tag poses into a single hand pose
- IMU integration
- Workspace calibration (camera-to-robot transform)
- Inverse kinematics or robot arm communication
- YAML config files or config dataclasses
- Unit tests (get it working first)
- Web interface or any GUI beyond the OpenCV window

If you have questions about any of this, ask before you start building.