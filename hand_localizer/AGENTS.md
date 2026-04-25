Read this file completely before writing any code. These are not suggestions.

## What This Project Is

A real-time AprilTag pose estimation system. A webcam detects AprilTags on a wearable glove (mounted on a 3D-printed cube on the wrist) and estimates the cube's 6DOF pose in camera-frame coordinates. Individual tags are detected via `pupil-apriltags`, and their raw 2D corner positions are fused into a single cube pose using `cv2.solvePnP`. This is part of a larger teleoperation system. Later milestones will add workspace calibration, IMU fusion, and robot arm control.

## Project Structure (Do Not Deviate)

```
hand_localizer/
├── CLAUDE.md                   # You are here
├── README.md
├── requirements.txt
├── .gitignore
├── calibrate.py                # Standalone CLI script for camera intrinsic calibration
├── detect.py                   # Standalone CLI script for the main detection loop
├── workspace_calibrate.py      # Standalone CLI script for workspace (camera-to-robot) calibration
├── lib/
│   ├── __init__.py
│   ├── apriltag_detector.py    # AprilTag detection + pose estimation wrapper
│   ├── calibration_board.py    # Workspace calibration board geometry, generation, and pose estimation
│   ├── camera_params.py        # Load/save camera calibration data
│   ├── charuco.py              # ChArUco board creation and corner detection
│   ├── cube_model.py           # Cube geometry definition (face size, tag size, ID-to-face mapping)
│   ├── cube_pose.py            # Fuse multi-tag detections into single cube pose via solvePnP
│   ├── pose_math.py            # Rotation matrix -> Euler angles
│   ├── visualization.py        # Drawing overlays (axes, bounding boxes, HUD)
│   └── workspace_calibration.py # Compute, save, load, apply camera-to-robot transform
```

Do not add directories like `config/`, `utils/`, `core/`, `src/`, or `tests/`. Do not create `__main__.py` files. Do not create a `setup.py` or `pyproject.toml`. The three entry points are `calibrate.py`, `detect.py`, and `workspace_calibrate.py` at the project root.

## Hard Rules

### Tag family is `tag16h5`

Not `tag36h11`. Not `tagStandard41h12`. Not any other family. The cube faces are ~1.2 inches (~30mm), which is too small for families with more bits per cell. `tag16h5` has fewer, larger cells that detect reliably at this physical size. It only has 30 unique IDs, but the cube only needs 6. The higher false-positive rate is an accepted trade-off. Do not change this without explicit instructions.

### Detection library is `pupil-apriltags` (PyPI package: `pyapriltags`)

Not MediaPipe. MediaPipe does not detect AprilTags. Not `apriltag` (the older Python wrapper). Not a custom detector. Use `pupil-apriltags`.

If `pupil-apriltags` fails to install or import (it has native C dependencies), fall back to OpenCV's built-in ArUco module using the `DICT_APRILTAG_16h5` dictionary and `cv2.aruco.estimatePoseSingleMarkers()`. Print a visible warning when using the fallback. Both backends must produce the same `TagDetection` dataclass output.

### Pose estimation requires ALL THREE of these arguments

```python
detections = detector.detect(
    gray_image,
    estimate_tag_pose=True,
    camera_params=(fx, fy, cx, cy),
    tag_size=0.03
)
```

If any of `estimate_tag_pose`, `camera_params`, or `tag_size` is missing, you get 2D corner detections only — no 3D pose. This is the single most common mistake with this library. Do not assume pose data will be present without passing all three.

### Camera intrinsics come from calibration, never from defaults

Do not hardcode focal length values. Do not estimate them from resolution. Do not use "typical" values. The intrinsics must be loaded from a calibration file produced by `calibrate.py`. If the calibration file doesn't exist at runtime, `detect.py` must exit with a clear error message telling the user to run `python calibrate.py` first.

### Default tag size is 0.03 meters

The cube faces are ~1.2 inches ≈ 0.03m. This is the default for `--tag-size`. The actual printed tag will be slightly smaller than the cube face (the tag doesn't fill the entire face), so users should measure with calipers and override with `--tag-size`. But 0.03 is the right default, not 0.04.

### Do not undistort full frames

`pupil-apriltags` handles distortion internally when you pass it camera intrinsics. Running `cv2.undistort()` on every 1920x1080 frame is expensive and unnecessary. Pass raw frames to the detector. Pass the calibrated intrinsics separately.

### Multiple tags per frame is the normal case

The cube has tags on multiple faces. When 2-3 faces are visible, the detector returns multiple detections. Every function, data structure, loop, and output format must handle a list of N detections. Never write code that assumes exactly one tag per frame. Never index `detections[0]` without checking length.

### Cube pose fusion uses solvePnP, not pose averaging

Do NOT average per-tag poses. Do NOT average Euler angles. Do NOT do weighted quaternion blending. The correct approach: take the raw 2D corner pixel positions from all detected cube tags, look up their known 3D positions from `cube_model.py`, and call `cv2.solvePnP` once. This uses the most accurate data available (corner detections) and avoids the information loss of averaging processed poses.

```python
success, rvec, tvec = cv2.solvePnP(
    object_points_3d,    # (N*4, 3) all visible tag corners in cube frame
    image_points_2d,     # (N*4, 2) corresponding pixel positions
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
```

### Cube geometry lives in `lib/cube_model.py` and nowhere else

All physical measurements of the cube (face size, tag size, tag-ID-to-face mapping) are constants at the top of `lib/cube_model.py`. These are the ONLY values the user needs to edit when the physical cube changes. Do not scatter cube dimensions across multiple files. Do not hardcode corner coordinates; compute them from `CUBE_FACE_SIZE` and `TAG_SIZE` so changing one number updates everything.

### Corner ordering must match between 3D model and detector output

`pupil-apriltags` returns corners in a specific order: `[bottom-left, bottom-right, top-right, top-left]` when viewing the tag face-on. The 3D corner positions in `cube_model.py` MUST use the same order. If these don't match, `solvePnP` produces wildly wrong poses. This is the single hardest bug to diagnose in this system. If the cube pose is obviously wrong (flipped, rotated 90 degrees, offset), check corner ordering first.

### Euler angle convention is ZYX (roll, pitch, yaw)

Use `scipy.spatial.transform.Rotation` for all rotation conversions. The call is:

```python
r = Rotation.from_matrix(rotation_matrix)
yaw, pitch, roll = r.as_euler('ZYX', degrees=True)
return (roll, pitch, yaw)
```

Note the reordering: `as_euler('ZYX')` returns `[yaw, pitch, roll]`, and we return `(roll, pitch, yaw)`. Do not use manual rotation matrix decomposition. Do not use a different Euler order.

### Coordinate system is OpenCV camera convention

```
        ^ Y (down in image)
        |
        |
        +-------> X (right in image)
       /
      /
     v Z (forward, into scene)
```

- x positive = tag is to the right of camera center
- y positive = tag is below camera center
- z positive = tag is in front of camera (always positive for visible tags)

Do not convert to a different coordinate system. Later milestones will handle frame transforms. This milestone outputs raw camera-frame poses.

## Architecture Constraints

### No configuration framework

No YAML config files. No TOML. No config dataclasses. No config module. Defaults are constants at the top of the relevant file. CLI arguments (via `argparse`) override the ones that matter. That's it.

If you feel the urge to create a `config/` directory or a `Settings` class, stop. This is a hackathon milestone, not a production service.

### No web server, no Flask, no FastAPI, no GUI frameworks

The only visual interface is the OpenCV `cv2.imshow()` window. No Tkinter, no PyQt, no browser-based dashboards.

### No unit test framework

Do not create `tests/`, `test_*.py`, `conftest.py`, or install pytest. Correctness is verified by running the detector against a physical tag and observing the output. If a function is wrong, you'll see it immediately in the video feed.

### No async, no threading (in the main detection loop)

The detection loop is synchronous: capture frame, convert to grayscale, detect tags, draw overlays, display, repeat. `pupil-apriltags` uses internal threading (controlled by `nthreads` parameter) for detection. Do not add your own threading, multiprocessing, or asyncio on top.

### Keep files short

No file should exceed 300 lines. No function should exceed 50 lines. If either limit is hit, the code is doing too much and should be split.

## Dependencies (Exact List)

```
opencv-python>=4.8.0
pupil-apriltags>=1.0.4
numpy>=1.24.0
scipy>=1.11.0
```

Do not add PyYAML (no YAML configs). Do not add click (use argparse). Do not add matplotlib (use OpenCV for display). Do not add any dependency not on this list without explicit approval.

Note: `calibrate.py` saves and loads calibration data. Use NumPy's `.npz` format or OpenCV's `cv2.FileStorage` for this. Both are already available from the existing dependencies. If you choose a simple YAML-like format for the calibration file, use `cv2.FileStorage` which is built into OpenCV — do not add PyYAML as a dependency.

## CLI Interfaces

### `calibrate.py`

```
python calibrate.py [--camera 0] [--output calibration_data.yaml] [--num-images 20] [--generate-board board.png]
```

- `--generate-board` creates a printable ChArUco board PNG and exits
- Without `--generate-board`, opens live camera feed for interactive calibration
- SPACE captures a frame (only if enough ChArUco corners detected)
- ESC finishes capture and computes calibration
- Q quits without saving
- Print RMS reprojection error when done (should be < 0.5 pixels)

### `detect.py`

```
python detect.py [--camera 0] [--calibration calibration_data.yaml] [--tag-size 0.03] [--print-interval 10] [--log-timing]
```

- Exits with clear error if calibration file is missing
- `--print-interval N` prints pose data to terminal every N frames
- `--log-timing` writes per-frame timing to a CSV file
- ESC or Q quits

## Terminal Output Format

```
[Tag 03] Pos: (x=+0.052, y=-0.031, z=+0.347) m | Rot: (R:+2.1, P:-5.3, Y:+12.7) deg | dt: 8.2ms
```

All detected tags printed, one line per tag. Always include detection time.

## Performance Requirements

- Log detection time (milliseconds) per frame. This is mandatory, not optional.
- If detection exceeds 15ms per frame consistently, try `quad_decimate=2.0` as a first fix.
- The `--log-timing` flag should write a CSV with columns: `frame_number, capture_ms, detection_ms, total_ms, num_tags_detected`

## What Is NOT In Scope

Do not build any of the following. They will be separate milestones:

- IMU sensor integration
- Inverse kinematics or robot arm communication
- Filtering or smoothing of pose data (Kalman filter, low-pass, etc.)
- Recording or saving pose data to datasets
- Network communication of any kind
- LeRobot API integration

If a task or question touches any of these, stop and ask before proceeding.

## Workspace Calibration Rules

### Calibration board tag IDs must not overlap with cube tag IDs

The cube uses tag IDs 0-5 (defined in `cube_model.py`). The calibration board uses tag IDs 10-14 (defined in `calibration_board.py`). These ranges must never overlap. If the detector sees a tag with ID 10-14, it belongs to the calibration board. If it sees 0-5, it belongs to the cube. Any other ID is ignored.

### Board geometry lives in `lib/calibration_board.py` and nowhere else

Same principle as `cube_model.py`: all physical measurements (tag spacings, tag size, page margins) are constants at the top of one file. Changing a spacing value updates all tag positions automatically.

### The workspace calibration routine is a separate script

`workspace_calibrate.py` is its own entry point, not embedded inside `detect.py` or `calibrate.py`. The scripts are independent. `detect.py` only loads a saved transform file; it does not run calibration itself.

### Board placement constraint: flat on table, top edge aligned with robot +X

The calibration assumes the board is placed flat (Z=0 in robot frame) with the TAG_A-to-TAG_B edge aligned with the robot's +X axis. This means the rotation component of the board-to-robot transform is identity. The user only enters a translational offset (x, y, z). This constraint must be clearly communicated to the user during calibration.

### Transform is saved as numpy .npz

Use `numpy.savez` / `numpy.load` for the 4x4 homogeneous transform matrix. Do not use pickle, JSON, or YAML for this.

## Common Mistakes to Avoid

1. **Calling `detector.detect(gray)` without the three pose arguments.** You get corners but no pose. This fails silently — no error, just `None` pose fields.

2. **Using MediaPipe for AprilTag detection.** MediaPipe detects hands and body poses. It does not detect AprilTags. They are completely different things.

3. **Forgetting to convert the frame to grayscale before passing to the detector.** `pupil-apriltags` expects a single-channel uint8 image. Passing a BGR frame will either crash or produce garbage.

4. **Using `tag36h11` because it's more common.** Read the tag family section above. We use `tag16h5` for physical size reasons.

5. **Undistorting the full frame before detection.** Unnecessary and slow. Pass raw frames + intrinsics.

6. **Assuming one tag per frame.** The cube shows multiple faces. Always iterate over the full detection list.

7. **Inventing focal length values instead of loading calibration.** This produces inaccurate pose estimates. Always load from calibration file.

8. **Creating a config system, test suite, or package structure.** This is a hackathon. Keep it simple. Constants + argparse.

9. **Adding PyYAML as a dependency.** Use `cv2.FileStorage` or numpy `.npz` for saving calibration data. Both are already available.

10. **Averaging per-tag poses instead of using solvePnP.** Per-tag poses are already the output of an internal PnP solve. Averaging them loses information. Use the raw 2D corners + known 3D geometry and solve once. See the cube pose fusion section above.

11. **Getting corner order wrong between the 3D model and the detector.** If the cube pose is flipped, offset, or rotated by 90 degrees, this is almost certainly the cause. Print the 2D corner positions for a single tag held upright and verify which corner is which before trusting the 3D model.

12. **Hardcoding cube corner coordinates.** Compute them from `CUBE_FACE_SIZE` and `TAG_SIZE` in `cube_model.py`. If someone changes the face size constant, all 24 corners should update automatically.

13. **Putting cube dimensions anywhere other than `lib/cube_model.py`.** All physical measurements live at the top of that one file. Do not scatter them across `detect.py`, `cube_pose.py`, or anywhere else.

14. **Using cube tag IDs (0-5) on the calibration board.** Calibration board tags use IDs 10-14. If ranges overlap, the detector can't tell cube tags from board tags, and both the cube pose and the board pose will be wrong.

15. **Embedding the workspace calibration routine inside `detect.py`.** Workspace calibration is `workspace_calibrate.py`. `detect.py` only loads a saved transform file. Keep them separate.

16. **Asking the user for rotation values during workspace calibration.** The board must be placed flat and axis-aligned. The rotation is identity. The user only enters x, y, z translation. Do not prompt for Euler angles or rotation matrices.

17. **Forgetting to invert the solvePnP output.** `solvePnP` returns the transform from object frame to camera frame ($T_{\text{board} \to \text{camera}}$). To go from camera to board, you need the inverse. Getting this backwards puts the cube pose on the wrong side of the camera.


## Finally: Git Workflow

- Commit after every moderate or larger change (new feature, bug fix, refactor, config update)
- Small changes (typos, single-line tweaks) may be batched
- Commit messages must be clear: describe **what** changed and **why**
- Never skip hooks or force-push unless explicitly instructed

