# hand_localizer

Real-time AprilTag pose estimation for wearable glove tracking.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate the Calibration Board

```bash
python calibrate.py --generate-board board.png
```

Print `board.png` at 100% scale (no fit-to-page). The board should be ~21cm x 15cm.

### 2. Calibrate Your Camera

```bash
python calibrate.py
```

**Controls:**
- **SPACE** - Capture frame (when corners are detected)
- **ESC** - Finish and compute calibration
- **Q** - Quit without saving

**Tips for good calibration:**
- Capture 15-20 images minimum
- Vary the board angle (tilt, rotate)
- Move the board to all areas of the frame, especially edges
- Keep the board steady when capturing
- Target RMS error < 0.5 pixels

### 3. Run the Detector

```bash
python detect.py
```

Hold a `tag16h5` AprilTag in front of the camera. You should see:
- Green outline around detected tags
- RGB coordinate axes (X=red, Y=green, Z=blue)
- HUD with FPS, detection time, and tag count
- Pose data printed to terminal

**Controls:**
- **ESC** or **Q** - Quit

### 4. Calibrate Camera-to-Robot Workspace Transform (Optional)

```bash
# Generate printable workspace board with tag IDs 10-14
python workspace_calibrate.py --generate-board workspace_board.png

# Run interactive workspace calibration and save camera_to_robot transform
python workspace_calibrate.py --output workspace_transform.npz
```

During workspace calibration, place the board flat on the table and align the
`ID 10 -> ID 11` edge with robot `+X`. Rotation is constrained to identity; you
only enter board origin translation `(x, y, z)` in robot frame.

Then load the transform in detection:

```bash
python detect.py --workspace-transform workspace_transform.npz
```

## Command Line Options

### calibrate.py

```
--camera SPEC        c920 (default, Logitech C920 on Linux), integer index, or /dev/videoN
--output PATH        Output calibration file (default: calibration_data.yaml)
--num-images N       Target number of captures (default: 20)
--generate-board PATH  Generate printable board and exit
```

On Linux, the default `c920` selects the Logitech HD Pro Webcam C920 (USB `046d:082d`) under `/dev/v4l/by-id/` so the laptop camera is not used. Use `--camera 0` for the first V4L device, or `--camera /dev/video2` for an explicit path.

### detect.py

```
--camera SPEC        c920 (default, Logitech C920 on Linux), integer index, or /dev/videoN
--calibration PATH   Calibration file (default: calibration_data.yaml)
--tag-size METERS    Physical tag size (default: 0.03 = 3cm)
--print-interval N   Print pose every N frames (default: 10)
--log-timing         Log timing data to CSV file
--workspace-transform PATH  Optional .npz camera_to_robot transform
```

### workspace_calibrate.py

```
--camera SPEC        c920 (default, Logitech C920 on Linux), integer index, or /dev/videoN
--calibration PATH   Camera calibration file (default: calibration_data.yaml)
--output PATH        Output transform file (default: workspace_transform.npz)
--generate-board [PATH]  Generate printable board image and exit
```

## Coordinate System

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

**Translation vector** (meters from camera to tag):
- x positive = tag is to the right of camera center
- y positive = tag is below camera center
- z positive = tag is in front of camera

**Euler angles** (ZYX convention, degrees):
- Roll: rotation about Z axis (tag spins in-plane)
- Pitch: rotation about X axis (tag tilts toward/away)
- Yaw: rotation about Y axis (tag turns left/right)

## Troubleshooting

### Camera not found

```
ERROR: Cannot open camera ...
```

- Default is `c920` (Logitech C920 on Linux). For the built-in webcam use `--camera 0`
- Try different indices: `--camera 1`, `--camera 2`
- Or pass the device path from `ls -l /dev/v4l/by-id/`
- Check if another application is using the camera
- On Linux, verify permissions: `ls -la /dev/video*`
- Auto-detection reads `/dev/v4l/by-id/*` and `/sys/class/video4linux/*/name` for “C920” / `046d:082d`

### Qt / Wayland warnings (OpenCV window)

If you see errors about `wayland` Qt plugins or missing font directories from OpenCV’s Qt backend, `calibrate.py` and `detect.py` set `QT_QPA_PLATFORM=xcb` before loading OpenCV so the window uses X11/Wayland-XWayland when available. If the window still fails, try running from an X11 session or `export QT_QPA_PLATFORM=xcb` before `python`.

### Calibration file missing

```
ERROR: Calibration file not found: calibration_data.yaml
       Run 'python calibrate.py' first to create it.
```

Run calibration before detection:
```bash
python calibrate.py
python detect.py
```

### pupil-apriltags installation failed

If you see:

```
WARNING: pupil-apriltags not available, using OpenCV ArUco fallback
```

The system will use OpenCV's built-in ArUco detector. This works but may be slightly less accurate. To install pupil-apriltags:

```bash
# Ubuntu/Debian
sudo apt-get install libapriltag-dev
pip install pupil-apriltags

# macOS
brew install apriltag
pip install pupil-apriltags
```

### High detection time (>15ms)

If detection consistently takes >15ms:

1. Reduce resolution (not recommended for accuracy)
2. Increase decimation in `lib/apriltag_detector.py`:
   ```python
   quad_decimate=2.0  # Instead of 1.0
   ```
3. Reduce `nthreads` if CPU is thermal throttling

### Tags not detected

- Verify you're using `tag16h5` family tags (not `tag36h11`)
- Ensure good lighting (avoid glare, shadows)
- Check tag size is set correctly with `--tag-size`
- Make sure printed tags have clean white borders

## Terminal Output Format

```
[Tag 03] Pos: (x=+0.052, y=-0.031, z=+0.347) m | Rot: (R:+2.1, P:-5.3, Y:+12.7) deg | dt: 8.2ms
```

- `Tag 03`: AprilTag ID
- `Pos`: Translation in meters (camera to tag center)
- `Rot`: Euler angles in degrees (Roll, Pitch, Yaw)
- `dt`: Detection time in milliseconds

## Project Structure

```
hand_localizer/
├── calibrate.py          # Camera calibration CLI
├── detect.py             # Main detection loop CLI
├── workspace_calibrate.py  # Workspace calibration CLI
├── lib/
│   ├── apriltag_detector.py  # Detection wrapper
│   ├── calibration_board.py  # Workspace board geometry + pose estimation
│   ├── camera_params.py      # Calibration I/O
│   ├── charuco.py            # ChArUco board utilities
│   ├── cube_model.py         # Cube dimensions + ID-to-face mapping
│   ├── cube_pose.py          # Multi-tag cube solvePnP fusion
│   ├── pose_math.py          # Rotation conversions
│   ├── visualization.py      # Drawing overlays
│   └── workspace_calibration.py  # Transform math and .npz I/O
├── requirements.txt
└── README.md
```
