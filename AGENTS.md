# Glove Teleoperation System — Complete Project Summary

## What This Is

This is a hackathon project that lets a human wear a glove and remotely control a robot arm by moving their hand. Instead of using a physical "leader arm" (a mechanical puppet arm that's mechanically linked to the robot), the operator wears a glove with visual markers on it. A webcam tracks the glove, figures out where the hand is and how it's oriented, and tells the robot arm to mirror those movements.

The end goal is to use this system to collect training data for imitation learning. A human demonstrates a task (pick up an object, sort things, use a tool) many times while wearing the glove. Each demonstration is recorded. Those recordings are then used to train an AI model that can perform the task autonomously. The robot arm uses the Hugging Face LeRobot ecosystem for both control and training.

But for right now, the system is focused on the real-time control part: track the hand, move the arm.

## Why Not Just Use a Leader Arm?

Traditional teleoperation uses a "leader-follower" setup: two identical robot arms mechanically linked so that when you move one, the other mirrors it. This works but has problems. The leader arm is expensive (sometimes as much as the robot itself), mechanically fragile, and constrains your movement to whatever the arm's joints can do. Your hand can't move more naturally than the robot arm can. It also means training data is limited by the physical properties of the leader arm, not by human dexterity.

The glove replaces the leader arm entirely. It's cheap (webcam + printed markers + 3D-printed plastic), doesn't constrain your movements, and captures natural human motion. The trade-off is that tracking accuracy is worse than encoder-quality leader arm data, but for many practical tasks (pick and place, sorting, tool use), the accuracy is sufficient.

## The Two Codebases

The system is split into two independent projects that communicate over a TCP socket:

### 1. `hand_localizer` (you'll primarily be debugging this)

This is the vision and tracking system. It handles everything from camera input to producing a stream of "the hand is at position (x, y, z) with orientation (roll, pitch, yaw) in the robot's coordinate frame." It does NOT touch the robot arm at all. It doesn't know or care what kind of arm is connected, or even if one is connected.

### 2. `arm_mover` (the other codebase)

This is the robot control system. It connects to the hand localizer over TCP, receives pose data, converts it to motor commands via inverse kinematics, and sends those commands to a LeRobot arm. It does NOT do any vision or tracking. It just receives coordinates and moves the arm.

The separation is intentional. The hand localizer is designed to be a general-purpose hand tracking system that could be used with any robot arm (or any other consumer of position data). The arm mover is specific to the LeRobot hardware.

---

## How the Hand Localizer Works (The Important Part)

The hand localizer's job is to answer one question every frame: "Where is the hand, and how is it oriented, in the robot's coordinate frame?" Getting from raw camera pixels to that answer involves a pipeline of four stages. Each stage was built as a separate milestone.

### Stage 1: AprilTag Detection

The operator wears a glove with a small 3D-printed cube attached to the top of the wrist. Each face of the cube has an AprilTag printed on it. AprilTags are black-and-white square markers (like simplified QR codes) that are designed to be detected quickly and precisely by computer vision algorithms. Each tag has a unique ID.

A 1080p 60fps webcam watches the glove. Each frame is converted to grayscale and fed to the `pupil-apriltags` library, which finds any tags in the image and returns their 2D corner positions (pixel coordinates) and a per-tag 6DOF pose estimate (position and rotation relative to the camera).

Key details:
- The tag family is `tag16h5`. This family has fewer, larger cells per tag, which is important because the cube faces are only ~30mm across. A family with smaller cells (like `tag36h11`) would be unreliable at this physical size.
- The cube uses tag IDs 0 through 5, one per face.
- The camera must be calibrated first (lens distortion and focal length). Without calibrated intrinsics, the 3D pose estimates are inaccurate. Camera calibration uses a ChArUco board and is a one-time step per camera.
- The detector needs three specific arguments to produce 3D poses: `estimate_tag_pose=True`, `camera_params=(fx, fy, cx, cy)`, and `tag_size` in meters. Missing any one of these silently produces 2D-only results with no error message. This is the most common debugging pitfall.

**Files involved:** `lib/apriltag_detector.py`, `lib/camera_params.py`, `lib/charuco.py`, `calibrate.py`

### Stage 2: Cube Pose Fusion

When the webcam sees the cube, it typically detects 2 or 3 tags simultaneously (the faces that are pointing toward the camera). Each detection gives 2D corner positions in the image. But we don't want 3 separate poses — we want one pose for the cube's center.

The system fuses multiple tag detections into a single cube pose using OpenCV's `cv2.solvePnP`. Here's how:

1. For each detected tag, look up its tag ID in a geometry table that maps each ID to a cube face. The geometry table knows the 3D position of each tag corner relative to the cube's center (defined in `cube_model.py`).

2. Collect all the 2D corner pixel positions from all detected tags into one array. Collect the corresponding 3D corner positions (from the geometry table) into another array. If 2 tags are visible, that's 8 point correspondences. If 3 are visible, 12.

3. Call `cv2.solvePnP` with these point pairs. It finds the rotation and translation that best maps the known 3D points onto the observed 2D points, given the camera's intrinsics. The output is the cube's pose (position + rotation) in camera-frame coordinates.

This approach is better than averaging the per-tag poses because it uses the raw corner detections (the most accurate data available) instead of processed per-tag poses (which have already lost information through the per-tag PnP solve).

The cube geometry is defined by constants at the top of `cube_model.py`: `CUBE_FACE_SIZE` (the cube's edge length) and `TAG_SIZE` (the AprilTag's printed size, which is smaller than the face). Changing these constants automatically updates all 24 corner positions. If the cube pose is wildly wrong (flipped, rotated 90 degrees, axes pointing the wrong way), the first thing to check is the corner ordering — the 3D model's corner order must match what `pupil-apriltags` returns.

**Files involved:** `lib/cube_model.py`, `lib/cube_pose.py`

### Stage 3: Workspace Calibration (Camera-to-Robot Transform)

After Stage 2, the cube's pose is in camera-frame coordinates. The camera's coordinate system has its origin at the lens, X pointing right, Y pointing down, and Z pointing forward into the scene. This is useless for the robot arm, which has its own coordinate system centered at its base.

Workspace calibration finds the rigid transform (rotation + translation) that converts camera-frame coordinates to robot-frame coordinates. Once you have this 4x4 transformation matrix, converting is just a matrix multiplication applied every frame.

The calibration uses a printed calibration board with 5 `tag16h5` AprilTags (IDs 10–14, deliberately non-overlapping with the cube's IDs 0–5) arranged in an asymmetric L-shaped pattern. The asymmetry prevents 180-degree ambiguity. The calibration process:

1. The user places the calibration board flat on the table near the robot arm, with the top edge aligned with the robot's +X axis.

2. The camera detects the board's tags and computes the board's pose in camera frame (another `solvePnP` call, same technique as the cube).

3. The user measures and enters the position of the board's origin (center of the top-left tag) relative to the robot's base, in meters: x, y, z. Because the board is flat and axis-aligned, the rotation between the board frame and the robot frame is identity — only translation matters.

4. The system computes the full camera-to-robot transform by chaining: camera-to-board (from solvePnP, inverted) then board-to-robot (from the user's measurement).

5. The transform is saved to a `.npz` file. On subsequent startups, the user can load the saved transform or recalibrate.

**Files involved:** `lib/calibration_board.py`, `lib/workspace_calibration.py`, `workspace_calibrate.py`

### Stage 4: Pose Streaming

With stages 1–3 complete, the hand localizer can produce a stream of robot-frame poses. Stage 4 sends those poses to the arm mover over TCP.

The hand localizer runs a TCP server (on `localhost:9876` by default) in a background daemon thread. The server is enabled with the `--serve` flag on `detect.py`. Each frame where the cube is visible and a run is active, the server broadcasts a JSONL message (one JSON object per line, newline-delimited) to all connected clients:

```json
{"status": "running", "run_id": 1, "x": 0.15, "y": -0.03, "z": 0.05, "roll": 2.0, "pitch": -5.1, "yaw": 12.9, "num_tags": 2, "reproj_err": 0.31, "timestamp": 1713450000.123}
```

The arm mover connects as a client, reads these messages in a background thread, and always uses the most recent complete message (discarding any older messages that accumulated in the buffer). This "drain and use latest" pattern is critical — without it, the arm lags behind the hand by an ever-growing amount.

The server uses `TCP_NODELAY` to disable Nagle's algorithm (otherwise small messages get batched, adding 10–40ms of latency). All numpy floats are cast to Python `float()` before JSON serialization (numpy types aren't JSON-serializable).

**Files involved:** `lib/pose_server.py`, and `pose_client.py` on the arm mover side

---

## Run Control and Motion Mapping

The system has a "run" concept that governs when the arm follows the hand.

**Starting and stopping a run:** The operator presses SPACE in the hand localizer's OpenCV window to start a run, and SPACE again to stop it. A prominent green indicator ("RUN 1 ● ACTIVE") appears on the video feed during an active run. When no run is active, a yellow "READY — press SPACE to start" message is shown. ESC during an active run immediately stops the run and exits the program (the emergency stop).

**During a run:** Pose messages stream to the arm mover every frame where the cube is visible. If the cube goes out of view, no messages are sent, and the arm mover detects stale data and holds position.

**When a run stops:** Exactly one `{"status": "stopped", "run_id": N}` message is sent, then silence. The arm holds its last position.

**How position mapping works: relative (delta-based).** When a run starts, the arm mover captures two things: the hand's current position (from the first received pose) and the arm's current physical position (from the arm's own encoders/sensors). From then on, every pose is converted to a positional delta from the hand's starting position, and that delta is applied to the arm's starting position:

```
arm_target_xyz = arm_start_xyz + (current_hand_xyz - hand_start_xyz)
```

This means the hand and the arm don't need to be in the same physical location. The operator can hold their hand in a comfortable position, press SPACE, and the arm mirrors relative movements from wherever it already is. When a new run starts, new reference positions are captured, so the delta resets to zero and the arm doesn't move until the hand does.

**How orientation mapping works: absolute (direct pass-through).** Unlike position, orientation goes straight from the hand to the arm with no delta. If the hand is tilted 30 degrees, the arm's end effector tilts to 30 degrees. This is because orientation describes how the gripper is pointed, not where it is, so there's no collision risk. And relative orientation would be confusing for the operator — they'd have to mentally track "how much have I rotated since the run started" instead of just pointing where they want the gripper to aim.

---

## Data Flow Diagram

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                              hand_localizer process                                       │
│                                                                                           │
│  ┌──────────┐    ┌────────────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │  Webcam   │───>│ AprilTag       │───>│ Cube Pose    │───>│ Workspace     │              │
│  │ 1080p60   │    │ Detector       │    │ Fusion       │    │ Transform     │              │
│  │           │    │ (pupil-        │    │ (solvePnP    │    │ (camera →     │              │
│  │           │    │  apriltags)    │    │  on all      │    │  robot frame) │              │
│  │           │    │                │    │  visible     │    │               │              │
│  │           │    │ tag corners    │    │  tag corners │    │ 4x4 matrix    │              │
│  │           │    │ + per-tag pose │    │  + cube      │    │ multiply      │              │
│  └──────────┘    └────────────────┘    │  geometry)   │    └──────┬────────┘              │
│                                         └──────────────┘           │                       │
│                                                                     │ robot-frame pose     │
│                                                                     ▼                      │
│                                                          ┌──────────────────┐              │
│                                                          │ Pose Server      │              │
│                                                          │ (TCP, JSONL,     │──── only     │
│                                                          │  background      │     during   │
│                                                          │  thread)         │     active   │
│                                                          └────────┬─────────┘     run      │
│                                                                   │                        │
└───────────────────────────────────────────────────────────────────┼────────────────────────┘
                                                                    │
                                                         TCP localhost:9876
                                                           JSONL messages
                                                                    │
┌───────────────────────────────────────────────────────────────────┼────────────────────────┐
│                              arm_mover process                    │                        │
│                                                                   ▼                        │
│                                                          ┌──────────────────┐              │
│                                                          │ Pose Client      │              │
│                                                          │ (background      │              │
│                                                          │  reader thread,  │              │
│                                                          │  drain-latest)   │              │
│                                                          └────────┬─────────┘              │
│                                                                   │                        │
│                                                                   ▼                        │
│                                                      ┌────────────────────────┐            │
│                                                      │ Relative Motion Logic  │            │
│                                                      │                        │            │
│                                                      │ position: delta-based  │            │
│                                                      │   arm_tgt = arm_start  │            │
│                                                      │   + (hand - hand_start)│            │
│                                                      │                        │            │
│                                                      │ orientation: absolute  │            │
│                                                      │   pass through directly│            │
│                                                      └────────────┬───────────┘            │
│                                                                   │                        │
│                                                                   ▼                        │
│                                                          ┌──────────────────┐              │
│                                                          │ IK Solver        │              │
│                                                          │ (LeRobot API)    │              │
│                                                          │                  │              │
│                                                          │ x,y,z,r,p,y     │              │
│                                                          │ → motor angles   │              │
│                                                          └────────┬─────────┘              │
│                                                                   │                        │
│                                                                   ▼                        │
│                                                          ┌──────────────────┐              │
│                                                          │ Robot Arm        │              │
│                                                          │ (physical)       │              │
│                                                          └──────────────────┘              │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## The hand_localizer File-by-File

### Entry points (project root)

**`detect.py`** — The main program. Opens the camera, runs the detection loop, fuses cube pose, applies workspace transform, manages run state (SPACE to start/stop), broadcasts poses over TCP when `--serve` is enabled, draws overlays on the video feed. This is where most debugging happens. Key flags: `--serve` (enable TCP server), `--port` (TCP port), `--tag-size` (AprilTag physical size), `--calibration` (camera calibration file), `--workspace` (workspace transform file), `--skip-workspace` (run without workspace calibration), `--print-interval` (terminal output frequency), `--log-timing` (performance CSV).

**`calibrate.py`** — Camera intrinsic calibration utility. Shows a live feed, user holds a ChArUco board in front of the camera, presses SPACE to capture frames from various angles. When done, computes camera matrix and distortion coefficients and saves them. One-time per camera. Can also generate a printable ChArUco board image with `--generate-board`.

**`workspace_calibrate.py`** — Workspace calibration utility. Detects the calibration board (5 AprilTags, IDs 10–14), asks the user to enter the board's position relative to the robot base, computes and saves the camera-to-robot transform. Needs to be redone whenever the camera or robot moves.

### Library modules (`lib/`)

**`apriltag_detector.py`** — Wraps `pupil-apriltags` in a clean class. Takes a grayscale frame, returns a list of `TagDetection` dataclasses (tag_id, corners, center, and optionally translation/rotation if camera intrinsics are provided). Has a fallback to OpenCV's ArUco module if `pupil-apriltags` fails to install.

**`camera_params.py`** — Save and load camera calibration data (camera matrix, distortion coefficients). Provides a helper to extract `(fx, fy, cx, cy)` from the camera matrix, which is what the detector needs.

**`charuco.py`** — ChArUco board creation, corner detection, and camera calibration math. Used only by `calibrate.py`.

**`cube_model.py`** — Defines the physical geometry of the AprilTag cube. Constants at the top: `CUBE_FACE_SIZE`, `TAG_SIZE`, and `TAG_ID_TO_FACE` (which tag ID is on which face). The main function `get_tag_corners_3d(tag_id)` returns the 3D positions of a tag's 4 corners in cube-local coordinates, computed from the constants. If the cube pose is wrong, this file is the first place to look.

**`cube_pose.py`** — Takes a list of tag detections, filters for cube tags (IDs 0–5), looks up their 3D corners from `cube_model.py`, collects the corresponding 2D corners from the detections, and runs `cv2.solvePnP` to produce a single `CubePose` (translation, rotation, number of tags used, reprojection error). Returns None if no cube tags are detected.

**`pose_math.py`** — Rotation matrix to Euler angle conversion. Uses `scipy.spatial.transform.Rotation` with ZYX convention. Returns (roll, pitch, yaw) in degrees.

**`pose_server.py`** — TCP server that runs in a background daemon thread. Accepts client connections, broadcasts JSONL messages to all connected clients. The main thread calls `server.broadcast(pose_dict)` and the server handles the rest. Removes disconnected clients gracefully. Sets `TCP_NODELAY` on all connections.

**`calibration_board.py`** — Defines the workspace calibration board's geometry (5 tags in an L-shape, IDs 10–14, configurable spacings). Can generate a printable PNG. Can estimate the board's pose from detections using the same solvePnP approach as the cube.

**`workspace_calibration.py`** — Computes the camera-to-robot 4x4 transform from the board's pose in camera frame and the user's entered board-to-robot offset. Saves and loads the transform as a numpy `.npz` file. Provides `apply_transform()` to convert a pose from camera frame to robot frame.

**`visualization.py`** — All the OpenCV drawing functions: tag outlines (green quadrilaterals), pose axes (RGB lines, X=red, Y=green, Z=blue), FPS counter, detection time, run status indicator, server status indicator, frame indicator (CAMERA vs ROBOT).

---

## Coordinate Systems

There are three coordinate frames in play. Understanding which frame data is in at each point in the pipeline is essential for debugging.

### Camera frame (OpenCV convention)

This is what the detector outputs. Origin at the camera lens.

```
        ^ Y (down in image)
        |
        |
        +-------> X (right in image)
       /
      /
     v Z (forward, into scene)
```

- Z is always positive for visible objects (they're in front of the camera)
- Y-down is counterintuitive but it's the OpenCV standard
- All raw detection output (per-tag poses, cube pose before transform) is in this frame

### Cube-local frame

Defined in `cube_model.py`. Origin at the cube's geometric center. Used internally for defining where tag corners sit on each face. You almost never need to think about this directly — it's consumed by `solvePnP` to produce the cube's pose in camera frame.

### Robot frame

Defined by the robot arm. Origin at the base of the first joint (this is universal across robot arms). The workspace calibration transform converts from camera frame to robot frame. After the transform, all output is in this frame. This is what gets sent to the arm mover.

**When debugging, always ask: "What frame is this data in?"** If the cube pose looks wrong, is it wrong in camera frame (Stage 2 problem) or wrong after the workspace transform (Stage 3 problem)? Running with `--skip-workspace` outputs camera-frame poses, which isolates Stages 1–2 from Stage 3.

---

## Common Debugging Scenarios

### The cube pose is wildly wrong (flipped, rotated 90°, axes pointing the wrong way)

Almost always a corner ordering mismatch. `pupil-apriltags` returns corners in a specific order (bottom-left, bottom-right, top-right, top-left when viewing the tag face-on). The 3D model in `cube_model.py` must use the same order. Print the 2D corners for a single tag held upright and verify which index corresponds to which physical corner.

### The cube pose jumps or is unstable

Check `num_tags` in the output. If it's flickering between 1 and 2, tags are going in and out of detection, and the solvePnP solution is less constrained when only 1 tag is visible (the coplanar ambiguity problem). Also check the reprojection error — if it's above 2–3 pixels, the geometry constants in `cube_model.py` may not match the physical cube.

### "Error, more than one new minima found"

This comes from `pupil-apriltags`' internal pose estimator. It means a tag is in an ambiguous orientation (near edge-on). This affects the per-tag pose but should NOT affect the cube fusion, which uses 2D corners (still accurate) rather than per-tag poses. If you see this alongside bad cube poses, the problem is elsewhere.

### Huge numbers in axis drawing (billions of pixels)

The solvePnP output is garbage, and projecting 3D axes onto the image produces absurd coordinates. Check: corner ordering, tag size constant, cube face size constant. An order-of-magnitude error in tag size (e.g., 0.3 instead of 0.03) will produce this.

### Poses look correct in camera frame but wrong after workspace transform

The workspace calibration is wrong. Common causes: the calibration board wasn't flat or wasn't aligned with the robot's X axis, the entered x/y/z offset was wrong, or the solvePnP output wasn't inverted correctly (it gives board-to-camera, you need camera-to-board).

### The arm mover isn't receiving poses

Check: is `--serve` enabled? Is workspace calibration loaded (required for serving)? Is a run active (SPACE must be pressed)? Is the cube visible (no messages sent if cube is out of frame)? Test with `nc localhost 9876` to see raw messages.

### The arm lags behind the hand

The arm mover isn't draining the TCP buffer. It should always use the LAST complete line in the buffer, discarding everything older. If it processes messages in order, it falls behind.

---

## How to Run the System

### First time setup

```bash
cd hand_localizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate and print a ChArUco board
python calibrate.py --generate-board charuco_board.png

# Calibrate the camera (hold the printed board at various angles)
python calibrate.py

# Generate and print the workspace calibration board
python workspace_calibrate.py --generate-board calibration_board.png

# Calibrate the workspace (place board near robot, enter position)
python workspace_calibrate.py
```

### Running the teleoperation system

Terminal 1 (hand localizer):
```bash
cd hand_localizer
python detect.py --serve
```

Terminal 2 (arm mover):
```bash
cd arm_mover
python main.py --pose-host 127.0.0.1 --pose-port 9876
```

Then:
1. Position the arm where you want it to start
2. Hold your gloved hand in view of the camera
3. Press SPACE to start a run
4. Move your hand — the arm mirrors your movements
5. Press SPACE to stop
6. ESC to quit entirely (emergency stop)

---

## Dependencies

The hand localizer uses only four Python packages (plus the standard library):

```
opencv-python>=4.8.0     — Camera capture, image processing, solvePnP, visualization
pupil-apriltags>=1.0.4   — AprilTag detection (with OpenCV ArUco as fallback)
numpy>=1.24.0            — Array math, transform matrices
scipy>=1.11.0            — Rotation matrix ↔ Euler angle conversion
```

The TCP server and client use only Python builtins: `socket`, `threading`, `json`, `time`. No additional packages.

---

## Glossary

**AprilTag** — A type of fiducial marker (like a simplified QR code) designed for fast, robust detection by computer vision. Each tag has a unique ID encoded in its black-and-white pattern.

**tag16h5** — A specific AprilTag family with a 4x4 data grid. Has only 30 unique IDs but larger cells, making it more reliably detected at small physical sizes.

**ChArUco board** — A hybrid calibration target combining a checkerboard (for precise corner detection) with ArUco markers (for unambiguous identification). Used for camera intrinsic calibration.

**Camera intrinsics** — The internal optical properties of a camera: focal length (fx, fy), principal point (cx, cy), and lens distortion coefficients. Must be measured through calibration.

**solvePnP** — An OpenCV function that finds the 3D pose (rotation + translation) of an object given known 3D points on the object and their corresponding 2D positions in a camera image. Used twice in this system: once to fuse cube tags into a cube pose, and once to detect the calibration board's pose.

**6DOF (six degrees of freedom)** — A pose with 3 translational components (x, y, z position) and 3 rotational components (roll, pitch, yaw orientation). Fully describes where something is and how it's oriented.

**Inverse kinematics (IK)** — The math that converts a target end-effector position (x, y, z, roll, pitch, yaw) into specific joint angles for a robot arm. Handled by the arm mover, not the hand localizer.

**Workspace calibration** — The process of determining the rigid transform (rotation + translation) between the camera's coordinate frame and the robot arm's coordinate frame.

**JSONL (newline-delimited JSON)** — A format where each line of a file or stream is a complete JSON object. Used for the TCP protocol between the hand localizer and arm mover.

**Euler angles (ZYX convention)** — A way to represent 3D rotation as three sequential rotations: first around Z (yaw), then Y (pitch), then X (roll). The system uses degrees, not radians.

**TCP_NODELAY** — A socket option that disables Nagle's algorithm (which batches small messages for efficiency). Essential for real-time streaming where latency matters more than throughput.

**Run** — An operator-initiated session where the arm actively follows the hand. Started and stopped with SPACE. Each run gets an incrementing `run_id`. Positional reference points are re-captured at the start of each new run.