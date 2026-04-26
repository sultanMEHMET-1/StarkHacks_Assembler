# Milestone 6: Glove IMU Integration & Palm Offset — Hand Localizer

## Read `CLAUDE.md` Before Writing Any Code. Then Read This Entire Prompt.

You are modifying the hand_localizer project to integrate data from an ESP32 glove that streams IMU readings and button states over Bluetooth. The ESP32 is a separate project — you are NOT writing firmware here. You are writing the Python side that receives and uses the data.

---

## Context: What Exists

The hand_localizer currently:
1. Detects AprilTags on a cube mounted on the wrist.
2. Fuses them into a single cube pose via `solvePnP`.
3. Transforms the cube pose to robot-frame coordinates via workspace calibration.
4. Streams the pose over TCP to the arm mover when a run is active.

The cube pose gives the position of the **cube** in robot-frame coordinates. But the cube sits on top of the wrist. The robot arm's end effector should match the position and orientation of the **palm**, not the cube. There are two offsets between the cube and the palm:

1. **Cube-to-wrist offset**: A fixed vector from the cube center downward (in the cube's local frame) to the wrist surface. This doesn't change with wrist angle.
2. **Wrist-to-palm offset**: A vector from the wrist along the hand toward the palm center. The *length* of this vector is constant, but its *direction in world space* changes when the wrist flexes up or down. The IMU on the back of the hand measures this wrist flexion angle.

Additionally, the glove has 5 buttons (one per finger) that represent grip state. These need to be included in the pose stream so the arm mover can control the gripper.

---

## What the ESP32 Sends

The ESP32 is paired with the computer over Bluetooth Classic SPP. It appears as a serial port. It sends JSONL at 50Hz:

```json
{"roll":12.3,"pitch":-5.1,"yaw":45.2,"btn":[1,0,0,1,0],"t":12345}
```

| Field | Type | Description |
|-------|------|-------------|
| `roll` | float (degrees) | Hand roll (tilt left/right) |
| `pitch` | float (degrees) | Wrist flexion (up = positive, down = negative) |
| `yaw` | float (degrees) | Hand rotation about vertical axis (drifts, gyro-only) |
| `btn` | array of 5 ints | [thumb, index, middle, ring, pinky], 1=pressed 0=released |
| `t` | int (ms) | ESP32 `millis()` timestamp |

**Important:** The IMU's pitch is the wrist flexion angle. This is the value that changes the wrist-to-palm offset direction. When pitch=0, the hand is flat and the palm is directly in front of the wrist. When pitch=+30, the fingers are pointing upward and the palm is in front of and above the wrist.

---

## New Files to Create

```
lib/
├── glove_reader.py         # NEW: Bluetooth serial reader, background thread
├── palm_model.py           # NEW: Offset geometry (cube → wrist → palm)
```

## Files to Modify

```
detect.py                   # Add --glove-port flag, integrate glove data into pipeline
lib/visualization.py        # Show glove connection status and button states on overlay
```

## New Dependency

Add `pyserial` to `requirements.txt`:

```
pyserial>=3.5
```

This is the ONLY new dependency. `pyserial` is the standard Python library for serial port communication. The Bluetooth SPP serial port is opened exactly like a USB serial port — no Bluetooth-specific library is needed on the Python side.

---

## `lib/glove_reader.py` — Bluetooth Serial Reader

This module connects to the ESP32 glove over a Bluetooth serial port and continuously reads IMU + button data in a background thread. It exposes the latest readings via thread-safe getters. If the connection drops or data stops arriving, it reports stale data rather than crashing.

This follows the exact same pattern as `pose_client.py`'s `_read_loop`: background thread reads continuously, stores the latest complete message, caller polls for it.

```python
import serial
import threading
import json
import time
from dataclasses import dataclass, field


@dataclass
class GloveState:
    """Immutable snapshot of the glove's current state."""
    roll: float = 0.0          # degrees
    pitch: float = 0.0         # degrees (wrist flexion, this is the important one)
    yaw: float = 0.0           # degrees (drifts, use with caution)
    buttons: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])  # [thumb, index, middle, ring, pinky]
    timestamp_ms: int = 0      # ESP32 millis() timestamp
    received_at: float = 0.0   # time.time() when this reading was received on the computer


class GloveReader:
    """
    Reads IMU and button data from the ESP32 glove over Bluetooth serial.
    
    Usage:
        reader = GloveReader("/dev/rfcomm0")  # or "COM5" on Windows
        reader.start()
        
        # In your main loop:
        state = reader.get_latest()
        if state is not None:
            print(f"pitch={state.pitch}, buttons={state.buttons}")
        
        # On shutdown:
        reader.stop()
    """

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.latest_state = None
        self.state_lock = threading.Lock()
        self.connected = False

    def start(self):
        """
        Open the serial port and start reading in a background thread.
        Raises serial.SerialException if the port can't be opened.
        """
        self.serial_conn = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1.0  # 1 second read timeout so the thread can check self.running
        )
        self.connected = True
        self.running = True

        reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        reader_thread.start()

        print(f"Glove reader started on {self.port}")

    def _read_loop(self):
        """
        Background thread: continuously read lines from serial,
        parse JSON, store the latest GloveState.
        
        If the serial port produces garbage or partial lines, skip them.
        If the port disconnects, set connected=False and exit the loop.
        """
        while self.running:
            try:
                # readline() blocks until \n or timeout
                raw = self.serial_conn.readline()
                if not raw:
                    continue  # timeout, no data

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines (partial messages, garbage)

                state = GloveState(
                    roll=float(data.get("roll", 0)),
                    pitch=float(data.get("pitch", 0)),
                    yaw=float(data.get("yaw", 0)),
                    buttons=data.get("btn", [0, 0, 0, 0, 0]),
                    timestamp_ms=int(data.get("t", 0)),
                    received_at=time.time(),
                )

                with self.state_lock:
                    self.latest_state = state

            except (serial.SerialException, OSError):
                self.connected = False
                print("Glove serial connection lost.")
                break

    def get_latest(self) -> GloveState | None:
        """
        Return the most recently received glove state, or None if
        no data has been received yet.
        
        If no new data arrives, this keeps returning the SAME last-known
        state (not None). The caller can check received_at to determine
        freshness.
        """
        with self.state_lock:
            return self.latest_state  # None only if no data ever received

    def get_data_age(self) -> float | None:
        """
        Return how many seconds ago the latest reading was received,
        or None if no data ever received.
        """
        with self.state_lock:
            if self.latest_state is not None:
                return time.time() - self.latest_state.received_at
            return None

    def stop(self):
        """Clean shutdown."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except OSError:
                pass
        self.connected = False
        print("Glove reader stopped.")
```

### Key implementation details

**`readline()` with a 1-second timeout.** This is important. Without a timeout, the thread blocks forever on `readline()` and can't check `self.running` to exit cleanly. The 1-second timeout means the thread checks `self.running` at least once per second, allowing clean shutdown.

**The serial port path varies by OS:**
- Linux: `/dev/rfcomm0` (after pairing and binding with `rfcomm`)
- macOS: `/dev/tty.GloveIMU-ESP32SPP` or similar
- Windows: `COM5` or whatever Windows assigns

The port must be a CLI argument on `detect.py`, not hardcoded. There is no reliable cross-platform way to auto-discover Bluetooth serial ports.

**When data stops arriving**, `get_latest()` keeps returning the last known state. It does NOT return None. The caller checks freshness via `get_data_age()`. This is the "keep reading the same one" behavior requested — if the Bluetooth drops a few packets, the system keeps using the last good reading rather than losing orientation.

**`get_latest()` returns None ONLY if no data has EVER been received** (the very start of the connection). Once the first reading arrives, it's always available.

---

## `lib/palm_model.py` — Offset Geometry

This module computes the palm position from the cube pose and the IMU's wrist pitch angle.

### The geometry

```
Side view of the hand (wrist pitch = 0, hand flat):

        ┌─────┐
        │CUBE │  ← AprilTag cube, sitting on top of wrist
        └──┬──┘
           │  ← CUBE_TO_WRIST_OFFSET (fixed, straight down in cube-local frame)
    ───────┼─────────────────
    WRIST  ●──────────────●  PALM CENTER
           │              │
           └──────────────┘
           ← WRIST_TO_PALM_DISTANCE (fixed length, direction depends on wrist pitch)


Side view of the hand (wrist pitch = +30°, fingers pointing up):

        ┌─────┐
        │CUBE │
        └──┬──┘
           │
    ───────┼─────
    WRIST  ●
            \
             \  ← same distance, but now angled upward
              \
               ●  PALM CENTER (now higher than before)
```

### Constants (at the top of the file, clearly labeled)

```python
"""
Palm offset model.

MODIFY THESE VALUES to match your physical glove.
Measure with a ruler. All units are meters.
"""

# === MEASURE AND UPDATE THESE ===

# Distance from the cube center to the wrist surface.
# Measured along the cube's local -Z axis (from cube down toward hand).
# This is roughly half the cube height plus any mounting offset.
CUBE_TO_WRIST_DISTANCE = 0.025  # 25mm — PLACEHOLDER, measure this

# Distance from the wrist joint to the center of the palm.
# Measured along the hand when the hand is flat.
WRIST_TO_PALM_DISTANCE = 0.080  # 80mm — PLACEHOLDER, measure this

# === END OF USER-EDITABLE SECTION ===
```

### Functions

```python
import numpy as np
from scipy.spatial.transform import Rotation


def compute_palm_pose(
    cube_translation: np.ndarray,    # [x, y, z] cube center in robot frame
    cube_rotation_matrix: np.ndarray, # 3x3 rotation matrix of cube in robot frame
    wrist_pitch_degrees: float,       # IMU pitch reading (wrist flexion angle)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the palm's position and orientation from the cube pose and IMU wrist pitch.
    
    Returns:
        (palm_position, palm_rotation_matrix)
        
        palm_position: [x, y, z] in robot frame, meters
        palm_rotation_matrix: 3x3 rotation matrix in robot frame
    
    The palm orientation incorporates the wrist pitch from the IMU.
    The cube's orientation gives us the overall hand orientation (yaw, roll from vision),
    and the IMU pitch refines the wrist flexion that the cube can't capture well.
    """
```

The computation inside `compute_palm_pose`:

**Step 1: Cube-to-wrist offset.**

The cube sits on top of the wrist. The offset from cube center to wrist is along the cube's local -Z axis (or whichever axis points from the cube toward the hand — this depends on how the cube is mounted). This is a fixed vector in cube-local coordinates, rotated into robot frame by the cube's rotation matrix.

```python
# Offset from cube center to wrist, in cube-local frame
# The cube's local frame Z axis points away from the hand (outward from the cube top)
# So the wrist is in the -Z direction
cube_to_wrist_local = np.array([0, 0, -CUBE_TO_WRIST_DISTANCE])

# Rotate into robot frame
cube_to_wrist_robot = cube_rotation_matrix @ cube_to_wrist_local

# Wrist position in robot frame
wrist_position = cube_translation + cube_to_wrist_robot
```

**IMPORTANT:** The direction of the cube-to-wrist vector in the cube's local frame depends on how the cube is physically mounted on the glove. The assumption above is that the cube's local +Z axis points away from the hand (upward, away from the wrist). If the cube is mounted differently, this vector changes. **Put a comment next to the vector definition explaining the assumption, and note that it may need to be adjusted based on the actual mounting.**

**Step 2: Wrist-to-palm offset.**

The palm is offset from the wrist along the hand's forward direction. When the wrist is flat (pitch=0), this is along the hand's forward axis. When the wrist flexes upward (pitch>0), the palm direction rotates upward.

The "hand forward" direction in the cube's local frame (again, depends on mounting) is roughly along the cube's local +X or +Y axis. The wrist pitch from the IMU rotates this direction in the pitch plane.

```python
# Hand-forward direction in cube-local frame when wrist is flat (pitch=0)
# Assumption: the hand extends in the cube's local +X direction.
# ADJUST THIS if the cube is mounted differently.
hand_forward_local = np.array([1, 0, 0])

# Apply wrist pitch rotation around the cube's local Y axis (or whichever axis
# is perpendicular to both "forward" and "up")
# Positive pitch = fingers pointing up = rotate forward vector upward
pitch_rad = np.radians(wrist_pitch_degrees)
pitch_rotation = Rotation.from_euler('Y', pitch_rad).as_matrix()

# Rotated hand-forward direction, still in cube-local frame
hand_forward_pitched = pitch_rotation @ hand_forward_local

# Scale by palm distance and rotate into robot frame
wrist_to_palm_robot = cube_rotation_matrix @ (hand_forward_pitched * WRIST_TO_PALM_DISTANCE)

# Palm position in robot frame
palm_position = wrist_position + wrist_to_palm_robot
```

**Step 3: Palm orientation.**

The palm's orientation combines the cube's rotation (gives overall hand orientation from vision) with the IMU's pitch (gives wrist flexion that the cube can't capture because the cube is on the wrist, not on the palm).

```python
# Start with the cube's rotation (overall hand orientation)
# Apply the IMU pitch as an additional rotation around the hand's pitch axis
imu_pitch_rotation = Rotation.from_euler('Y', pitch_rad).as_matrix()

# Palm rotation = cube rotation with additional wrist pitch applied in local frame
palm_rotation_matrix = cube_rotation_matrix @ imu_pitch_rotation
```

### Critical notes about coordinate frame assumptions

The offset vectors (`cube_to_wrist_local`, `hand_forward_local`) and the pitch rotation axis (`'Y'`) all depend on how the cube is physically mounted on the glove. The assumptions are:
- Cube's local +Z points away from the hand (upward from the wrist)
- Cube's local +X points along the hand (toward the fingers)
- Cube's local +Y points across the hand (toward the thumb or pinky)

**If these don't match your physical cube mounting, the offsets will be wrong.** The symptom will be the palm position being offset in the wrong direction. When you first test this:

1. Hold your hand flat in front of the camera.
2. Check that the palm position is offset *forward* from the cube (toward the fingers), not sideways or upward.
3. Flex your wrist up. Check that the palm position moves *upward*, not downward or sideways.

If either is wrong, the axis assignments in the offset vectors need to change. This is a "measure once, get right, then forget about it" calibration — but getting it wrong means the arm tracks in the wrong direction.

**Put constants for the axis directions at the top of the file so they're easy to swap:**

```python
# Axis mapping for cube mounting.
# Change these if the cube is mounted in a different orientation on the glove.
# Each is a unit vector in cube-local coordinates.
CUBE_DOWN_AXIS = np.array([0, 0, -1])    # Direction from cube toward wrist
HAND_FORWARD_AXIS = np.array([1, 0, 0])  # Direction from wrist toward fingers
PITCH_ROTATION_AXIS = 'Y'                # Axis perpendicular to forward and down
```

---

## Changes to `detect.py`

### New CLI arguments

```
--glove-port PORT      Bluetooth serial port for glove (e.g., /dev/rfcomm0, COM5)
```

No default value. If `--glove-port` is not provided, the glove reader is not started and the system works exactly as before (cube position only, no palm offset, no buttons). This preserves backward compatibility.

### Startup

If `--glove-port` is provided:

```python
from lib.glove_reader import GloveReader

glove = GloveReader(args.glove_port)
try:
    glove.start()
except serial.SerialException as e:
    print(f"ERROR: Could not open glove port {args.glove_port}: {e}")
    print("Check that the ESP32 is paired and the port is correct.")
    exit(1)
```

Print glove status on startup:
```
Glove connected on /dev/rfcomm0
Waiting for first IMU reading...
```

Once the first reading arrives (after the ESP32's gyro calibration, about 2 seconds):
```
Glove data received. IMU active.
```

### Detection loop changes

After computing the cube pose and workspace transform (existing code), add the palm offset:

```python
if cube_pose is not None and workspace_transform is not None:
    robot_pos, robot_rot = apply_transform(workspace_transform, ...)

    if glove is not None:
        glove_state = glove.get_latest()
        
        if glove_state is not None:
            # Apply palm offset using IMU pitch
            palm_pos, palm_rot = compute_palm_pose(
                robot_pos, robot_rot, glove_state.pitch
            )
            # Use palm_pos and palm_rot instead of robot_pos and robot_rot
            # for everything downstream (streaming, display, terminal output)
            output_pos = palm_pos
            output_rot = palm_rot
            buttons = glove_state.buttons
        else:
            # No glove data yet, use cube position without offset
            output_pos = robot_pos
            output_rot = robot_rot
            buttons = [0, 0, 0, 0, 0]
    else:
        # No glove connected, use cube position without offset
        output_pos = robot_pos
        output_rot = robot_rot
        buttons = None  # don't include buttons in stream if no glove
```

### Updated pose broadcast

When glove is connected, add buttons to the broadcast message:

```python
pose_dict = {
    "status": "running",
    "run_id": run_id,
    "x": float(output_pos[0]),
    "y": float(output_pos[1]),
    "z": float(output_pos[2]),
    "roll": float(roll),
    "pitch": float(pitch),
    "yaw": float(yaw),
    "num_tags": cube_pose.num_tags_used,
    "reproj_err": float(cube_pose.reprojection_error),
}

if buttons is not None:
    pose_dict["buttons"] = buttons

pose_server.broadcast(pose_dict)
```

The `buttons` field is optional in the stream. If no glove is connected, it's omitted. The arm mover should handle both cases (with and without buttons).

### Terminal output

When glove is connected, add glove info to the terminal output:

```
[PALM/ROBOT] Pos: (x=+0.150, y=-0.030, z=+0.050) m | Rot: (R:+2.0, P:-5.1, Y:+12.9) deg | tags: 2 | grip: XX--- | imu_age: 12ms
```

Note the label changes from `[CUBE/ROBOT]` to `[PALM/ROBOT]` when glove is connected, to indicate the palm offset is being applied. The `grip` field shows button states (X=pressed, dash=released). The `imu_age` shows how fresh the IMU data is.

### Shutdown

Call `glove.stop()` in the cleanup section alongside `pose_server.stop()`.

---

## Changes to `lib/visualization.py`

Add a glove status display:

```python
def draw_glove_status(frame, connected: bool, data_age_ms: float | None, buttons: list[int] | None):
    """
    Draw glove status in a corner of the video feed:
    
    If not connected (no --glove-port):
        Show nothing (glove is optional)
    
    If connected but no data yet:
        "Glove: WAITING" in yellow
    
    If connected and receiving data:
        "Glove: OK (12ms)" in green
        "Grip: [X][X][ ][ ][ ]" showing finger states
        
    If data is stale (> 200ms old):
        "Glove: STALE" in red
    """
```

Also draw the palm offset vector on the video feed (optional but helpful for debugging): a line from the cube position to the computed palm position, projected onto the image.

---

## Updated Pose Stream Protocol

The JSONL message now optionally includes `buttons`:

```json
{"status":"running","run_id":1,"x":0.15,"y":-0.03,"z":0.05,"roll":2.0,"pitch":-5.1,"yaw":12.9,"buttons":[1,0,0,1,0],"num_tags":2,"reproj_err":0.31,"timestamp":1713450000.123}
```

The `buttons` field is:
- Present when a glove is connected and data is available
- Absent when no glove is connected (backward compatible)
- An array of 5 integers: `[thumb, index, middle, ring, pinky]`, 1=pressed, 0=released

The arm mover should check for the presence of `buttons` before using it:
```python
buttons = pose.get("buttons")  # None if no glove
if buttons and any(buttons):
    # close gripper
```

---

## Implementation Order

### Phase 1: Glove reader

Create `lib/glove_reader.py`. Test it standalone before integrating:

```python
# Quick standalone test (run from hand_localizer root)
from lib.glove_reader import GloveReader
import time

reader = GloveReader("/dev/rfcomm0")  # your actual port
reader.start()

for _ in range(200):
    state = reader.get_latest()
    if state:
        print(f"pitch={state.pitch:.1f}  buttons={state.buttons}  age={reader.get_data_age():.3f}s")
    time.sleep(0.05)

reader.stop()
```

Verify: data arrives, pitch changes when you flex your wrist, buttons respond correctly, age stays under 50ms.

### Phase 2: Palm model

Create `lib/palm_model.py`. Test with synthetic data first — pass known cube poses and pitch values and verify the output makes geometric sense. Then test with real data by printing the palm position alongside the cube position and checking that the offset direction is correct.

### Phase 3: Integration

Wire everything into `detect.py`. Verify: palm position is offset from cube position in the correct direction, wrist flexion changes the palm position, buttons appear in the terminal output.

### Phase 4: Visualization

Add glove status and button display to the overlay.

---

## Verification Checklist

- [ ] `--glove-port /dev/rfcomm0` connects to the ESP32 and receives data
- [ ] `detect.py` without `--glove-port` works exactly as before (no glove, no offset)
- [ ] Glove data age stays under 50ms during normal operation
- [ ] Palm position is offset from cube position in the direction of the fingers
- [ ] Flexing wrist upward (positive pitch) moves palm position upward
- [ ] Flexing wrist downward (negative pitch) moves palm position downward
- [ ] When hand is flat, palm position is directly in front of (and below) the cube
- [ ] Rotating the whole hand (changing cube orientation) rotates the offset vectors correctly
- [ ] Buttons appear in terminal output: `grip: XX---` format
- [ ] Buttons appear in the streamed JSONL message when glove is connected
- [ ] No `buttons` field in JSONL when glove is not connected
- [ ] Terminal label shows `[PALM/ROBOT]` when glove is connected, `[CUBE/ROBOT]` when not
- [ ] Video overlay shows glove connection status
- [ ] Lost Bluetooth connection doesn't crash detect.py (prints warning, uses last known state)
- [ ] Clean shutdown with ESC/Q closes both pose server and glove reader

---

## What Is NOT In Scope

- ESP32 firmware (separate project)
- Bluetooth pairing/discovery (user pairs manually, provides port via CLI)
- IMU yaw fusion with AprilTag yaw (the IMU yaw drifts; use AprilTag yaw from the cube for absolute heading)
- Flex sensors (buttons only)
- Gripper control logic (arm mover's responsibility, not hand localizer's)
- Filtering or smoothing the IMU data beyond what the ESP32's complementary filter already does
- Auto-detection of the Bluetooth serial port

If you have questions, ask before building.