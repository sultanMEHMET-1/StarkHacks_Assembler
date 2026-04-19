# Milestone 5: Run Control & Relative Motion Mode

## Read This First

Read `CLAUDE.md` in the hand_localizer project before doing anything. Then read this entire prompt. You are modifying both the **hand_localizer** and the **arm_mover** projects. Ask questions if anything is unclear.

---

## The Problem

Right now, the hand localizer streams the hand's absolute position in robot-frame coordinates, and the arm mover sends those coordinates directly to the robot arm's IK solver. This means the robot arm tries to go exactly where the hand is. The hand and the end effector compete for the same physical space. This is dangerous and fundamentally broken for teleoperation.

## The Solution: Two Changes

### Change 1: Run control from the hand localizer

The hand localizer gains the concept of a "run." The user presses SPACE to start a run and SPACE again to stop it. While a run is active, the server streams pose messages. While no run is active, the server sends nothing (or sends a "stopped" status message so the arm mover knows to hold still). This gives the operator explicit control over when the arm follows the hand.

### Change 2: Relative position, absolute orientation on the arm mover

The arm mover does NOT send the hand's absolute position to the robot. Instead, when it receives the first pose of a new run, it records the hand's current position as the hand's positional reference and records the arm's current position as the arm's positional reference. Every subsequent pose computes a positional delta from the hand's start and applies that delta to the arm's start:

```
pos_delta = current_hand_pos - hand_start_pos
arm_target_pos = arm_start_pos + pos_delta
```

This means the hand and the arm can be in completely different parts of the workspace. The hand controls relative translational motion, not absolute position.

**Orientation is absolute, not relative.** The hand's roll, pitch, and yaw are passed directly to the arm's IK solver. If the hand is tilted 30 degrees, the end effector goes to 30 degrees. No delta, no reference capture needed for angles. This is correct because orientation describes how the gripper is pointed, not where it is, so there's no collision risk.

---

## Protocol Changes

### Updated message format

The JSONL message gains a `status` field:

```json
{"status": "running", "run_id": 3, "x": 0.150, "y": -0.030, "z": 0.050, "roll": 2.0, "pitch": -5.1, "yaw": 12.9, "num_tags": 2, "reproj_err": 0.31, "timestamp": 1713450000.123}
```

New and changed fields:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"running"` during an active run, `"stopped"` when a run ends |
| `run_id` | int | Monotonically increasing run counter, starts at 1. Increments each time SPACE starts a new run. |

**When a run is active (`status: "running"`):** One message per frame with full pose data, same as before.

**When the user stops a run (presses SPACE again):** Send exactly ONE message with `status: "stopped"` and the current `run_id`. No pose fields needed (the arm mover should ignore them if present). Then stop sending until the next run starts.

**When no run is active:** Send nothing. The arm mover uses the absence of messages (checked via `get_pose_age()`) plus the last received `status: "stopped"` to know the system is idle.

**When no cube is visible during an active run:** Do NOT send a message. The arm mover will notice the pose age growing and hold position. Do not send messages with null or zero position. Do not send a "paused" status. Just stop sending until the cube is visible again or the user stops the run.

### Why `run_id` matters

The arm mover needs to know when a NEW run starts so it can re-capture the starting positions. Without `run_id`, if the user stops and starts quickly, the arm mover can't distinguish "same run resuming" from "new run starting." The `run_id` incrementing tells the arm mover unambiguously: "reset your reference points, this is a fresh run."

---

## Hand Localizer Changes

### Changes to `detect.py`

Add run state management:

```python
# Run state
run_active = False
run_id = 0
```

**SPACE key behavior (in the main loop's key handling):**

```python
key = cv2.waitKey(1) & 0xFF

if key == ord(' '):
    if not run_active:
        # Start a new run
        run_active = True
        run_id += 1
        print(f"Run {run_id} STARTED")
    else:
        # Stop the current run
        run_active = False
        # Send one "stopped" message
        if serve_enabled:
            pose_server.broadcast({
                "status": "stopped",
                "run_id": run_id,
            })
        print(f"Run {run_id} STOPPED")
```

**Pose broadcasting logic (in the main loop, after cube pose computation):**

```python
if serve_enabled and run_active and cube_pose is not None and workspace_transform is not None:
    robot_pos, robot_rot = apply_transform(...)
    roll, pitch, yaw = rotation_matrix_to_euler(robot_rot)

    pose_server.broadcast({
        "status": "running",
        "run_id": run_id,
        "x": float(robot_pos[0]),
        "y": float(robot_pos[1]),
        "z": float(robot_pos[2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
        "num_tags": cube_pose.num_tags_used,
        "reproj_err": float(cube_pose.reprojection_error),
    })
```

Note: poses are only sent when `run_active` is True AND a cube pose is available. If the cube is not visible, nothing is sent, even during an active run.

### Changes to `lib/visualization.py`

Add a run status indicator to the video overlay:

```python
def draw_run_status(frame, run_active: bool, run_id: int):
    """
    Draw a prominent indicator on the video feed:
    
    When active:
      - Large green circle (recording dot) in the top-left area
      - Text: "RUN 3 ● ACTIVE" in green
    
    When stopped:
      - Text: "READY — press SPACE to start" in yellow
    
    This should be very visible. The operator needs to know at a glance
    whether the arm is following their hand.
    """
```

This is important for safety. The operator must always know whether the arm is active. Make the indicator large and obvious, not a subtle corner label.

### No changes to `lib/pose_server.py`

The server's `broadcast()` method already sends whatever dict you give it. The `status` and `run_id` fields are just additional keys in the dict. The server doesn't need to know about runs.

---

## Arm Mover Changes

### Changes to `pose_client.py`

The `PoseClient` class needs to be aware of run state so the arm mover can react to run transitions. Add tracking for the current run:

```python
class PoseClient:
    def __init__(self, host="127.0.0.1", port=9876):
        # ... existing fields ...
        self.current_run_id = None
        self.run_active = False
```

Update `_read_loop` to parse the `status` and `run_id` fields:

```python
# Inside _read_loop, after parsing the JSON:
pose = json.loads(latest_line)
with self.pose_lock:
    self.latest_pose = pose
    
    status = pose.get("status")
    new_run_id = pose.get("run_id")
    
    if status == "stopped":
        self.run_active = False
    elif status == "running":
        if new_run_id != self.current_run_id:
            # New run started — arm mover should reset reference points
            self.current_run_id = new_run_id
            self._new_run_started = True
        self.run_active = True
```

Add a method the arm mover polls:

```python
def check_new_run(self) -> bool:
    """
    Returns True exactly once when a new run starts.
    The arm mover calls this to know when to re-capture reference positions.
    """
    with self.pose_lock:
        if self._new_run_started:
            self._new_run_started = False
            return True
        return False

def is_run_active(self) -> bool:
    """Returns whether a run is currently active."""
    with self.pose_lock:
        return self.run_active
```

### Arm mover main loop: relative position, absolute orientation

The position is relative: the arm moves by the same delta as the hand. The orientation is absolute: the arm adopts whatever orientation the hand has, directly. This is because orientation describes how the gripper is pointed, not where it is, so there's no physical collision problem.

Here's the full updated control loop pattern. Adapt to the arm mover's actual code structure, but the logic must match:

```python
from pose_client import PoseClient
import numpy as np

client = PoseClient(host="127.0.0.1", port=9876)

try:
    client.connect(timeout=10.0)
except ConnectionError as e:
    print(e)
    exit(1)

# Reference positions — set when a new run starts
hand_start_pos = None       # np.array([x, y, z])
arm_start_pos = None        # np.array([x, y, z])

STALE_THRESHOLD = 0.5  # seconds

try:
    while True:
        # Check if a new run just started
        if client.check_new_run():
            pose = client.get_latest_pose()
            if pose:
                hand_start_pos = np.array([pose["x"], pose["y"], pose["z"]])
                
                # Capture the arm's CURRENT position as the arm's starting reference
                # This is wherever the arm happens to be right now
                current_arm = arm.get_current_position()  # adapt to your arm API
                arm_start_pos = np.array([current_arm.x, current_arm.y, current_arm.z])
                
                print(f"New run — hand ref: {hand_start_pos}, arm ref: {arm_start_pos}")

        # If no run active, hold position
        if not client.is_run_active():
            time.sleep(0.01)
            continue

        # If references aren't set yet, wait
        if hand_start_pos is None or arm_start_pos is None:
            time.sleep(0.01)
            continue

        pose = client.get_latest_pose()
        if pose is None:
            time.sleep(0.01)
            continue

        # Check freshness
        age = client.get_pose_age()
        if age is not None and age > STALE_THRESHOLD:
            # Stale pose — hold position, don't move to old target
            time.sleep(0.01)
            continue

        # POSITION: relative delta from hand start, applied to arm start
        current_hand_pos = np.array([pose["x"], pose["y"], pose["z"]])
        pos_delta = current_hand_pos - hand_start_pos
        arm_target_pos = arm_start_pos + pos_delta

        # ORIENTATION: absolute, pass through directly from hand to arm
        # The hand's roll/pitch/yaw in robot frame IS the target orientation.
        # No delta, no offset, no reference capture needed for angles.

        # Send to IK solver
        arm.move_to(
            x=arm_target_pos[0],
            y=arm_target_pos[1],
            z=arm_target_pos[2],
            roll=pose["roll"],
            pitch=pose["pitch"],
            yaw=pose["yaw"],
        )

        time.sleep(0.01)  # 100Hz max

except KeyboardInterrupt:
    pass
finally:
    client.disconnect()
    arm.close()
```

### Key details about the motion logic

**Position is relative (delta-based).** Subtract the hand start from the current hand position, add to the arm start. This decouples the two workspaces so they don't collide.

**Orientation is absolute (direct pass-through).** The hand's roll, pitch, and yaw in robot frame are sent directly to the arm's IK solver with no transformation. If the hand is tilted 30 degrees, the end effector goes to 30 degrees. This is correct because orientation describes how the gripper is pointed, not where it is in space. There's no collision risk from matching orientation, and relative orientation would feel unintuitive to the operator (they'd have to mentally track "how much have I rotated since the run started" rather than just pointing).

**The arm's starting position must come from the arm itself** (via `arm.get_current_position()` or whatever the arm mover's API provides), NOT from the hand localizer. This is what decouples the two coordinate spaces. The hand can be 50cm in front of the camera, and the arm can be in a completely different part of its workspace. When the run starts, the arm records its own position reference, and from then on only the position deltas are shared. No starting reference is needed for orientation because it's absolute.

**When a run stops**, the arm should hold its last position. It should NOT return to its starting position, go limp, or do anything dramatic. Just stop updating. The operator can manually jog the arm before starting the next run if they want it somewhere else.

---

## Startup and Workflow

The full workflow for operating the system:

```
Terminal 1:
$ cd hand_localizer
$ python detect.py --serve --port 9876
> Camera calibration loaded
> Workspace calibration loaded
> Pose server listening on 0.0.0.0:9876
> READY — press SPACE to start

Terminal 2:
$ cd arm_mover
$ python main.py --pose-host 127.0.0.1 --pose-port 9876
> Connected to pose server at 127.0.0.1:9876
> Waiting for run to start...
```

Operator workflow:
1. Position the arm where you want it to start (jog manually or via arm mover controls).
2. Hold your gloved hand in a comfortable starting position in view of the camera.
3. Press SPACE in the hand localizer window.
4. Move your hand. The arm mirrors your movements relative to both starting positions.
5. Press SPACE again to stop. The arm holds its last position.
6. Reposition hand and/or arm if desired.
7. Press SPACE to start the next run.

---

## Safety Considerations

**The operator must be able to stop the arm instantly.** SPACE stops the run, but there should also be an emergency option. Add an ESC key behavior: when pressed during an active run, it stops the run AND sends a "stopped" message AND exits detect.py entirely. This is the "oh no" button. Document this clearly on the video overlay.

**When the cube goes out of view during a run**, the server sends nothing, the arm mover detects stale data (via `get_pose_age()`), and the arm holds position. This is correct. The arm should NOT try to extrapolate or predict where the hand went. It just freezes until the cube is visible again.

**When the run stops**, the arm holds its last position. It does not return home, it does not go limp. The operator has full control over when and where the arm moves.

---

## What to Change: File Summary

### hand_localizer

| File | Action | What changes |
|------|--------|-------------|
| `detect.py` | **MODIFY** | Add `run_active` / `run_id` state, SPACE key toggles run, broadcast only during active runs, send "stopped" message on stop |
| `lib/visualization.py` | **MODIFY** | Add `draw_run_status()` — prominent green/yellow indicator showing run state |

No other hand_localizer files change. The pose server, detection pipeline, cube fusion, and workspace calibration are all untouched.

### arm_mover

| File | Action | What changes |
|------|--------|-------------|
| `pose_client.py` | **MODIFY** | Add `run_active`, `current_run_id`, `check_new_run()`, `is_run_active()` |
| Main control script | **MODIFY** | Replace absolute positioning with relative delta logic, handle run start/stop |

---

## Verification Checklist

- [ ] SPACE starts a run. Video overlay shows "RUN 1 ● ACTIVE" in green
- [ ] SPACE again stops the run. Overlay shows "READY — press SPACE to start" in yellow
- [ ] No pose messages are sent when run is not active (verify with `nc localhost 9876`)
- [ ] Pose messages during active run include `"status": "running"` and `"run_id": N`
- [ ] One `"status": "stopped"` message is sent when run stops
- [ ] `run_id` increments with each new run (1, 2, 3, ...)
- [ ] Arm mover resets POSITION reference when a new `run_id` appears (no angle reference needed)
- [ ] Arm does NOT try to go to the hand's absolute XYZ position
- [ ] Arm mirrors relative hand POSITION from respective starting positions
- [ ] Moving hand 10cm right moves arm 10cm right (1:1 positional mapping)
- [ ] Hand and arm can be in different parts of the workspace (positionally decoupled)
- [ ] Arm DOES match the hand's absolute orientation (tilt hand 30°, arm goes to 30°, not +30° from start)
- [ ] Arm holds position when run stops (does not return home)
- [ ] Arm holds position when cube goes out of view during a run (stale pose detection)
- [ ] ESC during active run: stops run, sends stopped message, exits detect.py
- [ ] Starting a new run after stopping: arm uses new POSITION reference, orientation stays absolute

---

## What Is NOT In Scope

- Scaling factor for hand-to-arm motion (1:1 only for now)
- Gripper control (flex sensors, later milestone)
- Recording demonstration data to files
- Replaying recorded demonstrations
- IMU fusion
- Any UI beyond the OpenCV window and terminal

If you have questions about any of this, ask before you start building.