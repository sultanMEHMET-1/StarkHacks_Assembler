# Milestone 4: Pose Streaming — Connecting Hand Localizer to Arm Mover

## Read This First

You are implementing both sides of a TCP connection between two independent Python projects:

1. **hand_localizer** — Already exists and works. Detects AprilTags on a glove cube, fuses them into a single 6DOF pose via solvePnP, transforms to robot-frame coordinates via workspace calibration. Currently prints the pose to terminal and draws overlays on a video feed.

2. **arm_mover** — Already exists and works. Has a function/class that accepts a target position and orientation and commands a LeRobot arm to move there via inverse kinematics. Currently operates standalone.

Your job is to connect them. The hand localizer becomes a **pose server** that broadcasts the cube's position over TCP. The arm mover becomes a **pose client** that receives those positions and passes them to the robot arm's IK solver. Both run as separate processes on the same machine.

---

## Architecture Overview

```
┌─────────────────────────┐         TCP (localhost:9876)         ┌─────────────────────────┐
│    hand_localizer       │  ──────────────────────────────────> │     arm_mover           │
│                         │         JSONL pose messages          │                         │
│  camera → detect tags   │                                     │  receive pose           │
│  → fuse cube pose       │                                     │  → IK solve             │
│  → transform to robot   │                                     │  → command motors       │
│  → broadcast via TCP    │                                     │                         │
│                         │                                     │                         │
│  (server, 1 or N        │                                     │  (client, connects      │
│   clients can connect)  │                                     │   on startup)           │
└─────────────────────────┘                                     └─────────────────────────┘
```

The hand localizer is the **server**. It listens on a port and accepts client connections. Every time it computes a new cube pose in robot-frame coordinates, it sends that pose to all connected clients. If no clients are connected, it drops the data and keeps running. If a client disconnects, the server removes it and continues.

The arm mover is the **client**. It connects to the server on startup, reads pose messages, and acts on them. It always uses the **most recent** complete message, discarding any older messages that accumulated in the buffer.

---

## Protocol: Newline-Delimited JSON (JSONL)

Each message is a single line of JSON followed by a newline character (`\n`). No headers, no handshake, no framing beyond the newline delimiter. One pose per line.

### Message format

```json
{"x": 0.150, "y": -0.030, "z": 0.050, "roll": 2.0, "pitch": -5.1, "yaw": 12.9, "num_tags": 2, "reproj_err": 0.31, "timestamp": 1713450000.123}
```

Field definitions:

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `x` | float | meters | Position X in robot frame |
| `y` | float | meters | Position Y in robot frame |
| `z` | float | meters | Position Z in robot frame |
| `roll` | float | degrees | Roll (ZYX Euler convention) |
| `pitch` | float | degrees | Pitch (ZYX Euler convention) |
| `yaw` | float | degrees | Yaw (ZYX Euler convention) |
| `num_tags` | int | — | Number of AprilTags used in the fused pose |
| `reproj_err` | float | pixels | Reprojection error from solvePnP |
| `timestamp` | float | seconds | Unix timestamp (`time.time()`) when pose was computed |

All position values are in the robot's base frame (after workspace calibration transform). If workspace calibration is not loaded, the server must NOT stream poses. It should print a warning and wait, or refuse to start the server.

### Why JSONL

- Human-readable: you can debug with `nc localhost 9876` or `telnet localhost 9876` and see the data live.
- Trivially parsable in any language: `json.loads(line)` in Python, `JSON.parse(line)` in JS, etc.
- Self-delimiting: the newline is the message boundary. No need to track message length.
- No external dependencies: Python's `json` module is in the standard library.

### What NOT to use

- Do not use protobuf, msgpack, or any binary serialization. Overkill, adds dependencies.
- Do not use pickle. Security risk, Python-only.
- Do not use HTTP, REST, Flask, or FastAPI. Way too heavy for localhost pose streaming.
- Do not use ZMQ. Adds a dependency for no benefit on localhost.
- Do not use WebSockets. Unnecessary complexity.
- Do not use multiprocessing.Queue or shared memory. The two projects are separate processes, potentially separate virtualenvs.

---

## Hand Localizer Side: Pose Server

### New file: `lib/pose_server.py`

This module runs a TCP server in a **background thread**. The main detection loop stays synchronous and untouched. After computing a cube pose, the detection loop calls a method on the server to broadcast it. The server handles the networking in its own thread.

```python
import socket
import threading
import json
import time


class PoseServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        """
        Initialize the pose server. Does not start listening yet.
        
        host: "0.0.0.0" to accept connections from any interface,
              "127.0.0.1" to restrict to localhost only.
              Default to "0.0.0.0" for flexibility, but in practice
              this will almost always be used on localhost.
        port: TCP port to listen on.
        """
        self.host = host
        self.port = port
        self.clients = []          # list of connected client sockets
        self.clients_lock = threading.Lock()
        self.server_socket = None
        self.running = False

    def start(self):
        """
        Start listening for connections in a background thread.
        This method returns immediately.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # so accept() doesn't block forever on shutdown
        self.running = True

        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        accept_thread.start()

        print(f"Pose server listening on {self.host}:{self.port}")

    def _accept_loop(self):
        """
        Background thread: accept incoming client connections.
        """
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.clients_lock:
                    self.clients.append(client_sock)
                print(f"Client connected: {addr} (total: {len(self.clients)})")
            except socket.timeout:
                continue
            except OSError:
                break  # socket was closed

    def broadcast(self, pose_dict: dict):
        """
        Send a pose to all connected clients. Called from the main detection loop.
        
        pose_dict should contain: x, y, z, roll, pitch, yaw, num_tags, reproj_err.
        This method adds the timestamp automatically.
        
        If a client has disconnected (broken pipe), remove it silently.
        If no clients are connected, do nothing.
        """
        pose_dict["timestamp"] = time.time()
        message = json.dumps(pose_dict) + "\n"
        data = message.encode("utf-8")

        disconnected = []
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.sendall(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    disconnected.append(client)

            for client in disconnected:
                self.clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass
                print(f"Client disconnected (total: {len(self.clients)})")

    def stop(self):
        """
        Shut down the server. Close all client connections.
        Called during cleanup when detect.py exits.
        """
        self.running = False
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except OSError:
                    pass
            self.clients.clear()
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
        print("Pose server stopped.")
```

### Key implementation details for pose_server.py

**TCP_NODELAY is important.** Without it, the OS may buffer small messages (Nagle's algorithm) and introduce 10-40ms of latency. Since each pose message is small (~150 bytes), Nagle will try to batch them. `TCP_NODELAY` forces each `sendall()` to go out immediately. Set this on each client socket when it connects.

**The server thread is a daemon thread.** This means it dies automatically when the main process exits. No need for explicit cleanup on Ctrl+C, though `stop()` should still be called for clean shutdown.

**`broadcast()` is called from the main thread (detection loop).** It must be fast. The lock protects the client list but the actual `sendall()` happens inside the lock. For a small number of clients (1-3), this is fine. Do not add a message queue or async sending layer. Keep it simple.

**No reconnection logic on the server side.** If a client disconnects, it's removed. If a new client connects, it starts receiving from the next broadcast. The server doesn't care about client identity or state.

### Changes to detect.py

Add a `--serve` flag and a `--port` option:

```
python detect.py [existing args] [--serve] [--port 9876]
```

- `--serve` enables the pose server. Without this flag, detect.py behaves exactly as before (no networking).
- `--port` sets the TCP port (default 9876).
- `--serve` requires workspace calibration to be loaded. If no workspace transform is available, print an error and exit: "Cannot serve poses without workspace calibration. Run workspace_calibrate.py first, or use --skip-workspace to run without serving."

In the detection loop, after computing the cube pose and applying the workspace transform:

```python
if serve_enabled and cube_pose is not None and workspace_transform is not None:
    robot_pos, robot_rot = apply_transform(workspace_transform, cube_pose.translation, cube_pose.rotation_matrix)
    roll, pitch, yaw = rotation_matrix_to_euler(robot_rot)
    
    pose_server.broadcast({
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

Make sure to convert numpy floats to Python floats before passing to `json.dumps`. Numpy float64 is not JSON-serializable by default.

On startup, print the server status:

```
Pose server listening on 0.0.0.0:9876
Waiting for clients... (arm mover should connect to localhost:9876)
```

On shutdown (ESC/Q or Ctrl+C), call `pose_server.stop()`.

### Changes to visualization.py

Add a connection indicator to the video overlay:

```python
def draw_server_status(frame, num_clients: int, serving: bool):
    """
    Draw in the top-right area:
    - If not serving: nothing (or "Server: OFF" in gray)
    - If serving, 0 clients: "Server: WAITING" in yellow
    - If serving, 1+ clients: "Server: 1 client(s)" in green
    """
```

---

## Arm Mover Side: Pose Client

### New file in arm_mover project: `pose_client.py`

This is a self-contained module that the arm mover imports. It connects to the hand localizer's pose server and provides the latest pose on demand.

```python
import socket
import json
import threading
import time


class PoseClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        """
        Initialize the pose client. Does not connect yet.
        """
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.latest_pose = None
        self.pose_lock = threading.Lock()
        self.connected = False

    def connect(self, timeout: float = 5.0, retry_interval: float = 1.0):
        """
        Connect to the pose server. Retries until connected or timeout.
        Then starts a background thread to continuously read poses.
        
        Raises ConnectionError if unable to connect within timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connected = True
                self.running = True

                reader_thread = threading.Thread(target=self._read_loop, daemon=True)
                reader_thread.start()

                print(f"Connected to pose server at {self.host}:{self.port}")
                return
            except ConnectionRefusedError:
                time.sleep(retry_interval)

        raise ConnectionError(
            f"Could not connect to pose server at {self.host}:{self.port} "
            f"within {timeout}s. Is detect.py running with --serve?"
        )

    def _read_loop(self):
        """
        Background thread: continuously read from the socket,
        parse JSONL messages, and store the latest pose.
        
        CRITICAL: This must always use the MOST RECENT complete message.
        If multiple messages have accumulated in the buffer (because the
        arm mover is slower than the localizer), discard all but the last one.
        """
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096)
                if not data:
                    # Server disconnected
                    self.connected = False
                    print("Pose server disconnected.")
                    break

                buffer += data.decode("utf-8")

                # Split by newlines and keep only the LAST complete message
                lines = buffer.split("\n")

                # The last element is either empty (if buffer ended with \n)
                # or an incomplete message (keep it in the buffer)
                buffer = lines[-1]

                # Parse the last COMPLETE line (second-to-last element if buffer was empty,
                # or further back if there were multiple messages)
                complete_lines = lines[:-1]
                if complete_lines:
                    # Take only the most recent complete message
                    latest_line = complete_lines[-1].strip()
                    if latest_line:
                        try:
                            pose = json.loads(latest_line)
                            with self.pose_lock:
                                self.latest_pose = pose
                        except json.JSONDecodeError:
                            pass  # skip malformed messages

            except (ConnectionResetError, OSError):
                self.connected = False
                print("Connection to pose server lost.")
                break

    def get_latest_pose(self) -> dict | None:
        """
        Return the most recently received pose, or None if no pose
        has been received yet.
        
        Returns a dict with keys: x, y, z, roll, pitch, yaw,
        num_tags, reproj_err, timestamp
        
        The caller can check the timestamp to determine freshness.
        """
        with self.pose_lock:
            return self.latest_pose.copy() if self.latest_pose else None

    def get_pose_age(self) -> float | None:
        """
        Return how many seconds ago the latest pose was generated,
        or None if no pose received yet.
        """
        with self.pose_lock:
            if self.latest_pose and "timestamp" in self.latest_pose:
                return time.time() - self.latest_pose["timestamp"]
            return None

    def disconnect(self):
        """Clean shutdown."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
        self.connected = False
        print("Disconnected from pose server.")
```

### How the arm mover uses the client

The arm mover's main loop should look something like this (adapt to the actual arm mover code structure):

```python
from pose_client import PoseClient

client = PoseClient(host="127.0.0.1", port=9876)

try:
    client.connect(timeout=10.0)
except ConnectionError as e:
    print(e)
    exit(1)

try:
    while True:
        pose = client.get_latest_pose()
        
        if pose is None:
            # No pose received yet, wait
            time.sleep(0.01)
            continue
        
        # Check freshness — skip if pose is too old
        age = client.get_pose_age()
        if age is not None and age > 0.5:
            # Pose is more than 500ms old, something is wrong
            # (camera occluded, hand out of frame, etc.)
            # Stop the arm or hold position rather than acting on stale data
            print(f"WARNING: pose is {age:.1f}s old, skipping")
            time.sleep(0.01)
            continue
        
        # Send to IK solver / arm controller
        arm.move_to(
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            roll=pose["roll"],
            pitch=pose["pitch"],
            yaw=pose["yaw"],
        )
        
        time.sleep(0.01)  # 100Hz max, adjust as needed

except KeyboardInterrupt:
    pass
finally:
    client.disconnect()
    arm.close()  # or however the arm mover cleans up
```

### Critical detail: the "drain and use latest" pattern

The arm mover's IK loop might run slower than the localizer's camera loop. If the localizer sends 60 poses per second and the arm mover processes 50 per second, 10 poses per second accumulate in the TCP buffer. After a minute, the arm mover would be acting on poses from a second ago.

The `_read_loop` handles this by always taking the LAST complete line from the buffer, discarding everything older. This means the arm is always acting on the freshest data. The `get_pose_age()` method lets the arm mover double-check: if the latest pose is more than ~200-500ms old, something is wrong (camera blocked, hand out of frame, server lagging) and the arm should hold position rather than move to a stale target.

---

## Testing the Connection

Before integrating with the real arm, test the connection with two terminals:

**Terminal 1 — Start the hand localizer with serving:**
```bash
cd hand_localizer
python detect.py --serve --port 9876
```

**Terminal 2 — Test with netcat (simplest possible client):**
```bash
nc localhost 9876
```

You should see JSONL pose messages streaming in terminal 2 every frame. If you see nothing, the server isn't sending (check that workspace calibration is loaded and the cube is in view). If you see garbled data, there's an encoding issue.

**Terminal 2 — Test with the pose client:**
```bash
cd arm_mover
python -c "
from pose_client import PoseClient
import time
client = PoseClient()
client.connect(timeout=5)
for _ in range(100):
    pose = client.get_latest_pose()
    if pose:
        print(f'x={pose[\"x\"]:.3f} y={pose[\"y\"]:.3f} z={pose[\"z\"]:.3f} age={client.get_pose_age():.3f}s')
    time.sleep(0.1)
client.disconnect()
"
```

You should see position values updating smoothly. The age should consistently be under ~50ms. If age is growing over time, the drain-latest logic isn't working.

---

## Constants and Configuration

### In hand_localizer

At the top of `lib/pose_server.py`:

```python
DEFAULT_PORT = 9876
DEFAULT_HOST = "0.0.0.0"
```

Overridable via `--port` on `detect.py`. No config file needed.

### In arm_mover

At the top of `pose_client.py`:

```python
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
STALE_POSE_THRESHOLD = 0.5  # seconds — poses older than this are considered stale
```

The host and port should be overridable via CLI args on whatever the arm mover's entry point is.

---

## Error Handling

### Server side (hand_localizer)

| Scenario | Behavior |
|----------|----------|
| No clients connected | Drop pose data silently, keep running |
| Client disconnects mid-stream | Remove from client list, log "Client disconnected", keep running |
| `sendall()` raises BrokenPipeError | Same as disconnect: remove client, continue |
| `--serve` without workspace calibration | Exit with clear error message before starting |
| Port already in use | Exit with: "Port 9876 already in use. Is another instance running?" |
| Ctrl+C / ESC / Q | Call `pose_server.stop()`, close all connections, exit cleanly |

### Client side (arm_mover)

| Scenario | Behavior |
|----------|----------|
| Server not running at connect time | Retry for `timeout` seconds, then raise `ConnectionError` with clear message |
| Server disconnects during operation | Set `connected = False`, log message. Arm mover should check `client.connected` and either wait for reconnect or stop |
| Malformed JSON received | Skip the message silently (`json.JSONDecodeError` caught) |
| Pose is stale (age > threshold) | `get_pose_age()` returns the age; arm mover decides what to do (hold position, stop, etc.) |
| No pose received yet | `get_latest_pose()` returns None; arm mover should wait |

---

## What NOT to Build

- No automatic reconnection on the client side. If the server dies, the client reports it and the arm mover can decide to exit or wait. Manual restart is fine for a hackathon.
- No authentication, encryption, or access control. This is localhost.
- No message acknowledgment or request-response pattern. This is fire-and-forget streaming.
- No message queuing or buffering beyond the TCP buffer. The latest message wins.
- No discovery protocol. The client is told the host and port explicitly.
- No binary serialization. JSON is fine for ~60 messages/second of ~150 bytes each.

---

## File Summary

### hand_localizer (you are adding to an existing project)

| File | Action | Description |
|------|--------|-------------|
| `lib/pose_server.py` | **CREATE** | TCP server, background thread, broadcast method |
| `detect.py` | **MODIFY** | Add `--serve` and `--port` flags, call `pose_server.broadcast()` in loop, add startup/shutdown logic |
| `lib/visualization.py` | **MODIFY** | Add `draw_server_status()` for connection indicator overlay |

Do NOT modify any other files in hand_localizer. The detection pipeline, cube fusion, workspace calibration, and all other existing code stays untouched.

### arm_mover (you are adding to an existing project)

| File | Action | Description |
|------|--------|-------------|
| `pose_client.py` | **CREATE** | TCP client, background reader thread, get_latest_pose() |

The arm mover's main script needs to be modified to import `PoseClient`, connect on startup, and call `get_latest_pose()` in its control loop instead of however it currently gets target poses. The exact modification depends on the arm mover's current code structure — inspect it before changing it. The integration should be minimal: import the client, connect, read pose, pass to existing IK function.

---

## Dependencies

**None added.** Both the server and client use only Python standard library modules: `socket`, `threading`, `json`, `time`. No new entries in `requirements.txt` on either side.

---

## Verification Checklist

- [ ] `detect.py --serve` starts and prints "Pose server listening on 0.0.0.0:9876"
- [ ] `detect.py` without `--serve` works exactly as before (no networking)
- [ ] `--serve` without workspace calibration prints error and exits
- [ ] `nc localhost 9876` receives JSONL pose messages when cube is in view
- [ ] Messages are valid JSON with all required fields (x, y, z, roll, pitch, yaw, num_tags, reproj_err, timestamp)
- [ ] Position values are in robot frame (not camera frame)
- [ ] No messages sent when cube is not visible (no null poses)
- [ ] Multiple clients can connect simultaneously (test with two `nc` sessions)
- [ ] Client disconnect doesn't crash the server
- [ ] Server shutdown closes all connections cleanly
- [ ] `PoseClient.connect()` succeeds when server is running
- [ ] `PoseClient.connect()` raises `ConnectionError` with clear message when server is not running
- [ ] `get_latest_pose()` returns None before first pose received
- [ ] `get_latest_pose()` returns fresh poses during normal operation
- [ ] `get_pose_age()` stays under ~50ms during normal operation
- [ ] If client reads slowly (add artificial delay), it still gets the latest pose, not queued old ones
- [ ] Video overlay shows server status (waiting / N clients connected)
- [ ] Port in use produces clear error, not a stack trace

If you have questions about any of this, ask before you start building. 