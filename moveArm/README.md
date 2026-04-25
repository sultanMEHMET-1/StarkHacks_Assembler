# arm-mover

Small utility that uses [LeRobot](https://github.com/huggingface/lerobot) to send **preset Cartesian end-effector poses** to an SO-100 or SO-101 **follower** arm on a fixed schedule. End-effector goals are converted to joint commands through LeRobot's built-in IK processors.

## Requirements

- Python 3.12 or newer
- A USB-connected SO follower arm and the Feetech + kinematics stack (`lerobot[feetech,kinematics]`)

## Install

Create a virtual environment if needed, then activate it. The folder is often `.venv` or `venv`:

```bash
cd /path/to/moveArm
source .venv/bin/activate
# or: source venv/bin/activate
```

With `uv` (recommended):

```bash
uv pip install -e ".[dev]"
```

Or with `pip` / `pip3`:

```bash
pip3 install -e ".[dev]"
```

If you use `uv` to manage dependencies on a fresh project, you can also run `uv add "lerobot[feetech,kinematics]"` before installing this package in editable mode.

The project depends on `lerobot[feetech,kinematics]` (Feetech SDK + IK backend via placo).

Install note:

```bash
uv pip install "lerobot[feetech,kinematics]"
```

## Get the URDF

This repository **includes a vendored copy** of [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) under `SO-ARM100/` (URDF, simulation assets, and CAD). Use the SO-101 calibration URDF for IK:

```bash
cd /path/to/moveArm
export LEROBOT_SO101_URDF="$(pwd)/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
```

To refresh from upstream instead of using the copy in this repo:

```bash
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git
export LEROBOT_SO101_URDF=$(pwd)/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

Pass the URDF path to the CLI with `--urdf-path "$LEROBOT_SO101_URDF"` (required for IK unless you run `--dry-run --skip-ik`).

## Coordinate conventions

- End-effector position keys are `ee.x`, `ee.y`, `ee.z` in meters.
- Frame convention (from the SO-101 URDF): `+x` forward, `+y` left, `+z` up.
- Orientation uses rotation vectors (`ee.wx`, `ee.wy`, `ee.wz`), i.e. axis-angle in radians.
- Human-readable Euler inputs are converted internally using LeRobot `Rotation`.

Worked example (intrinsic ``xyz`` roll/pitch/yaw in degrees; LeRobot’s ``Rotation`` has no ``from_euler``, so this project uses ``Rz @ Ry @ Rx`` then ``from_matrix``):

```python
from arm_mover.poses import euler_xyz_to_rotvec

rotvec = euler_xyz_to_rotvec(0.0, 90.0, 0.0, degrees=True)
```

## Find the serial port

Use the helper shipped with LeRobot:

```bash
lerobot-find-port
```

On Linux the port is often `/dev/ttyACM0` or `/dev/ttyUSB0`; on macOS `/dev/tty.usbmodem...`; on Windows `COM5` (example).

## Calibration (one-time)

The first time you connect with a **new** `--robot-id`, `robot.connect()` runs **interactive** calibration: you will be asked to move the arm to mid-range, then through its ranges of motion. Complete this once per arm/id. Calibration is stored under:

`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot-id>.json`

Do **not** try to automate this step.

## Usage

**Always** clear the workspace, keep an emergency stop or power switch within reach, and start with a **dry run** (no hardware motion). Keep `--max-relative-target` **small** for first real runs (for example `5.0` degrees) so each command cannot move a joint farther than that per call.

Dry run (runs IK and prints computed joint actions, no bus connection):

```bash
arm-mover \
  --port /dev/ttyACM0 \
  --id my_so100 \
  --urdf-path "$LEROBOT_SO101_URDF" \
  --pose-units euler-deg \
  --dry-run
```

### Sequences

- `home_left_right`, `home_up`, `home_only`: small motions near a nominal “home” pose.
- `range_demo`: **10 waypoints** that sweep reach (in/out), lateral y, height (high/low), wrist roll, gripper open/close, and moderate yaw—intended to exercise a **meaningful slice** of the usable workspace. Re-measure poses in `poses.py` for your table and arm before trusting it on hardware.

### Real hardware run (range-of-motion demo)

What you should see: after `connect()` (and calibration if needed), the arm visits each waypoint in order. Between two waypoints, **arm-mover** emits several end-effector commands along a straight line in Cartesian space so each position step is at most `--max-ee-step` (LeRobot’s `EEBoundsAndSafety` rejects larger single-tick jumps). For every substep, the tool reads the current joint observation, runs IK, then sends joint targets. Joint motion is also limited by `--max-relative-target` (degrees per command). The gripper should open and close near the end of the sequence.

```bash
cd /path/to/moveArm
source .venv/bin/activate
export LEROBOT_SO101_URDF="$(pwd)/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

arm-mover \
  --port /dev/ttyACM0 \
  --id my_so101 \
  --urdf-path "$LEROBOT_SO101_URDF" \
  --sequence range_demo \
  --cycles 2 \
  --bounds-min=-0.25,-0.25,0.05 \
  --bounds-max 0.25,0.25,0.35 \
  --max-ee-step 0.01 \
  --max-relative-target 5.0
```

Use `lerobot-find-port` to pick `--port` if you are unsure. Increase `--cycles` to repeat the full sweep. If motion is too slow, you can raise `--max-ee-step` slightly after you are confident the workspace is safe (smaller is safer).

Shorter preset (small motions only):

```bash
arm-mover \
  --port /dev/ttyACM0 \
  --id my_so100 \
  --urdf-path "$LEROBOT_SO101_URDF" \
  --sequence home_left_right \
  --bounds-min=-0.2,-0.2,0.05 \
  --bounds-max 0.2,0.2,0.3 \
  --max-ee-step 0.01 \
  --max-relative-target 5.0
```

See all options:

```bash
arm-mover --help
```

### EE pose while moving the arm by hand

Use **`--monitor-ee`** to connect, **disable motor torque**, and log **EE read (FK)** from encoders at **`--monitor-interval`** seconds (default `0.2`). Press Ctrl+C to stop, or pass **`--monitor-samples N`** to exit after `N` lines. Requires **`--urdf-path`** (same URDF as IK). Not compatible with **`--dry-run`** or **`--skip-ik`**.

```bash
arm-mover --port /dev/ttyACM0 --id my_so100 \
  --urdf-path "$LEROBOT_SO101_URDF" \
  --monitor-ee
```

## Troubleshooting

If you see `ConnectionError` / “There is no status packet” from Feetech right **after** calibration (often on motor **id 2**, **shoulder_lift**), that is usually **USB/serial timing**, not bad poses. `arm-mover` **retries `connect()`** a few times with a short delay. If it still fails: check the cable, supply power, try another USB port (avoid flaky hubs), unplug/replug, and run again — calibration is already saved, so the next run should skip the full calibration flow if the file matches the hardware.

## Safety

- Start in `--dry-run` and confirm the generated joint commands are sensible.
- Begin with tight Cartesian limits via `--bounds-min` and `--bounds-max`.
- Keep `--max-ee-step` small (for example `0.01`) to cap per-tick Cartesian motion.
- Keep `--max-relative-target` small (for example `5.0`) to cap per-tick joint motion.
- Keep the robot power switch reachable at all times.

## Tests

Hardware-free unit tests and optional IK pipeline checks (which skip if `placo` or `LEROBOT_SO101_URDF` is unavailable). With the venv activated:

```bash
pytest
# or: python -m pytest
```

## License

See your project policy; default is your choice when you publish the package.
