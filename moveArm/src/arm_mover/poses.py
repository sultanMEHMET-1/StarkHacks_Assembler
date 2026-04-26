"""Cartesian end-effector poses for SO follower arms.

IMPORTANT: The built-in pose list is conservative guidance only. Re-measure x/y/z and
orientation values for your specific arm and environment, and validate in dry-run mode
before sending commands to hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from lerobot.utils.rotation import Rotation

POSE_UNITS_ROTATION_VECTOR: Final[str] = "rotvec"
POSE_UNITS_EULER_DEGREES: Final[str] = "euler-deg"
POSE_UNITS_EULER_RADIANS: Final[str] = "euler-rad"

POSE_UNITS_CHOICES: Final[tuple[str, ...]] = (
    POSE_UNITS_ROTATION_VECTOR,
    POSE_UNITS_EULER_DEGREES,
    POSE_UNITS_EULER_RADIANS,
)

_EULER_SEQUENCE_XYZ: Final[str] = "xyz"


def euler_xyz_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    *,
    degrees: bool,
) -> np.ndarray:
    """Intrinsic Tait–Bryan xyz: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    LeRobot's :class:`~lerobot.utils.rotation.Rotation` does not provide ``from_euler``;
    this matches the usual scipy-style ``"xyz"`` composition used for roll/pitch/yaw labels.
    """
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cosine_roll, sine_roll = np.cos(roll), np.sin(roll)
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine_roll, -sine_roll],
            [0.0, sine_roll, cosine_roll],
        ]
    )
    rotation_y = np.array(
        [
            [cosine_pitch, 0.0, sine_pitch],
            [0.0, 1.0, 0.0],
            [-sine_pitch, 0.0, cosine_pitch],
        ]
    )
    rotation_z = np.array(
        [
            [cosine_yaw, -sine_yaw, 0.0],
            [sine_yaw, cosine_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotation_z @ rotation_y @ rotation_x


def euler_xyz_to_rotvec(
    roll: float,
    pitch: float,
    yaw: float,
    *,
    degrees: bool = True,
) -> tuple[float, float, float]:
    """Convert roll/pitch/yaw (intrinsic xyz) to a rotation vector via LeRobot."""
    matrix = euler_xyz_to_rotation_matrix(roll, pitch, yaw, degrees=degrees)
    vector = Rotation.from_matrix(matrix).as_rotvec()
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def rotvec_to_euler_xyz(
    rotvec: tuple[float, float, float],
    *,
    degrees: bool = True,
) -> tuple[float, float, float]:
    """Inverse of :func:`euler_xyz_to_rotvec` for the same intrinsic xyz convention."""
    matrix = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    sine_pitch = -matrix[2, 0]
    cosine_pitch = np.hypot(matrix[2, 1], matrix[2, 2])
    pitch_angle = float(np.arctan2(sine_pitch, cosine_pitch))
    roll_angle = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
    yaw_angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    if degrees:
        return (
            float(np.rad2deg(roll_angle)),
            float(np.rad2deg(pitch_angle)),
            float(np.rad2deg(yaw_angle)),
        )
    return (roll_angle, pitch_angle, yaw_angle)


@dataclass(frozen=True)
class EEPose:
    name: str
    x: float
    y: float
    z: float
    rotvec: tuple[float, float, float]
    gripper_pos: float  # 0.0 .. 100.0
    hold_s: float = 1.0

    @classmethod
    def from_euler(
        cls,
        name: str,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        gripper_pos: float,
        hold_s: float = 1.0,
        sequence: str = "xyz",
        degrees: bool = True,
    ) -> "EEPose":
        """Create an :class:`EEPose` from Euler angles.

        ``sequence`` must be ``"xyz"`` (intrinsic roll→pitch→yaw, ``Rz @ Ry @ Rx``). This
        matches the intent of ``Rotation.from_euler("xyz", [...])`` from scientific Python
        stacks; LeRobot's :class:`~lerobot.utils.rotation.Rotation` only exposes matrix and
        quaternion constructors, so conversion goes through :func:`euler_xyz_to_rotvec`.

        Quaternion input is also supported by calling
        ``Rotation.from_quat([x, y, z, w]).as_rotvec()`` and passing that value to the
        ``rotvec`` field directly.
        """
        normalized_sequence = sequence.lower().strip()
        if normalized_sequence != _EULER_SEQUENCE_XYZ:
            raise ValueError(
                f"Only Euler sequence {_EULER_SEQUENCE_XYZ!r} is supported, got {sequence!r}"
            )
        rotation_vector = euler_xyz_to_rotvec(roll, pitch, yaw, degrees=degrees)
        return cls(
            name=name,
            x=x,
            y=y,
            z=z,
            rotvec=rotation_vector,
            gripper_pos=gripper_pos,
            hold_s=hold_s,
        )

    def to_action(self) -> dict[str, float]:
        return {
            "ee.x": self.x,
            "ee.y": self.y,
            "ee.z": self.z,
            "ee.wx": self.rotvec[0],
            "ee.wy": self.rotvec[1],
            "ee.wz": self.rotvec[2],
            "ee.gripper_pos": self.gripper_pos,
        }


def _float_value(pose_data: dict[str, Any], key: str) -> float:
    value = pose_data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Pose key {key!r} must be numeric, got {value!r}")
    return float(value)


def _tuple3(value: Any, key_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key_name!r} must be a 3-element list or tuple")
    return (float(value[0]), float(value[1]), float(value[2]))


def ee_pose_from_mapping(pose_data: dict[str, Any], pose_units: str) -> EEPose:
    """Parse an external pose mapping using explicit angle units."""
    if pose_units not in POSE_UNITS_CHOICES:
        raise ValueError(
            f"Unsupported pose units {pose_units!r}; expected one of {POSE_UNITS_CHOICES}"
        )
    name = str(pose_data.get("name", "unnamed"))
    x = _float_value(pose_data, "x")
    y = _float_value(pose_data, "y")
    z = _float_value(pose_data, "z")
    gripper_pos = _float_value(pose_data, "gripper_pos")
    hold_s = float(pose_data.get("hold_s", 1.0))
    if pose_units == POSE_UNITS_ROTATION_VECTOR:
        return EEPose(
            name=name,
            x=x,
            y=y,
            z=z,
            rotvec=_tuple3(pose_data.get("rotvec"), "rotvec"),
            gripper_pos=gripper_pos,
            hold_s=hold_s,
        )
    roll = _float_value(pose_data, "roll")
    pitch = _float_value(pose_data, "pitch")
    yaw = _float_value(pose_data, "yaw")
    return EEPose.from_euler(
        name=name,
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        gripper_pos=gripper_pos,
        hold_s=hold_s,
        sequence=str(pose_data.get("sequence", "xyz")),
        degrees=pose_units == POSE_UNITS_EULER_DEGREES,
    )


def load_poses_file(poses_path: str, pose_units: str) -> list[EEPose]:
    """Load pose list from JSON or YAML and normalize to :class:`EEPose`."""
    path = Path(poses_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        loaded = json.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML pose files require PyYAML. Install it with: pip install pyyaml"
            ) from exc
        loaded = yaml.safe_load(path.read_text())
    else:
        raise ValueError(
            f"Unsupported poses file format {suffix!r}; use .json, .yaml, or .yml"
        )
    if not isinstance(loaded, list):
        raise ValueError("Poses file must contain a top-level list of pose objects")
    return [ee_pose_from_mapping(pose_data, pose_units) for pose_data in loaded]


POSES: list[EEPose] = [
    EEPose.from_euler(
        name="HOME",
        x=.16,
        y=0.03,
        z=.40,
        roll=0.0,
        pitch=0.0,
        yaw=-200.0,
        gripper_pos=20,
        hold_s=1.0,
    ),
    EEPose.from_euler(
        name="LEFT_SHIFT",
        x=0.15,
        y=0.03,
        z=0.15,
        roll=0.0,
        pitch=0.0,
        yaw=5.0,
        gripper_pos=50.0,
        hold_s=1.0,
    ),
    EEPose.from_euler(
        name="RIGHT_SHIFT",
        x=0.15,
        y=-0.03,
        z=0.15,
        roll=0.0,
        pitch=0.0,
        yaw=-5.0,
        gripper_pos=50.0,
        hold_s=1.0,
    ),
    EEPose.from_euler(
        name="UP_SHIFT",
        x=0.15,
        y=0.00,
        z=0.17,
        roll=0.0,
        pitch=5.0,
        yaw=0.0,
        gripper_pos=50.0,
        hold_s=1.0,
    ),
]

# Wider end-effector sweep for bring-up: re-tune x/y/z and angles for your arm and table setup.
RANGE_DEMO_POSES: list[EEPose] = [
    EEPose.from_euler(
        name="HOME",
        x=0,
        y=10.0,
        z=0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        gripper_pos=-50.0,
        hold_s=3.3,
    ),
    EEPose.from_euler(
        name="REACH_FORWARD",
        x=-10.22,
        y=10.0,
        z=0.14,
        roll=0.0,
        pitch=8.0,
        yaw=0.0,
        gripper_pos=50.0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="REACH_IN",
        x=0.12,
        y=20.0,
        z=0.16,
        roll=0.0,
        pitch=6.0,
        yaw=0.0,
        gripper_pos=0.0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="LEFT",
        x=0.15,
        y=0.10,
        z=10.15,
        roll=0.0,
        pitch=0.0,
        yaw=18.0,
        gripper_pos=100.0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="RIGHT",
        x=0.15,
        y=-0.10,
        z=0.15,
        roll=40.0,
        pitch=0.0,
        yaw=-18.0,
        gripper_pos=10.0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="HIGH",
        x=0.15,
        y=0.0,
        z=0.27,
        roll=0.0,
        pitch=50.0,
        yaw=0.0,
        gripper_pos=0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="LOW",
        x=0.16,
        y=0.0,
        z=0.10,
        roll=0.0,
        pitch=-900.0,
        yaw=5.0,
        gripper_pos=10.0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="WRIST_ROLL",
        x=0.15,
        y=0.0,
        z=0.15,
        roll=22.0,
        pitch=0.0,
        yaw=0.0,
        gripper_pos=100.0,
        hold_s=2.2,
    ),
    EEPose.from_euler(
        name="GRIPPER_OPEN",
        x=0.15,
        y=0.0,
        z=0.15,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        gripper_pos=0.0,
        hold_s=2.0,
    ),
    EEPose.from_euler(
        name="GRIPPER_CLOSED",
        x=0.15,
        y=0.0,
        z=0.15,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        gripper_pos=85.0,
        hold_s=2.0,
    ),
]

SEQUENCES: dict[str, list[EEPose]] = {
    "home_left_right": [POSES[0], POSES[1], POSES[2]],
    "home_up": [POSES[0], POSES[3]],
    "home_only": [POSES[0]],
    "range_demo": RANGE_DEMO_POSES,
}
