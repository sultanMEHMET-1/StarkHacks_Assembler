"""Tests for EE segment sub-stepping (EEBoundsAndSafety compatibility)."""

from __future__ import annotations

import numpy as np
import pytest

from arm_mover.runner import _EE_STEP_MARGIN_M, _ee_actions_along_segment


def _minimal_ee(x: float, y: float, z: float) -> dict[str, float]:
    return {
        "ee.x": x,
        "ee.y": y,
        "ee.z": z,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 50.0,
    }


def test_segment_none_start_is_single_goal() -> None:
    goal = _minimal_ee(0.15, 0.0, 0.15)
    actions = _ee_actions_along_segment(None, goal, max_step_m=0.02)
    assert actions == [goal]


def test_segment_position_steps_respect_max_step() -> None:
    start = _minimal_ee(0.0, 0.0, 0.0)
    end = _minimal_ee(0.07, 0.0, 0.0)
    actions = _ee_actions_along_segment(start, end, max_step_m=0.02)
    assert len(actions) == 4
    max_step_m = 0.02
    effective_cap = max_step_m - _EE_STEP_MARGIN_M
    for index in range(1, len(actions)):
        previous = np.array(
            [actions[index - 1]["ee.x"], actions[index - 1]["ee.y"], actions[index - 1]["ee.z"]]
        )
        current = np.array([actions[index]["ee.x"], actions[index]["ee.y"], actions[index]["ee.z"]])
        step_m = float(np.linalg.norm(current - previous))
        assert step_m <= effective_cap + 1e-9
    assert actions[-1]["ee.x"] == pytest.approx(0.07)
