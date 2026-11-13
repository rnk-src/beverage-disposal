"""What counts as a success: wholly inside the goal, on its surface, upright and at rest."""

import mujoco
import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.layout import goals
from beverage_disposal.sim.rollout import judge
from beverage_disposal.sim.scene import build_scene

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
GOAL = goals(CONFIG)["pad"]
HALF_HEIGHT = 0.061
MODEL = build_scene(CONFIG, SO101 / MJCF)


def verdict(dx=0.0, dy=0.0, dz=0.0, tilt_deg=0.0, speed=0.0, goal=GOAL):
    """Judge a can placed at the goal's centre, moved by (dx, dy, dz), tilted and moving."""
    data = mujoco.MjData(MODEL)
    address = MODEL.joint("container").qposadr[0]
    half = np.radians(tilt_deg) / 2
    data.qpos[address : address + 7] = [
        goal.center_x + dx,
        goal.center_y + dy,
        goal.rest_z + HALF_HEIGHT + dz,
        np.cos(half),
        np.sin(half),
        0.0,
        0.0,
    ]
    data.qvel[MODEL.joint("container").dofadr[0]] = speed
    mujoco.mj_forward(MODEL, data)
    return judge(CONFIG, data, goal)


def test_a_can_standing_in_the_middle_of_the_goal_is_a_success():
    assert verdict().success


def test_a_can_just_inside_the_edge_is_a_success_and_just_outside_is_not():
    limit = GOAL.half_x - CONFIG.container.diameter / 2
    assert verdict(dx=limit - 1e-3).success
    outside = verdict(dx=limit + 1e-3)
    assert not outside.success and "inside" in outside.reason


def test_a_can_floating_above_the_surface_is_not_a_success():
    result = verdict(dz=0.01)
    assert not result.success and "surface" in result.reason


def test_a_can_leaning_over_is_not_a_success():
    assert verdict(tilt_deg=4.0).success
    result = verdict(tilt_deg=8.0)
    assert not result.success and "upright" in result.reason


def test_a_can_still_moving_is_not_a_success():
    assert verdict(speed=0.005).success
    result = verdict(speed=0.05)
    assert not result.success and "rest" in result.reason


@pytest.mark.parametrize("name", ["pad", "bin", "set_aside"])
def test_each_goal_judges_a_can_at_its_own_resting_height(name):
    goal = goals(CONFIG)[name]
    assert verdict(goal=goal).success
