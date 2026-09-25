"""Headless Gazebo steps a world with a falling ball and agrees with free fall and MuJoCo."""

import json

import mujoco
import pytest
from states import same_state
from support import dockerenv

pytestmark = pytest.mark.medium

G = 9.81
DT = 0.001
STEPS = 100
RELEASE_HEIGHT = 10.0

# Exact for a semi-implicit Euler integrator (velocity first, then position): after n steps the
# drop is g * dt^2 * n * (n + 1) / 2. Gazebo's DART and MuJoCo both integrate this way.
DISCRETE_DROP = G * DT**2 * STEPS * (STEPS + 1) / 2
# Continuous free fall, 0.5 * g * t^2 at t = 0.1 s. The discrete drop sits 1 % above it.
ANALYTIC_DROP = 0.5 * G * (STEPS * DT) ** 2

WORLD = "/worlds/falling_body.sdf"


def gazebo_drop(built_image, steps=STEPS):
    assert built_image.returncode == 0, built_image.stderr[-2000:]
    result = dockerenv.run("step_world", WORLD, str(steps))
    assert result.returncode == 0, result.stderr[-2000:]
    state = json.loads(result.stdout)
    return state, RELEASE_HEIGHT - state["poses"]["ball"][2]


def mujoco_drop():
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><option timestep="{DT}"/><worldbody>
          <body pos="0 0 {RELEASE_HEIGHT}">
            <freejoint/><geom type="sphere" size="0.1" mass="1"/>
          </body>
        </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
    return RELEASE_HEIGHT - data.qpos[2]


def test_gazebo_advances_exactly_the_requested_iterations(built_image):
    state, _ = gazebo_drop(built_image)
    assert state["iterations"] == STEPS
    assert state["sim_time"] == pytest.approx(STEPS * DT)


def test_the_ball_drops_by_the_discrete_free_fall_distance(built_image):
    _, drop = gazebo_drop(built_image)
    assert drop == pytest.approx(DISCRETE_DROP, abs=1e-6)


def test_the_drop_is_within_two_percent_of_continuous_free_fall(built_image):
    _, drop = gazebo_drop(built_image)
    assert drop == pytest.approx(ANALYTIC_DROP, rel=0.02)


def test_gazebo_and_mujoco_drop_the_same_distance(built_image):
    _, drop = gazebo_drop(built_image)
    assert drop == pytest.approx(mujoco_drop(), abs=1e-6)


def test_zero_steps_read_the_released_pose_without_stepping(built_image):
    state, drop = gazebo_drop(built_image, steps=0)
    assert state["iterations"] == 0
    assert drop == pytest.approx(0.0, abs=1e-12)


def test_two_runs_give_identical_poses(built_image):
    first, _ = gazebo_drop(built_image)
    second, _ = gazebo_drop(built_image)
    assert same_state(first, second), (first, second)
