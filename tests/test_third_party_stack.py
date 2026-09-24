"""Smoke tests that the pinned physics and space libraries behave on this machine."""

import gymnasium
import mujoco
import numpy as np
import pytest

GRAVITY = 9.81
# Empty and full can masses from the task decision, in kg.
CAN_MASSES = [0.014, 0.37]

FLOOR = '<geom type="plane" size="1 1 0.1"/>'


def world(body_mass, floor=""):
    return f"""
    <mujoco>
      <worldbody>
        {floor}
        <body pos="0 0 0.5">
          <freejoint/>
          <geom type="sphere" size="0.05" mass="{body_mass}"/>
        </body>
      </worldbody>
    </mujoco>
    """


def simulate(xml, seconds):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


@pytest.mark.parametrize("mass", CAN_MASSES)
def test_free_fall_covers_the_distance_and_speed_physics_predicts(mass):
    data = simulate(world(mass), 0.1)
    dropped = 0.5 - data.qpos[2]
    # 0.5 * g * t**2 by hand; semi-implicit Euler lands about 2% above it, hence the 3%.
    assert dropped == pytest.approx(0.5 * GRAVITY * 0.1**2, rel=0.03)
    assert data.qvel[2] == pytest.approx(-GRAVITY * 0.1, rel=0.03)


def test_a_sphere_dropped_on_a_floor_comes_to_rest_on_it():
    data = simulate(world(0.37, FLOOR), 2.0)
    assert data.qpos[2] == pytest.approx(0.05, abs=0.002)
    assert abs(data.qvel[2]) < 0.01


ACTION_SPACE = gymnasium.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)


def test_the_action_space_has_the_decided_shape_and_dtype():
    assert ACTION_SPACE.shape == (4,)
    assert ACTION_SPACE.dtype == np.float32


def test_sampled_actions_stay_inside_the_bounds():
    ACTION_SPACE.seed(0)
    samples = np.array([ACTION_SPACE.sample() for _ in range(300)])
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
    assert samples.min() < -0.5 and samples.max() > 0.5, "samples do not spread over the range"


@pytest.mark.parametrize("edge", [-1.0, 1.0])
def test_the_edges_of_the_range_are_valid_actions(edge):
    assert ACTION_SPACE.contains(np.full(4, edge, dtype=np.float32))


@pytest.mark.parametrize("dimension", range(4))
@pytest.mark.parametrize("value", [-1.01, 1.01])
def test_a_value_just_outside_the_range_is_rejected_on_every_dimension(dimension, value):
    action = np.zeros(4, dtype=np.float32)
    action[dimension] = value
    assert not ACTION_SPACE.contains(action)


def test_an_action_of_the_wrong_shape_is_rejected():
    assert not ACTION_SPACE.contains(np.zeros(3, dtype=np.float32))
