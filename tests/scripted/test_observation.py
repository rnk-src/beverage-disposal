"""The observation the skill sees, its noise hook and the controller interface."""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest
from support.observations import observation

from beverage_disposal.core.baselines import RandomController
from beverage_disposal.core.interfaces import Controller, Observation, PositionNoise
from beverage_disposal.core.kinematics import HOME

OBS = observation(HOME, 0.5, (0.30, 0.0, 0.061), (0.20, -0.17, 0.005))


def test_the_observation_is_version_one_and_read_only():
    assert Observation.VERSION == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        OBS.gripper = 1.0


def test_the_helper_stores_the_container_and_goal_relative_to_the_tip():
    assert OBS.tip + OBS.container == pytest.approx([0.30, 0.0, 0.061])
    assert OBS.tip + OBS.goal == pytest.approx([0.20, -0.17, 0.005])


def test_zero_noise_returns_the_same_numbers():
    noisy = PositionNoise(sigma=0.0).apply(OBS, np.random.default_rng(0))
    np.testing.assert_array_equal(noisy.container, OBS.container)


def test_noise_only_touches_the_container_position():
    noisy = PositionNoise(sigma=0.01).apply(OBS, np.random.default_rng(0))
    assert not np.array_equal(noisy.container, OBS.container)
    np.testing.assert_array_equal(noisy.joints, OBS.joints)
    np.testing.assert_array_equal(noisy.tip, OBS.tip)
    np.testing.assert_array_equal(noisy.goal, OBS.goal)
    assert noisy.gripper == OBS.gripper


def test_noise_is_gaussian_with_the_stated_spread_and_repeatable_by_seed():
    sigma = 0.005
    rng = np.random.default_rng(1)
    draws = np.array(
        [PositionNoise(sigma).apply(OBS, rng).container - OBS.container for _ in range(4000)]
    )
    # A spread estimated from 4000 draws is good to about 1.1 %, so 10 % is a wide margin.
    assert draws.std() == pytest.approx(sigma, rel=0.1)
    assert abs(draws.mean()) < 5 * sigma / np.sqrt(len(draws) * 3)
    again = PositionNoise(sigma).apply(OBS, np.random.default_rng(5)).container
    assert np.array_equal(
        again, PositionNoise(sigma).apply(OBS, np.random.default_rng(5)).container
    )


def test_a_negative_noise_spread_is_rejected():
    with pytest.raises(ValueError, match="sigma"):
        PositionNoise(sigma=-0.001)


def test_a_random_controller_acts_with_four_numbers_in_range_and_is_a_controller():
    controller = RandomController(seed=0)
    controller.reset(OBS)
    action = controller.act(OBS)
    assert isinstance(controller, Controller)
    assert action.shape == (4,) and np.all(np.abs(action) <= 1)


def test_the_same_random_seed_gives_the_same_actions():
    first, second = RandomController(seed=3), RandomController(seed=3)
    for controller in (first, second):
        controller.reset(OBS)
    assert np.array_equal(first.act(OBS), second.act(OBS))


WITHOUT_MUJOCO = """
import sys
sys.modules["mujoco"] = None
import numpy as np
from beverage_disposal.core.baselines import RandomController
from beverage_disposal.core.interfaces import Observation, PositionNoise
from beverage_disposal.core.scripted import ScriptedController
print("ok")
"""


def test_the_skill_side_modules_import_with_mujoco_unavailable():
    result = subprocess.run([sys.executable, "-c", WITHOUT_MUJOCO], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
