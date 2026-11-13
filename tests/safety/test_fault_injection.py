"""Faults injected into a running episode: each must end in a safe stop, never a placement."""

import dataclasses
import math
from dataclasses import replace

import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.layout import goals
from beverage_disposal.core.safety_config import load_safety_limits
from beverage_disposal.core.scripted import ScriptedController
from beverage_disposal.core.supervised import SupervisedController
from beverage_disposal.core.supervisor import RecordingIndicators, Supervisor
from beverage_disposal.sim.rollout import run_episode

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
LIMITS = load_safety_limits(ROOT / "configs" / "safety.toml")
GOAL = goals(CONFIG)["set_aside"]


class NanFrom:
    """Observation noise that turns the container position into NaN from call `step` on."""

    def __init__(self, step):
        self._step, self._calls = step, 0

    def apply(self, observation, rng):
        self._calls += 1
        if self._calls > self._step:
            return replace(observation, container=np.full(3, math.nan))
        return observation


class Stuck:
    """A controller that never finishes and never moves."""

    done = False

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.zeros(4)


class BrokenSupervisor(Supervisor):
    def check_observation(self, observation):
        raise RuntimeError("injected fault")


def supervised(controller, supervisor_class=Supervisor):
    indicators = RecordingIndicators()
    supervisor = supervisor_class(LIMITS, indicators)
    return SupervisedController(controller, supervisor), supervisor, indicators


def episode(controller, **options):
    return run_episode(CONFIG, SO101 / MJCF, controller, GOAL, seed=0, **options)


@pytest.mark.medium
def test_a_normal_supervised_run_still_succeeds_and_the_supervisor_stays_silent():
    controller, supervisor, indicators = supervised(ScriptedController(CONFIG))
    result = episode(controller)
    assert result.success, result.reason
    assert supervisor.events == [] and indicators.log == []


@pytest.mark.medium
def test_nan_in_the_container_position_stops_the_arm_and_nothing_is_placed():
    controller, supervisor, indicators = supervised(ScriptedController(CONFIG))
    result = episode(controller, noise=NanFrom(step=30))
    assert result.reason == "safe stop: non_finite_observation"
    assert not result.success
    assert [kind for kind, _ in indicators.log].count("light") == 1


@pytest.mark.medium
def test_a_sensor_that_dies_at_once_stops_before_the_arm_moves_far():
    controller, _, _ = supervised(ScriptedController(CONFIG))
    result = episode(controller, noise=NanFrom(step=0))
    assert result.reason == "safe stop: non_finite_observation"
    assert result.steps == 1 + 0  # stopped on the first step; no settling run follows
    assert np.linalg.norm(result.trajectory.tip[-1] - result.trajectory.tip[0]) < 0.01


@pytest.mark.medium
def test_a_controller_that_never_finishes_is_stopped_by_the_timeout():
    controller, supervisor, _ = supervised(Stuck())
    result = episode(controller)
    assert result.reason == "safe stop: timeout"
    # 20 s at 20 Hz is 400 steps; the stop comes on the first step past that.
    assert result.steps == pytest.approx(401, abs=1)
    assert supervisor.events[0].code == "timeout"


@pytest.mark.medium
def test_a_supervisor_that_throws_stops_the_arm_instead_of_letting_it_go_on():
    controller, supervisor, indicators = supervised(ScriptedController(CONFIG), BrokenSupervisor)
    result = episode(controller)
    assert result.reason == "safe stop: supervisor_fault"
    assert "injected fault" in supervisor.events[0].message
    assert indicators.log


@pytest.mark.medium
def test_the_stop_ends_the_episode_at_once_with_the_arm_told_to_hold_still():
    controller, _, _ = supervised(ScriptedController(CONFIG))
    result = episode(controller, noise=NanFrom(step=40))
    # The episode ends on the step the fault is seen: it does not run on to the end of the script.
    assert result.steps < 45
    assert (
        result.trajectory.action[-1, :3].max() == 0.0
        and result.trajectory.action[-1, :3].min() == 0.0
    )


def test_a_non_finite_action_from_the_controller_is_caught_before_it_reaches_the_arm():
    class Bad:
        done = False

        def reset(self, observation):
            pass

        def act(self, observation):
            return np.array([0.0, math.nan, 0.0, 0.0])

    controller, supervisor, _ = supervised(Bad())
    from support.observations import observation

    from beverage_disposal.core.kinematics import HOME

    obs = observation(HOME, 0.5, (0.27, 0.0, 0.061), (0.17, 0.15, 0.02))
    controller.reset(obs)
    action = controller.act(obs)
    assert np.all(action == 0.0)
    assert controller.stop_event.code == "non_finite_action"
    assert dataclasses.is_dataclass(controller.stop_event)


def test_once_stopped_the_controller_ignores_a_good_observation_and_holds_still():
    class Moves:
        done = False

        def reset(self, observation):
            pass

        def act(self, observation):
            return np.array([1.0, 1.0, 1.0, 1.0])

    from support.observations import observation

    from beverage_disposal.core.kinematics import HOME

    good = observation(
        HOME, 0.5, (0.27, 0.0, 0.061), (0.17, 0.15, 0.02), previous_action=[0, 0, 0, 0.4]
    )
    controller, _, _ = supervised(Moves())
    controller.reset(good)
    assert controller.act(good)[0] == 1.0  # healthy: the controller's action goes through
    bad = replace(good, container=np.full(3, math.nan))
    controller.act(bad)  # the fault
    held = controller.act(good)  # the sensors are fine again, but the arm stays stopped
    assert held[:3].tolist() == [0.0, 0.0, 0.0] and held[3] == pytest.approx(0.4)
    assert controller.done
