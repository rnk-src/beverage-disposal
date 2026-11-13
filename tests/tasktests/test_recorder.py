"""Recording a real episode: what the recorder keeps, and that the task tests accept a good run."""

import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.kinematics import HOME
from beverage_disposal.core.layout import goals
from beverage_disposal.core.scripted import ScriptedController
from beverage_disposal.core.task_tests import evaluate
from beverage_disposal.sim.loop import CONTROL_PERIOD, ArmLoop
from beverage_disposal.sim.recorder import Recorder
from beverage_disposal.sim.rollout import run_episode
from beverage_disposal.sim.scene import build_scene

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
GOALS = goals(CONFIG)


def scripted_episode(goal, seed=0):
    return run_episode(CONFIG, SO101 / MJCF, ScriptedController(CONFIG), GOALS[goal], seed=seed)


def test_a_recorded_episode_has_one_row_per_step_in_the_control_period():
    episode = scripted_episode("set_aside")
    trajectory = episode.trajectory
    assert len(trajectory.time) == episode.steps
    assert np.diff(trajectory.time) == pytest.approx(CONTROL_PERIOD)
    assert trajectory.tip.shape == (episode.steps, 3)
    assert trajectory.action.shape == (episode.steps, 4)
    assert trajectory.finished


def test_the_last_recorded_container_position_is_where_the_episode_ended():
    episode = scripted_episode("set_aside")
    assert episode.trajectory.container[-1] == pytest.approx(episode.container, abs=5e-3)


def test_the_recorded_actions_are_the_controllers_actions():
    # The script starts by moving towards the can with the gripper open and never exceeds [-1, 1].
    actions = scripted_episode("pad").trajectory.action
    assert np.all(np.abs(actions) <= 1.0)
    assert actions[0, 3] == pytest.approx(
        2 * (CONFIG.grasp.open_angle - (-0.1745)) / (1.7453 + 0.1745) - 1, abs=1e-3
    )


@pytest.mark.parametrize("goal", ["set_aside", "pad", "bin"])
def test_the_task_tests_pass_a_scripted_run_on_every_goal(goal):
    episode = scripted_episode(goal)
    verdict = evaluate(episode.trajectory, GOALS[goal], CONFIG)
    assert verdict.success, verdict.passed


def test_the_task_tests_fail_a_scripted_run_judged_against_the_wrong_goal():
    episode = scripted_episode("set_aside")
    verdict = evaluate(episode.trajectory, GOALS["bin"], CONFIG)
    assert verdict.passed["grasped"] and not verdict.passed["placed"]


def test_the_arm_pressing_into_the_table_is_recorded_as_force_and_a_resting_can_is_not():
    model = build_scene(CONFIG, SO101 / MJCF)
    loop = ArmLoop(model)
    start = np.array(HOME, dtype=float)
    start[4] = CONFIG.grasp.wrist_roll
    pressed = solve_ik(np.array([0.2, -0.1, -0.01]), np.radians(60), start).angles[:5]
    loop.reset(pressed, 0.5)
    recorder = Recorder(model)
    for _ in range(20):
        loop.step(pressed, 0.5)
        recorder.record(loop, np.zeros(4))
    forces = recorder.trajectory(finished=False).arm_force
    assert forces.max() > 1.0  # the pads dig into the table

    quiet = ArmLoop(model)
    quiet.reset(HOME, 0.0)
    recorder = Recorder(model)
    for _ in range(20):
        quiet.step(HOME, 0.0)
        recorder.record(quiet, np.zeros(4))
    # The can rests on the table, the arm touches nothing: no arm force.
    assert recorder.trajectory(finished=False).arm_force.max() == 0.0
