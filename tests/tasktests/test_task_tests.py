"""The task's pass-fail and indicative functions, checked on trajectories with a known story."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.paths import ROOT
from support.trajectories import HALF_HEIGHT, trajectory

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.layout import goals
from beverage_disposal.core.task_tests import evaluate

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
PAD = goals(CONFIG)["pad"]
BIN = goals(CONFIG)["bin"]
SPAWN = np.array([0.27, 0.0, HALF_HEIGHT])
LIFTED = SPAWN + [0.0, 0.0, 0.03]
ON_PAD = np.array([PAD.center_x, PAD.center_y, PAD.rest_z + HALF_HEIGHT])
OVER_PAD = ON_PAD + [0.0, 0.0, 0.03]
TIP_OFFSET = np.array([0.0, -0.025, 0.04])  # the tip sits beside and above the can's centre


def perfect():
    return [
        {"steps": 10, "tip": SPAWN + TIP_OFFSET + [0, 0, 0.06], "container": SPAWN},
        {
            "steps": 20,
            "tip": (SPAWN + TIP_OFFSET, LIFTED + TIP_OFFSET),
            "container": (SPAWN, LIFTED),
        },
        {
            "steps": 30,
            "tip": (LIFTED + TIP_OFFSET, OVER_PAD + TIP_OFFSET),
            "container": (LIFTED, OVER_PAD),
        },
        {"steps": 20, "tip": OVER_PAD + TIP_OFFSET + [0, 0, 0.05], "container": ON_PAD},
    ]


def verdict(segments, goal=PAD, finished=True):
    return evaluate(trajectory(segments, finished=finished), goal, CONFIG)


def test_a_perfect_run_passes_every_test():
    result = verdict(perfect())
    assert all(result.passed.values()), result.passed
    assert result.success


def test_a_can_dropped_on_the_way_has_been_grasped_but_is_not_on_the_goal():
    story = perfect()[:3] + [
        {"steps": 20, "tip": OVER_PAD + TIP_OFFSET, "container": SPAWN + [0.05, 0, 0]}
    ]
    result = verdict(story)
    assert result.passed["grasped"] and not result.passed["placed"]
    assert not result.success


def test_a_can_that_never_moves_was_never_grasped():
    result = verdict([{"steps": 60, "tip": SPAWN + TIP_OFFSET, "container": SPAWN}])
    assert not result.passed["grasped"] and not result.passed["placed"]


def test_a_can_lifted_for_only_a_moment_was_not_grasped():
    flick = [
        {"steps": 10, "tip": SPAWN + TIP_OFFSET, "container": SPAWN},
        {"steps": 2, "tip": SPAWN + TIP_OFFSET, "container": LIFTED},
        {"steps": 40, "tip": SPAWN + TIP_OFFSET, "container": SPAWN},
    ]
    assert not verdict(flick).passed["grasped"]


def test_a_can_up_in_the_air_far_from_the_tool_was_not_grasped():
    # Lifted and staying up, but half a metre from the tip: the jaws are not holding it.
    away = [
        {"steps": 10, "tip": SPAWN + TIP_OFFSET, "container": SPAWN},
        {"steps": 30, "tip": SPAWN + TIP_OFFSET, "container": LIFTED + [0.4, 0.0, 0.0]},
    ]
    assert not verdict(away).passed["grasped"]


def test_a_can_that_tips_over_on_the_way_fails_not_knocked_over():
    story = perfect()
    story[2] = dict(story[2], upright=0.2)
    result = verdict(story)
    assert not result.passed["not_knocked_over"] and not result.success


def test_a_can_flung_fast_fails_not_thrown():
    story = perfect()
    story[2] = dict(story[2], speed=3.0)
    result = verdict(story)
    assert not result.passed["not_thrown"] and not result.success


def test_an_arm_pressing_hard_on_the_table_fails_arm_safe():
    story = perfect()
    story[0] = dict(story[0], arm_force=60.0)
    result = verdict(story)
    assert not result.passed["arm_safe"] and not result.success


def test_an_episode_that_runs_out_of_time_fails_in_time():
    result = verdict(perfect(), finished=False)
    assert not result.passed["in_time"] and not result.success


def test_a_can_on_the_wrong_goal_has_not_been_placed():
    inside_bin = np.array([BIN.center_x, BIN.center_y, BIN.rest_z + HALF_HEIGHT])
    story = perfect()[:3] + [{"steps": 20, "tip": OVER_PAD + TIP_OFFSET, "container": inside_bin}]
    result = verdict(story)
    assert result.passed["grasped"] and not result.passed["placed"]
    assert verdict(story, goal=BIN).passed["placed"]


def test_the_indicators_are_numbers_that_say_how_close_the_run_came():
    result = verdict(perfect())
    # By hand: the can rises from the table (0.061 m) to 0.03 m above the pad top (0.02 m).
    assert result.indicators["height_gained"] == pytest.approx(0.02 + 0.03, abs=1e-9)
    assert result.indicators["final_distance"] == pytest.approx(0.0, abs=1e-9)
    assert result.indicators["closest_approach"] == pytest.approx(
        np.linalg.norm(TIP_OFFSET), abs=1e-6
    )
    assert result.indicators["duration"] == pytest.approx(79 * 0.05)
    assert result.indicators["smoothness"] == pytest.approx(0.0)


def test_smoothness_is_the_mean_jump_between_consecutive_actions():
    story = [
        {"steps": 5, "tip": SPAWN, "container": SPAWN, "action": (0.0, 0.0, 0.0, 0.0)},
        {"steps": 5, "tip": SPAWN, "container": SPAWN, "action": (1.0, 0.0, 0.0, 0.0)},
    ]
    # One jump of 1.0 among nine steps between ten actions.
    assert verdict(story).indicators["smoothness"] == pytest.approx(1.0 / 9)


@settings(max_examples=40, deadline=None)
@given(near=st.floats(0.0, 0.03), far=st.floats(0.0, 0.2))
def test_a_closer_final_position_gives_a_smaller_distance_and_never_loses_the_placement(near, far):
    near, far = sorted([near, far])

    def at(offset):
        end = ON_PAD + [offset, 0, 0]
        story = perfect()[:3] + [{"steps": 20, "tip": OVER_PAD + TIP_OFFSET, "container": end}]
        return verdict(story)

    closer, farther = at(near), at(far)
    assert closer.indicators["final_distance"] <= farther.indicators["final_distance"]
    assert closer.passed["placed"] or not farther.passed["placed"]
