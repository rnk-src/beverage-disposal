"""The three places a container can be put: the pad, the bin and the set-aside area."""

import pytest
from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.layout import goals

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
CAN_RADIUS = 0.033


def test_there_are_three_goals_with_their_release_and_rest_heights():
    found = goals(CONFIG)
    assert set(found) == {"pad", "bin", "set_aside"}
    # A container is set down on the pad (top 0.02 m) and on the table, and dropped into the bin
    # from just above its wall (0.06 m plus the 0.01 m release clearance) to its floor (0.005 m).
    assert (found["pad"].release_z, found["pad"].rest_z) == pytest.approx((0.02, 0.02))
    assert (found["bin"].release_z, found["bin"].rest_z) == pytest.approx((0.07, 0.005))
    assert (found["set_aside"].release_z, found["set_aside"].rest_z) == pytest.approx((0.0, 0.0))


def test_the_bin_goal_is_its_free_floor_not_its_outer_edge():
    bin_ = goals(CONFIG)["bin"]
    assert (bin_.half_x, bin_.half_y) == pytest.approx((0.05, 0.05))


@pytest.mark.parametrize("name", ["pad", "bin", "set_aside"])
def test_a_container_holds_when_wholly_inside_the_goal_and_not_when_it_overhangs(name):
    goal = goals(CONFIG)[name]
    limit = goal.half_x - CAN_RADIUS
    assert goal.holds(goal.center_x, goal.center_y, CAN_RADIUS)
    assert goal.holds(goal.center_x + limit - 1e-6, goal.center_y, CAN_RADIUS)
    assert not goal.holds(goal.center_x + limit + 1e-6, goal.center_y, CAN_RADIUS)
