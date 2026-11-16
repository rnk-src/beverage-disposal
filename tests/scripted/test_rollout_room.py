"""The episode runner starts above whatever container it is given and lets the tip go that high."""

import dataclasses

from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.scripted import approach_height_m
from beverage_disposal.sim.rollout import READY_TIP, WORKSPACE, ready_tip

SCENE = load_scene_config(ROOT / "configs" / "scene.toml")
# The tallest 500 mL bottle of the family is 0.234 m; on the pad its top is at 0.254 m.
TALL = dataclasses.replace(SCENE.container, height=0.234, grasp_height=0.219)


def test_the_can_starts_from_the_same_ready_point_as_before():
    assert ready_tip(SCENE, surface_z=0.0) == READY_TIP == (0.26, 0.0, 0.15)


def test_a_tall_container_on_the_pad_is_started_from_above_its_approach_height():
    config = dataclasses.replace(SCENE, container=TALL)
    x, y, z = ready_tip(config, surface_z=SCENE.pad.height)
    assert (x, y) == READY_TIP[:2]
    assert z == SCENE.pad.height + approach_height_m(config)


def test_the_tip_may_go_as_high_as_the_carry_of_the_tallest_reachable_bottle_over_the_bin():
    carry = SCENE.bin.height + SCENE.reach.release_clearance + TALL.grasp_height + 0.015
    assert WORKSPACE[1][2] >= carry + 0.02
