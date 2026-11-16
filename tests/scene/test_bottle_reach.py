"""Reach of one leg from the pad, for any container: the poses and the pitch that reaches them."""

import dataclasses

import pytest
from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.reach import container_poses, reach_for_sizes, steepest_common_pitch
from beverage_disposal.core.scripted import CARRY_CLEARANCE

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
CAN = CONFIG.container  # grasp height 0.10 m
TALL = dataclasses.replace(CAN, name="tall", height=0.62, grasp_height=0.60)


def test_the_poses_follow_the_leg_from_the_pad_at_the_containers_grasp_height():
    poses = dict(container_poses(CONFIG, CAN))
    pad, bin_, area, reach = CONFIG.pad, CONFIG.bin, CONFIG.set_aside, CONFIG.reach
    assert poses["pad:approach"] == pytest.approx((pad.center_x, pad.center_y, 0.02 + 0.10 + 0.04))
    assert poses["pad:grasp"] == pytest.approx((pad.center_x, pad.center_y, 0.02 + 0.10))
    assert poses["pad:lift"] == pytest.approx((pad.center_x, pad.center_y, 0.02 + 0.10 + 0.03))
    # Carried with the bottom `release_clearance` over the wall plus the script's carry clearance.
    over_wall = bin_.height + reach.release_clearance + 0.10
    assert poses["bin:carry"] == pytest.approx(
        (bin_.center_x, bin_.center_y, over_wall + CARRY_CLEARANCE)
    )
    assert poses["bin:release"] == pytest.approx((bin_.center_x, bin_.center_y, over_wall))
    # The carry to the set-aside area stays at the lift height: it is above the area's own carry.
    assert poses["set_aside:carry"] == pytest.approx((area.center_x, area.center_y, 0.15))
    assert poses["set_aside:place"] == pytest.approx((area.center_x, area.center_y, 0.10))


def test_the_poses_move_with_the_containers_own_grasp_height():
    neck = dataclasses.replace(CAN, grasp_height=0.17)
    assert dict(container_poses(CONFIG, neck))["pad:grasp"][2] == pytest.approx(0.19)


def test_the_cans_leg_from_the_pad_is_reachable_and_a_grasp_at_60_cm_is_not():
    report = reach_for_sizes(CONFIG, {"can": CAN, "tall": TALL})
    assert all(r.best_pitch_deg is not None for r in report["can"])
    assert all(r.best_pitch_deg is None for r in report["tall"])
    assert [r.label for r in report["can"]] == [label for label, _ in container_poses(CONFIG, CAN)]


def test_the_sweep_goes_below_the_scenes_tilt_floor_down_to_level():
    # 0.26 m above the pad the arm cannot point down 60 degrees, only nearly level.
    high = dataclasses.replace(CAN, grasp_height=0.26)
    grasp = next(
        r for r in reach_for_sizes(CONFIG, {"high": high})["high"] if r.label == "pad:grasp"
    )
    assert grasp.best_pitch_deg is not None
    assert grasp.best_pitch_deg < CONFIG.reach.min_pitch_deg


def test_the_common_pitch_is_one_that_reaches_every_pose_and_the_steepest_such():
    poses = container_poses(CONFIG, dataclasses.replace(CAN, grasp_height=0.17))
    common = steepest_common_pitch(poses)
    each = [
        r.best_pitch_deg
        for r in reach_for_sizes(CONFIG, {"b": dataclasses.replace(CAN, grasp_height=0.17)})["b"]
    ]
    # No pose can be reached steeper than its own best, so the common pitch is at most the least.
    assert common is not None and common <= min(each)
    assert common >= 0 and common % 5 == 0


def test_there_is_no_common_pitch_when_a_pose_is_out_of_reach_or_the_floor_is_too_steep():
    assert steepest_common_pitch(container_poses(CONFIG, TALL)) is None
    assert steepest_common_pitch(container_poses(CONFIG, CAN), floor_deg=90) is None
