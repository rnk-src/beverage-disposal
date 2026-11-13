"""Every pose the pick and place needs is within the arm's reach, at a tilt it can hold."""

import dataclasses

import pytest
from support.paths import ROOT

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.reach import reach_report

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
GRASP_Z = 0.10  # where the pads take the can, near the rim (2.4)


def test_every_pose_of_the_pick_and_place_is_reachable_within_the_allowed_tilt():
    report = reach_report(CONFIG)
    failed = [r.label for r in report if r.best_pitch_deg is None]
    assert not failed, f"out of reach at {CONFIG.reach.min_pitch_deg} degrees or steeper: {failed}"


def test_the_report_covers_the_pick_the_pad_the_bin_and_the_set_aside_area():
    labels = {r.label for r in reach_report(CONFIG)}
    for site in ["spawn_centre", "pad", "bin", "set_aside"]:
        assert any(label.startswith(site) for label in labels), site
    assert {"spawn_x_min", "spawn_x_max", "spawn_y_min", "spawn_y_max"} <= {
        label.rsplit(":", 1)[0] for label in labels
    }


def test_the_poses_are_where_the_layout_and_the_grasp_height_put_them():
    poses = {r.label: r.position for r in reach_report(CONFIG)}
    cx, cy = CONFIG.container_x, CONFIG.container_y
    assert poses["spawn_centre:grasp"] == pytest.approx((cx, cy, GRASP_Z))
    assert poses["pad:place"][2] == pytest.approx(CONFIG.pad.height + GRASP_Z)
    # Released with the can's bottom `release_clearance` above the wall top.
    expected = CONFIG.bin.height + CONFIG.reach.release_clearance + GRASP_Z
    assert poses["bin:release"][2] == pytest.approx(expected)


def test_a_bin_placed_out_of_reach_is_reported_by_name():
    far = dataclasses.replace(CONFIG, bin=dataclasses.replace(CONFIG.bin, center_x=0.55))
    failed = [r.label for r in reach_report(far) if r.best_pitch_deg is None]
    assert failed and all(label.startswith("bin") for label in failed)


def test_a_steeper_required_tilt_than_the_arm_can_hold_is_reported():
    strict = dataclasses.replace(
        CONFIG, reach=dataclasses.replace(CONFIG.reach, min_pitch_deg=90.0)
    )
    assert any(r.best_pitch_deg is None for r in reach_report(strict))
